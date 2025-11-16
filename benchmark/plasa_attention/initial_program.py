"""
Adaptive Per-Layer Sparse Attention Implementation

This module implements sparse attention with layer-specific top-k values.
Based on research showing different layers specialize in different functions:
- Early layers: Local patterns, short-range dependencies
- Middle layers: Feature composition, functionally redundant
- Late layers: Global context consolidation, semantic abstraction

Key Innovation: Each layer has a different sparsity budget (k value) optimized
for its functional role in the transformer hierarchy.

References:
- "Learning to Skip the Middle Layers of Transformers" (2025)
- "Transformer Layers as Painters" - Emergence.ai (2025)
- DeepSeek-V3.2-Exp Lightning Indexer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum


class SparsitySchedule(Enum):
    """Predefined sparsity schedules for different hypotheses"""
    DENSE_BASELINE = "dense_baseline"
    UNIFORM_SPARSE = "uniform_sparse"
    DENSE_TO_SPARSE = "dense_to_sparse"
    AGGRESSIVE_MIDDLE = "aggressive_middle"
    PROGRESSIVE_SPARSE = "progressive_sparse"
    REVERSE_PROGRESSIVE = "reverse_progressive"


@dataclass
class LayerSparsityConfig:
    """Configuration for per-layer sparsity"""
    schedule_name: str
    layer_k_values: List[int]  # k value for each layer
    layer_k_ratios: List[float]  # k as fraction of sequence length
    description: str

    def get_k_for_layer(self, layer_idx: int, seq_len: int) -> int:
        """Get k value for a specific layer"""
        if layer_idx >= len(self.layer_k_ratios):
            # Default to last value if layer index exceeds config
            ratio = self.layer_k_ratios[-1]
        else:
            ratio = self.layer_k_ratios[layer_idx]

        k = int(seq_len * ratio)
        return max(1, min(k, seq_len))  # Clamp to [1, seq_len]


def create_sparsity_schedule(
    schedule: SparsitySchedule,
    n_layers: int,
    seq_len: int
) -> LayerSparsityConfig:
    """
    Create a sparsity schedule based on predefined patterns

    Args:
        schedule: Schedule type
        n_layers: Number of transformer layers
        seq_len: Sequence length

    Returns:
        LayerSparsityConfig with per-layer k values
    """
    if schedule == SparsitySchedule.DENSE_BASELINE:
        # All layers dense (no sparsity)
        ratios = [1.0] * n_layers
        description = "Baseline: All layers dense (k=L)"

    elif schedule == SparsitySchedule.UNIFORM_SPARSE:
        # All layers uniform 50% sparsity (Exp2 baseline)
        ratios = [0.5] * n_layers
        description = "Uniform: All layers k=L/2 (Exp2 baseline)"

    elif schedule == SparsitySchedule.DENSE_TO_SPARSE:
        # Conservative: Dense early, gradually sparse
        # Early (0-33%): Dense (k=L)
        # Middle (33-66%): Moderate sparse (k=L/2)
        # Late (66-100%): Light sparse (k=3L/4)
        ratios = []
        early_cutoff = n_layers // 3
        middle_cutoff = 2 * n_layers // 3

        for i in range(n_layers):
            if i < early_cutoff:
                ratios.append(1.0)  # Dense
            elif i < middle_cutoff:
                ratios.append(0.5)  # Moderate sparse
            else:
                ratios.append(0.75)  # Light sparse
        description = "Dense-to-Sparse: Early=Dense, Middle=L/2, Late=3L/4"

    elif schedule == SparsitySchedule.AGGRESSIVE_MIDDLE:
        # Based on redundancy research: Middle layers most sparse
        # Early: Moderate (k=L/2)
        # Middle: Aggressive (k=L/4) - most redundant
        # Late: Moderate (k=L/2)
        ratios = []
        early_cutoff = n_layers // 3
        middle_cutoff = 2 * n_layers // 3

        for i in range(n_layers):
            if i < early_cutoff:
                ratios.append(0.5)  # Moderate
            elif i < middle_cutoff:
                ratios.append(0.25)  # Aggressive sparse
            else:
                ratios.append(0.5)  # Moderate
        description = "Aggressive-Middle: Early=L/2, Middle=L/4, Late=L/2"

    elif schedule == SparsitySchedule.PROGRESSIVE_SPARSE:
        # Original hypothesis: Dense foundation, aggressive middle, moderate late
        # Early: Dense (k=L)
        # Middle: Aggressive (k=L/4)
        # Late: Moderate (k=L/2)
        ratios = []
        early_cutoff = n_layers // 3
        middle_cutoff = 2 * n_layers // 3

        for i in range(n_layers):
            if i < early_cutoff:
                ratios.append(1.0)  # Dense
            elif i < middle_cutoff:
                ratios.append(0.25)  # Aggressive sparse
            else:
                ratios.append(0.5)  # Moderate
        description = "Progressive-Sparse: Early=Dense, Middle=L/4, Late=L/2"

    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    # Compute actual k values
    k_values = [int(seq_len * ratio) for ratio in ratios]

    return LayerSparsityConfig(
        schedule_name=schedule.value,
        layer_k_values=k_values,
        layer_k_ratios=ratios,
        description=description
    )


class LightningIndexer(nn.Module):
    """
    Lightning Indexer for DeepSeek Sparse Attention

    Computes index scores I_{t,s} = Σ w_{t,j} · ReLU(q_{t,j} · k_s)

    Args:
        d_model: Model dimension
        indexer_heads: Number of indexer heads (H_I)
        indexer_dim: Dimension of indexer queries/keys (d_I)
        dropout: Dropout probability
    """
    def __init__(
        self,
        d_model: int,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.indexer_heads = indexer_heads
        self.indexer_dim = indexer_dim

        # Indexer query projection: h_t -> {q_{t,j}^I}
        self.q_proj = nn.Linear(d_model, indexer_heads * indexer_dim, bias=False)

        # Indexer key projection: h_s -> k_s^I
        self.k_proj = nn.Linear(d_model, indexer_dim, bias=False)

        # Indexer weights: w_{t,j}^I for each head
        self.w_proj = nn.Linear(d_model, indexer_heads, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute index scores between all pairs of tokens

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            index_scores: Index scores [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # Compute indexer queries: [batch, seq_len, indexer_heads, indexer_dim]
        queries = self.q_proj(x).reshape(batch_size, seq_len, self.indexer_heads, self.indexer_dim)

        # Compute indexer keys: [batch, seq_len, indexer_dim]
        keys = self.k_proj(x)

        # Compute indexer weights: [batch, seq_len, indexer_heads]
        weights = self.w_proj(x)

        # Compute dot products: q_{t,j} · k_s for all t, s, j
        dots = torch.einsum('bthd,bsd->bths', queries, keys)

        # Apply ReLU activation
        activated = F.relu(dots)

        # Weight each head: w_{t,j} · ReLU(q_{t,j} · k_s)
        weighted = activated * weights.unsqueeze(-1)

        # Sum across heads: Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)
        index_scores = weighted.sum(dim=2)

        return index_scores


class AdaptiveTopKSelector(nn.Module):
    """
    Adaptive Top-K Token Selection with per-layer k values

    Args:
        default_top_k: Default k value (can be overridden per forward pass)
    """
    def __init__(self, default_top_k: int = 512):
        super().__init__()
        self.default_top_k = default_top_k

    def forward(
        self,
        index_scores: torch.Tensor,
        top_k: Optional[int] = None,
        apply_causal_mask: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Select top-k tokens based on index scores

        Args:
            index_scores: Index scores [batch, seq_len_q, seq_len_k]
            top_k: Number of tokens to select (overrides default)
            apply_causal_mask: Whether to apply causal masking

        Returns:
            - top_k_mask: Boolean mask [batch, seq_len_q, seq_len_k]
            - top_k_indices: Indices of selected tokens [batch, seq_len_q, k]
            - stats: Dictionary with selection statistics
        """
        batch_size, seq_len_q, seq_len_k = index_scores.shape

        # Use provided k or default
        k = top_k if top_k is not None else self.default_top_k

        # Apply causal mask: token t can only attend to tokens <= t
        if apply_causal_mask:
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=index_scores.device),
                diagonal=1
            ).bool()
            index_scores = index_scores.masked_fill(causal_mask.unsqueeze(0), -1e9)

        # Select top-k indices for each query token
        actual_k = min(k, seq_len_k)
        top_k_values, top_k_indices = torch.topk(
            index_scores,
            k=actual_k,
            dim=-1,
            largest=True
        )

        # Create boolean mask from indices
        top_k_mask = torch.zeros_like(index_scores, dtype=torch.bool)
        top_k_mask.scatter_(2, top_k_indices, True)

        # Compute statistics
        sparsity = 1.0 - (top_k_mask.sum().item() / top_k_mask.numel())
        stats = {
            'sparsity': sparsity,
            'actual_k': actual_k,
            'k_ratio': actual_k / seq_len_k
        }

        return top_k_mask, top_k_indices, stats


class AdaptiveSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention with Adaptive Per-Layer Top-K

    Each layer can have a different sparsity level (k value) based on its
    functional role in the transformer hierarchy.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        layer_idx: Layer index (0-indexed)
        layer_top_k: Top-k value for this specific layer
        indexer_heads: Number of indexer heads
        indexer_dim: Dimension of indexer queries/keys
        dropout: Dropout probability
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        layer_idx: int,
        layer_top_k: int,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.layer_idx = layer_idx
        self.layer_top_k = layer_top_k

        # Main attention components
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = RotaryPositionalEmbeddings(dim=self.d_k, max_seq_len=max_seq_len, base=10000)
        self.dropout = dropout

        # Lightning indexer
        self.indexer = LightningIndexer(
            d_model=d_model,
            indexer_heads=indexer_heads,
            indexer_dim=indexer_dim,
            dropout=dropout
        )

        # Adaptive token selector
        self.selector = AdaptiveTopKSelector(default_top_k=layer_top_k)

        # Whether to use sparse attention
        self.use_sparse = True

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with adaptive sparse attention

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            return_stats: Whether to return selection statistics

        Returns:
            - output: Attention output [batch_size, seq_len, d_model]
            - stats: Selection statistics if return_stats=True
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)

        stats = None

        if self.use_sparse:
            # Compute index scores
            index_scores = self.indexer(x)

            # Select top-k tokens (using layer-specific k)
            top_k_mask, top_k_indices, selector_stats = self.selector(
                index_scores,
                top_k=self.layer_top_k,
                apply_causal_mask=True
            )

            # Create attention mask
            attn_mask = torch.zeros(
                batch_size, 1, seq_len, seq_len,
                device=x.device,
                dtype=Q.dtype
            )
            attn_mask = attn_mask.masked_fill(~top_k_mask.unsqueeze(1), float('-inf'))

            # Apply sparse attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0
            )

            if return_stats:
                stats = {
                    'layer_idx': self.layer_idx,
                    'layer_k': self.layer_top_k,
                    **selector_stats
                }
        else:
            # Dense attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0
            )

            if return_stats:
                stats = {
                    'layer_idx': self.layer_idx,
                    'layer_k': seq_len,
                    'sparsity': 0.0,
                    'k_ratio': 1.0
                }

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.w_o(attn_output)

        return output, stats

    def enable_sparse(self):
        """Enable sparse attention"""
        self.use_sparse = True

    def disable_sparse(self):
        """Disable sparse attention (use dense)"""
        self.use_sparse = False

    def update_layer_k(self, new_k: int):
        """Update the layer's top-k value dynamically"""
        self.layer_top_k = new_k
        self.selector.default_top_k = new_k


def print_schedule_info(config: LayerSparsityConfig, n_layers: int):
    """Print detailed information about a sparsity schedule"""
    print(f"\n{'='*80}")
    print(f"Sparsity Schedule: {config.schedule_name}")
    print(f"{'='*80}")
    print(f"Description: {config.description}")
    print(f"\nPer-Layer Configuration:")
    print(f"{'Layer':<10} {'k Ratio':<15} {'Function':<30}")
    print(f"{'-'*80}")

    for i in range(n_layers):
        ratio = config.layer_k_ratios[i] if i < len(config.layer_k_ratios) else config.layer_k_ratios[-1]

        # Categorize layer
        early_cutoff = n_layers // 3
        middle_cutoff = 2 * n_layers // 3
        if i < early_cutoff:
            function = "Early (local patterns)"
        elif i < middle_cutoff:
            function = "Middle (feature composition)"
        else:
            function = "Late (global context)"

        print(f"Layer {i:<4} {ratio:<15.2%} {function:<30}")
    print(f"{'='*80}\n")


# ================= Qwen3-Next Components (Fallback) =================

import torch.nn.functional as F
import math

class Qwen3NextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3NextMLP(nn.Module):
    def __init__(self, config=None, intermediate_size=512, hidden_size=128):
        super().__init__()
        if config:
            hidden_size = getattr(config, 'hidden_size', 128)
            intermediate_size = intermediate_size or hidden_size * 4
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            Qwen3NextMLP(intermediate_size=config.moe_intermediate_size, hidden_size=self.hidden_dim)
            for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros_like(hidden_states_flat)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if expert_mask.any():
                expert_input = hidden_states_flat[expert_mask]
                expert_output = expert_layer(expert_input)
                token_indices = expert_mask.nonzero(as_tuple=True)[0]
                expert_positions = (selected_experts[expert_mask] == expert_idx).nonzero(as_tuple=True)[1]
                weights = routing_weights[expert_mask, expert_positions].unsqueeze(-1)
                final_hidden_states[expert_mask] += expert_output * weights

        return final_hidden_states.view(batch_size, seq_len, hidden_dim)


class Qwen3NextRotaryEmbedding(nn.Module):
    """Fallback implementation of Qwen3NextRotaryEmbedding"""
    def __init__(self, config=None, dim=None, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        if config is not None:
            self.dim = config.head_dim if hasattr(config, 'head_dim') else config.hidden_size // config.num_attention_heads
            self.max_position_embeddings = config.max_position_embeddings
            self.base = config.rope_theta if hasattr(config, 'rope_theta') else 10000
        else:
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [batch_size, seq_len, ...]
        # position_ids: [batch_size, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3NextConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ==================== EXACT Exp3 PLASAQwen3 Implementation ====================

class PLASADecoderLayer(nn.Module):
    """
    Decoder layer that uses Per-Layer Adaptive Sparse Attention for ALL attention
    (replaces both full_attention and linear_attention)

    EXACT COPY from exp3_models.py lines 49-110
    """
    def __init__(self, config, layer_idx: int, layer_top_k: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Use Per-Layer Adaptive Sparse Attention with layer-specific k
        self.self_attn = AdaptiveSparseAttention(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            layer_idx=layer_idx,
            layer_top_k=layer_top_k,
            indexer_heads=getattr(config, 'indexer_heads', 4),
            indexer_dim=getattr(config, 'indexer_dim', 64),
            dropout=config.attention_dropout,
        )

        # MLP (same as Qwen3-Next)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3NextMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # DeepSeek Sparse Attention
        hidden_states, _ = self.self_attn(hidden_states)

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states

        return hidden_states


class PLASAQwen3Model(nn.Module):
    """
    Variant 2: All attention layers replaced with Per-Layer Adaptive Sparse Attention
    Uses PROGRESSIVE_SPARSE schedule: Early=Dense, Middle=L/4, Late=L/2

    EXACT COPY from exp3_models.py lines 202-277
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # Create sparsity schedule for per-layer k values
        sparsity_config = create_sparsity_schedule(
            schedule=SparsitySchedule.PROGRESSIVE_SPARSE,
            n_layers=config.num_hidden_layers,
            seq_len=config.max_position_embeddings
        )

        # Replace all layers with PLASA decoder layers with layer-specific k values
        self.layers = nn.ModuleList([
            PLASADecoderLayer(
                config,
                layer_idx,
                layer_top_k=sparsity_config.get_k_for_layer(layer_idx, config.max_position_embeddings)
            )
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3NextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return type('ModelOutput', (), {
            'last_hidden_state': hidden_states,
            'past_key_values': past_key_values,
        })()


class PLASAQwen3(nn.Module):
    """
    Variant 2: Per-Layer Adaptive Sparse Attention (PLASA)-Only Qwen3 (for CausalLM)

    EXACT COPY from exp3_models.py lines 280-309
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = PLASAQwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return type('CausalLMOutput', (), {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.past_key_values,
        })()


# Create PLASAModel wrapper for benchmark compatibility
class PLASAModel(nn.Module):
    """
    Wrapper for PLASAQwen3 that accepts evaluator-style parameters
    and converts them to a Qwen3NextConfig.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 32,
        intermediate_size: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()

        # Create Qwen3NextConfig from evaluator parameters (matching exp3)
        config = Qwen3NextConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=512,  # exp3 uses 512, not max_seq_len!
            rope_theta=10000.0,
            attention_dropout=dropout,
            hidden_dropout_prob=dropout,  # exp3 uses this
            partial_rotary_factor=1.0,  # exp3 uses this
            rms_norm_eps=rms_norm_eps,
            pad_token_id=0,
            # MoE parameters (EXACT match with exp3)
            num_experts=4,
            num_local_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=2,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=0,
            mlp_only_layers=[],
            # PLASA parameters (EXACT match with exp3)
            indexer_heads=4,
            indexer_dim=32,  # exp3 uses 32, not 64!
        )

        # Set attention implementation (required for full_attention layers)
        config._attn_implementation = "eager"

        # Create the actual PLASA model
        self.model = PLASAQwen3(config)

    def forward(self, input_ids, labels=None, **kwargs):
        return self.model(input_ids=input_ids, labels=labels, **kwargs)


__all__ = ['PLASAModel', 'AdaptiveSparseAttention', 'SparsitySchedule', 'create_sparsity_schedule']
