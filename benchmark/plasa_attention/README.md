# PLASA Attention Benchmark

Per-Layer Adaptive Sparse Attention (PLASA) optimization benchmark based on the latest research (November 2025).

## Overview

This benchmark evaluates implementations of Per-Layer Adaptive Sparse Attention (PLASA), which uses progressive sparsity scheduling based on transformer layer specialization research. The goal is to achieve the lowest validation perplexity on a 4-layer transformer trained on WikiText-2.

## Files

- **initial_program.py**: Complete PLASA implementation with Lightning Indexer and progressive sparsity
- **evaluator.py**: Training harness that trains a 4-layer model for 1000 steps and evaluates performance
- **initial_proposal.txt**: Detailed benchmark specifications and research context
- **README.md**: This file

## Quick Start

```bash
# Run the benchmark
cd benchmark/plasa_attention
python evaluator.py initial_program.py
```

## What is PLASA?

PLASA (Per-Layer Adaptive Sparse Attention) implements the DeepSeek Sparse Attention approach with layer-specific sparsity levels:

- **Layer 0 (Early)**: Dense attention (k=L) - captures local patterns
- **Layer 1-2 (Middle)**: Aggressive sparse (k=L/4) - functionally redundant layers
- **Layer 3 (Late)**: Moderate sparse (k=L/2) - consolidates global context

### Mathematical Formulation

**Lightning Indexer** computes relevance scores:
```
I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j}^I · k_s^I)
```

**Top-K Selection** identifies most relevant tokens:
```
S_t = TopK_k({ I_{t,s} : 1 ≤ s ≤ t })
```

**Sparse Attention** applies full attention only on selected tokens.

## Architecture

- **Layers**: 4 transformer layers (all using PLASA)
- **Hidden Dim**: 128
- **Attention Heads**: 4
- **Sequence Length**: 128
- **Parameters**: ~1.5M (including indexer)

## Training

- **Dataset**: WikiText-2 (2M tokens)
- **Steps**: 1000
- **Batch Size**: 2
- **Learning Rate**: 3e-4 (AdamW)
- **Optimizer**: AdamW with gradient clipping (1.0)

## Scoring

**Score = 1 / perplexity** (higher is better)

Additional metrics:
- Validation perplexity (lower is better)
- Validation accuracy (higher is better)
- Training loss

## Baseline Performance

The initial implementation achieves on WikiText-2:
- **Perplexity**: ~300-400 (expected range)
- **Accuracy**: ~25-30%
- **Score**: ~0.0025-0.0033

**Note**: WikiText-2 is significantly harder than synthetic educational datasets. The same PLASA architecture achieves ~72-80 perplexity on cosmopedia-v2, demonstrating that dataset difficulty has a ~4-5x impact on perplexity for small models

## Research Context

Based on cutting-edge research from September-November 2025:

### Layer Specialization (arXiv:2510.17469, Oct 2025)
- Early layers: Pattern recognition and memorization
- Middle layers: Consolidation (with functional redundancy)
- Late layers: Compositional reasoning and global context

### Dynamic Attention Mask (Oct 2025)
- Per-layer and per-head dynamic sparse patterns
- Context-aware sparsity structures
- No retraining required

### DeepSeek Sparse Attention - Lightning Indexer (Nov 2025)
- Fast FP8 token selection
- Two-stage: approximate indexer → exact attention
- 50% inference cost reduction, 30-40% memory savings

## Optimization Opportunities

Potential areas for improvement:

1. **Indexer Architecture**
   - Number of heads (currently 4)
   - Dimensionality (currently 32)
   - Activation functions (currently ReLU)

2. **Sparsity Schedules**
   - Alternative schedules (AGGRESSIVE_MIDDLE, DENSE_TO_SPARSE)
   - Dynamic k based on input
   - Learned threshold adaptation

3. **Training Techniques**
   - Weight initialization strategies
   - Regularization (dropout, gradient clipping)
   - Learning rate schedules

4. **Efficiency Optimizations**
   - Quantization (FP8 for indexer)
   - Sparse kernels
   - Fused operations

## Implementation Requirements

- Must use progressive sparsity principle (different k per layer)
- Must implement Lightning Indexer concept
- Model architecture fixed (4 layers, 128 dim, 4 heads)
- Training budget fixed (1000 steps)
- Self-contained in initial_program.py (PyTorch only)

## Comparison to Exp3

This benchmark is derived from `exp3_plasa_gdn_hybrid`, which showed:
- PLASA achieved 51.69% accuracy, 73.81 perplexity
- 18.4% better than full attention
- 33.9% better than uniform sparse attention
- 74% faster training than hybrid configurations

## References

- **DeepSeek Sparse Attention** (2025): Lightning Indexer, FP8 quantization
- **Dynamic Attention Mask** (GitHub: ResponsibleAILab/DAM, Oct 2025)
- **Layer Specialization** (arXiv:2510.17469, Oct 2025)
- **Transformer Layers as Painters** (Emergence.ai, Aug 2024-2025)

## License

Part of the alpha-research benchmark suite.
