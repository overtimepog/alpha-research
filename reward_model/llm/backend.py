import re
import os
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict

from vllm import LLM, SamplingParams
from openai import OpenAI
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# SCORING_PROMPT = f"""
# You are an expert reviewer tasked with evaluating the quality of a research abstract. 
# Your goal is to assign a score between 1 and 10 based on the abstract's clarity, novelty, technical rigor, and potential impact. Here are the criteria:
# 1. Read the following abstract carefully and provide a score from 1 to 10. 
# 2. Score 6 means slightly higher than the boardline, 5 is slightly lower than the boardline.
# Write the score in the {BOX}.
# **idea**:
# """


BOX=r"\boxed{}"
SYSTEM_PROMPT = "You are an expert reviewer tasked with evaluating the quality of a research proposal. "
SCORING_PROMPT = f"""
Your goal is to assign a score between 1 and 10 based on the proposal's clarity, novelty, technical rigor, and potential impact. Here are the criteria:
1. Read the following proposal carefully and provide a score from 1 to 10. 
2. Score 6 means slightly higher than the boardline, 5 is slightly lower than the boardline.
Write the score in the {BOX}.
**idea**:

"""

def parse_score_from_text(text: str) -> float:
    match = re.search(r'\\boxed\{(\d*\.?\d*)\}', text)
    if match:
        try:
            score = float(match.group(1))
            if 0 <= score <= 10:
                return score
        except ValueError:
            pass
    return -1.0  


def score_abstracts_with_vllm(data: List[Dict], model_name: str) -> List[Dict]:

    llm = LLM(model=model_name, gpu_memory_utilization=0.95)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": SCORING_PROMPT + item["title"] + "\n" + item["abstract"]
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  
        )
        for item in data
    ]
    
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=1000, 
    )

    outputs = llm.generate(prompts, sampling_params)

    print(prompts[0])
    print(outputs[0].outputs[0].text)

    results = []
    for output, item in zip(outputs, data):
        output_text = output.outputs[0].text.strip()
        score = parse_score_from_text(output_text)
        results.append({
            "score": score,
            "evaluation": output_text,
            "abstract": item["abstract"],
            "avg_rating": item["avg_rating"]
        })

    return results

def load_processed_ids(jsonl_file: str) -> set:
    """
    Loads the IDs of already processed abstracts from the JSONL file.

    Args:
        jsonl_file: Path to the JSONL file.

    Returns:
        Set of processed abstract titles.
    """
    processed_ids = set()
    if os.path.exists(jsonl_file):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get('score', -1.0) != -1.0:  # Only include valid scores
                        processed_ids.add(data['title'])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {jsonl_file}")
    return processed_ids

def write_result_to_jsonl(result: Dict, jsonl_file: str):
    """
    Writes a single result to the JSONL file if the score is valid.

    Args:
        result: Dictionary containing the result to write.
        jsonl_file: Path to the JSONL file.
    """
    if result['score'] != -1.0:  # Only write valid scores
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def score_abstracts_with_api(data: List[Dict], 
                             jsonl_file: str, 
                             model_name: str = "deepseek-chat", 
                             api_key: str = "sk-2c3f1f58031b4b86afdb6a8192ea02e2", 
                             base_url: str = "https://api.deepseek.com",
                             max_retries: int = 50, 
                             retry_delay: int = 5
                             ) -> List[Dict]:
    """
    Scores research proposals using OpenAI's API, writing valid results to a JSONL file incrementally.
    Resumes from the last successfully processed abstract (with valid score) in the JSONL file.

    Args:
        data: List of dictionaries containing 'title', 'abstract', and 'avg_rating'.
        jsonl_file: Path to the JSONL file for storing results.
        model_name: OpenAI model to use (default: 'gpt-4o').
        api_key: OpenAI API key (if not set, assumes it's configured in environment).
        max_retries: Maximum number of retries per abstract (default: 5).
        retry_delay: Seconds to wait between retries (default: 5).

    Returns:
        List of dictionaries with 'title', 'score', 'evaluation', 'abstract', and 'avg_rating'.
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Load already processed abstracts (with valid scores) to skip them
    processed_ids = load_processed_ids(jsonl_file)
    results = []

    # Filter out already processed abstracts
    data_to_process = [item for item in data if item['title'] not in processed_ids]
    print(f"Total abstracts: {len(data)}, To process: {len(data_to_process)}, Already processed: {len(processed_ids)}")

    # Prepare prompts for remaining abstracts
    prompts = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SCORING_PROMPT + item["title"] + "\n" + item["abstract"]}
        ]
        for item in data_to_process
    ]

    for prompt, item in zip(prompts, data_to_process):
        retries = 0
        score = -1.0
        output_text = ""

        # Keep retrying until a valid score is obtained or max_retries is reached
        while score == -1.0 and retries < max_retries:
            response = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                temperature=0,
                max_tokens=1000,
                top_p=1.0
            )
            output_text = response.choices[0].message.content.strip()
            score = parse_score_from_text(output_text)
            
            if score == -1.0:
                retries += 1
                print(f"Invalid score for abstract: {item['title']}, Retry {retries}/{max_retries}")
                time.sleep(retry_delay)  # Wait before retrying
            else:
                print(f"Prompt: {prompt}")
                print(f"Output: {output_text}")

        # Create result dictionary
        result = {
            "title": item["title"],
            "score": score,
            "evaluation": output_text,
            "abstract": item["abstract"],
            "avg_rating": item["avg_rating"]
        }

        # Write result to JSONL file only if score is valid
        write_result_to_jsonl(result, jsonl_file)
        results.append(result)

        if score == -1.0:
            print(f"Failed to get valid score for abstract: {item['title']} after {max_retries} retries")

    # Load previously processed results from JSONL to include in return
    if processed_ids:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    if result['title'] in processed_ids:
                        results.append(result)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {jsonl_file}")

    return results

if __name__ == '__main__':
    abst = """
Test-time scaling is a promising new approach to language modeling that uses extra test-time compute to improve performance. Recently, OpenAI's o1 model showed this capability but did not publicly share its methodology, leading to many replication efforts. We seek the simplest approach to achieve test-time scaling and strong reasoning performance. First, we curate a small dataset s1K of 1,000 questions paired with reasoning traces relying on three criteria we validate through ablations: difficulty, diversity, and quality. Second, we develop budget forcing to control test-time compute by forcefully terminating the model's thinking process or lengthening it by appending "Wait" multiple times to the model's generation when it tries to end. This can lead the model to double-check its answer, often fixing incorrect reasoning steps. After supervised finetuning the Qwen2.5-32B-Instruct language model on s1K and equipping it with budget forcing, our model s1-32B exceeds o1-preview on competition math questions by up to 27% (MATH and AIME24). Further, scaling s1-32B with budget forcing allows extrapolating beyond its performance without test-time intervention: from 50% to 57% on AIME24. """
    title = "s1: Simple test-time scaling"
    data = [{"title": title, "abstract": abst, "avg_rating": 0}]
    print(score_abstracts_with_vllm(data, '/data/zhuotaodeng/yzj/alpha-research/model/qwen25_grm_iclr/checkpoint-240'))
    # print(score_abstracts_with_vllm(data, '/data/zhuotaodeng/yzj/download_from_modelscope/Qwen/Qwen3-8B'))