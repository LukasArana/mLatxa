"""Evaluate text-only models on MMLU-Pro using vLLM offline inference + logprobs.

Same approach as mmstar_offline.py: no lmms-eval/harness dependency, direct
model loading with batching, scored both by extracted-letter exact match and
by which option letter got the highest first-token logprob.

Note: the official lmms-eval protocol for mmlu_pro is 5-shot chain-of-thought
with generate_until + regex extraction. This script intentionally trades that
away for a zero-shot direct-answer prompt, since the point of the logprob
score is "which option did the model consider most likely", not "can the
model reason its way to the letter via CoT". If you need the official
CoT-based number for a leaderboard comparison, keep using the harness for
this task; use this script for cheap/fast iteration during development.
"""
import argparse
import json
from pathlib import Path
from typing import List

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
from mcq_offline_utils import (  # noqa: E402
    add_common_vllm_args,
    build_interleaved_content,
    extract_mcq_answer,
    format_options_block,
    make_sampling_params,
    run_batched_chat,
    score_option_logprobs,
    strip_think_tags,
    summarize_mc_results,
)

from vllm import LLM
from datasets import load_dataset

MMLU_PRO_SUBJECTS = [
    "biology", "business", "chemistry", "computer science", "economics",
    "engineering", "health", "history", "law", "math", "other",
    "philosophy", "physics", "psychology",
]

PROMPT_TEMPLATE = """Answer the following multiple-choice question. Respond only with the letter of the correct option, nothing else.
Question: {question}
Options:
{options_block}
Answer:"""


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a model on MMLU-Pro using vLLM offline inference")
    add_common_vllm_args(parser, default_max_new_tokens=5)
    parser.add_argument("--dataset_name", type=str, default="TIGER-Lab/MMLU-Pro", help="Dataset name")
    parser.add_argument("--subject", type=str, default=None, choices=MMLU_PRO_SUBJECTS, help="Restrict to a single subject/category (default: all)")
    return parser.parse_args()


def prepare_inputs(dataset, batch_indices: List[int]) -> List[List[dict]]:
    inputs = []
    for idx in batch_indices:
        example = dataset[idx]
        options_block, _ = format_options_block(example["options"])
        prompt = PROMPT_TEMPLATE.format(question=example["question"], options_block=options_block)
        messages = [{"role": "user", "content": prompt}]
        inputs.append(messages)
    return inputs


def make_process_output_fn():
    def process_output(idx, example, output):
        _, valid_letters = format_options_block(example["options"])

        raw_text = strip_think_tags(output.outputs[0].text)
        prediction = raw_text.strip()
        ground_truth = example["answer"].strip().upper()

        extracted = extract_mcq_answer(prediction, choices=valid_letters)
        is_correct_exact = extracted == ground_truth

        option_log_probs = score_option_logprobs(output, valid_letters)
        best = max(option_log_probs, key=option_log_probs.get)
        is_correct_logprobs = best == ground_truth and option_log_probs[best] != float("-inf")

        return {
            "idx": idx,
            "category": example["category"],
            "question": example["question"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "extracted_answer": extracted,
            "option_log_probs": option_log_probs,
            "correct_exact_match": is_correct_exact,
            "correct_logprobs": is_correct_logprobs,
        }

    return process_output


def evaluate_model(args):
    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    sampling_params = make_sampling_params(args, stop=["\n"], logprobs=20)

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="test")
    if args.subject:
        dataset = dataset.filter(lambda x: x["category"] == args.subject)

    print(f"Evaluating {len(dataset)} examples with batch size {args.batch_size}...")
    results = run_batched_chat(
        llm=llm,
        sampling_params=sampling_params,
        dataset=dataset,
        batch_size=args.batch_size,
        prepare_inputs_fn=prepare_inputs,
        process_output_fn=make_process_output_fn(),
        output_path=args.output_file,
        chat_template_kwargs={"enable_thinking": False},
        limit=args.limit,
    )

    summarize_mc_results(results, args.output_file)

    # Per-category breakdown
    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)
    print("\nPer-category logprob accuracy:")
    for category, rows in sorted(by_category.items()):
        acc = sum(1 for r in rows if r["correct_logprobs"]) / len(rows) * 100
        print(f"  {category}: {acc:.2f}% ({len(rows)} examples)")


if __name__ == "__main__":
    args = get_arguments()
    evaluate_model(args)
