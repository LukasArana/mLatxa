"""Evaluate multimodal models on MMMU (validation split) using vLLM offline
inference, without the lmms-eval harness.

MMMU_val mixes two question types in the same dataset (doc["question_type"]):
  - "multiple-choice": scored the same way as mmstar/mmlu_pro/mmmu_pro --
    highest first-token logprob among the doc's option letters, plus a
    robust extracted-letter exact match as a secondary signal.
  - open-ended (free-form short answer): there is nothing to take logprobs
    over, so these are scored purely by generation + the official MMMU
    open-answer matcher (normalized substring/numeric match, vendored in
    _common/mmmu_open_ended.py from the official MMMU eval script).

Both question types are run in the same batched pass (one SamplingParams,
enough max_new_tokens for the longer open-ended answers); the scoring logic
branches per-doc afterwards.
"""
import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
from mcq_offline_utils import (  # noqa: E402
    ALL_CHOICE_LETTERS,
    add_common_vllm_args,
    build_interleaved_content,
    extract_mcq_answer,
    format_options_block,
    make_sampling_params,
    parse_options_field,
    run_batched_chat,
    score_option_logprobs,
    strip_think_tags,
)
from mmmu_open_ended import eval_open, parse_open_response  # noqa: E402

from vllm import LLM
from datasets import load_dataset

MC_POST_PROMPT = "Answer with the option's letter from the given choices directly."
OPEN_POST_PROMPT = "Answer the question using a single word or phrase."


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a model on MMMU validation using vLLM offline inference")
    # Open-ended answers need more room than a single letter.
    add_common_vllm_args(parser, default_max_new_tokens=32)
    parser.add_argument("--dataset_name", type=str, default="lmms-lab/MMMU", help="Dataset name")
    return parser.parse_args()


def prepare_inputs(dataset, batch_indices: List[int]) -> List[List[dict]]:
    inputs = []
    for idx in batch_indices:
        example = dataset[idx]
        if example["question_type"] == "multiple-choice":
            options = parse_options_field(example["options"])
            options_block, _ = format_options_block(options)
            full_text = f"{example['question']}\n{options_block}\n\n{MC_POST_PROMPT}"
        else:
            full_text = f"{example['question']}\n\n{OPEN_POST_PROMPT}"
        content = build_interleaved_content(full_text, example)
        inputs.append([{"role": "user", "content": content}])
    return inputs


def process_output(idx, example, output):
    raw_text = strip_think_tags(output.outputs[0].text)
    prediction = raw_text.strip()
    question_type = example["question_type"]

    result = {
        "idx": idx,
        "id": example.get("id"),
        "question_type": question_type,
        "ground_truth": example["answer"],
        "prediction": prediction,
    }

    if question_type == "multiple-choice":
        options = parse_options_field(example["options"])
        valid_letters = ALL_CHOICE_LETTERS[: len(options)]
        ground_truth = str(example["answer"]).strip().upper()

        extracted = extract_mcq_answer(prediction, choices=valid_letters)
        option_log_probs = score_option_logprobs(output, valid_letters)
        best = max(option_log_probs, key=option_log_probs.get)

        result.update({
            "extracted_answer": extracted,
            "option_log_probs": option_log_probs,
            "correct_exact_match": extracted == ground_truth,
            "correct_logprobs": best == ground_truth and option_log_probs[best] != float("-inf"),
        })
    else:
        parsed_pred = parse_open_response(prediction)
        result.update({
            "extracted_answer": parsed_pred,
            "option_log_probs": None,
            "correct_exact_match": eval_open(example["answer"], parsed_pred),
            "correct_logprobs": None,
        })

    return result


def summarize(results, output_path):
    mc_results = [r for r in results if r["question_type"] == "multiple-choice"]
    open_results = [r for r in results if r["question_type"] != "multiple-choice"]

    def acc(rows, key):
        rows = [r for r in rows if r.get(key) is not None]
        if not rows:
            return 0.0, 0
        return sum(1 for r in rows if r[key]) / len(rows) * 100, len(rows)

    overall_acc, overall_n = acc(results, "correct_exact_match")
    mc_exact_acc, mc_n = acc(mc_results, "correct_exact_match")
    mc_logprob_acc, _ = acc(mc_results, "correct_logprobs")
    open_acc, open_n = acc(open_results, "correct_exact_match")

    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"Total: {len(results)} (multiple-choice: {mc_n}, open-ended: {open_n})")
    print(f"Overall accuracy (extracted answer): {overall_acc:.2f}%")
    print(f"  Multiple-choice accuracy (extracted letter): {mc_exact_acc:.2f}%")
    print(f"  Multiple-choice accuracy (highest logprob):  {mc_logprob_acc:.2f}%")
    print(f"  Open-ended accuracy (normalized match):      {open_acc:.2f}%")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


def evaluate_model(args):
    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        mm_encoder_tp_mode="data",
        trust_remote_code=True,
    )
    sampling_params = make_sampling_params(args, stop=["\n"], logprobs=20)

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="validation")

    print(f"Evaluating {len(dataset)} examples with batch size {args.batch_size}...")
    results = run_batched_chat(
        llm=llm,
        sampling_params=sampling_params,
        dataset=dataset,
        batch_size=args.batch_size,
        prepare_inputs_fn=prepare_inputs,
        process_output_fn=process_output,
        output_path=args.output_file,
        chat_template_kwargs={"enable_thinking": False},
        limit=args.limit,
    )

    summarize(results, args.output_file)


if __name__ == "__main__":
    args = get_arguments()
    evaluate_model(args)
