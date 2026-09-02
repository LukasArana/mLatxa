"""Evaluate multimodal models on MMMU-Pro using vLLM offline inference + logprobs.

Same approach as mmstar_offline.py, extended to MMMU-Pro's two relevant
variants:
  - "standard": question + options + up to 7 images, referenced via
    "<image N>" tokens inside the question text. All questions are
    multiple-choice (10-way, A-J).
  - "vision": question and options are rendered *inside* a single image;
    there is no separate options field. We still score the highest-logprob
    letter among A-J, since the vision-only protocol always renders 10
    options (per the official MMMU-Pro design).

("composite" is intentionally not supported here, matching the harness's own
mmmu_pro.yaml group, which drops composite from the standard eval sweep.)
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
    summarize_mc_results,
)

from vllm import LLM
from datasets import load_dataset

POST_PROMPT = "Answer with the option letter from the given choices directly."
VISION_PROMPT = "Question: analyze the image and answer the associated question."


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a model on MMMU-Pro using vLLM offline inference")
    add_common_vllm_args(parser, default_max_new_tokens=5)
    parser.add_argument("--dataset_name", type=str, default="MMMU/MMMU_Pro", help="Dataset name")
    parser.add_argument("--variant", type=str, default="standard", choices=["standard", "vision"], help="MMMU-Pro variant to evaluate")
    return parser.parse_args()


def prepare_inputs(dataset, batch_indices: List[int], variant: str) -> List[List[dict]]:
    inputs = []
    for idx in batch_indices:
        example = dataset[idx]
        if variant == "standard":
            options = parse_options_field(example["options"])
            options_block, _ = format_options_block(options)
            full_text = f"{example['question']}\n{options_block}\n\n{POST_PROMPT}"
            content = build_interleaved_content(full_text, example)
        else:  # vision
            content = [
                {"type": "image_pil", "image_pil": example["image"].convert("RGB")},
                {"type": "text", "text": f"{VISION_PROMPT}\n{POST_PROMPT}"},
            ]
        inputs.append([{"role": "user", "content": content}])
    return inputs


def valid_letters_for(example, variant: str) -> List[str]:
    if variant == "standard":
        options = parse_options_field(example["options"])
        return ALL_CHOICE_LETTERS[: len(options)]
    # Vision variant renders all 10 options inside the image itself.
    return ALL_CHOICE_LETTERS[:10]


def make_process_output_fn(variant: str):
    def process_output(idx, example, output):
        valid_letters = valid_letters_for(example, variant)

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
            "id": example.get("id"),
            "subject": example.get("subject"),
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
        mm_encoder_tp_mode="data",
        trust_remote_code=True,
    )
    sampling_params = make_sampling_params(args, stop=["\n"], logprobs=20)

    dataset_config = "standard (10 options)" if args.variant == "standard" else "vision"
    print(f"Loading dataset: {args.dataset_name} [{dataset_config}]")
    dataset = load_dataset(args.dataset_name, dataset_config, split="test")

    print(f"Evaluating {len(dataset)} examples with batch size {args.batch_size}...")
    results = run_batched_chat(
        llm=llm,
        sampling_params=sampling_params,
        dataset=dataset,
        batch_size=args.batch_size,
        prepare_inputs_fn=lambda ds, idxs: prepare_inputs(ds, idxs, args.variant),
        process_output_fn=make_process_output_fn(args.variant),
        output_path=args.output_file,
        chat_template_kwargs={"enable_thinking": False},
        limit=args.limit,
    )

    summarize_mc_results(results, args.output_file)

    by_subject = {}
    for r in results:
        by_subject.setdefault(r["subject"], []).append(r)
    print("\nPer-subject logprob accuracy:")
    for subject, rows in sorted(by_subject.items(), key=lambda kv: str(kv[0])):
        acc = sum(1 for r in rows if r["correct_logprobs"]) / len(rows) * 100
        print(f"  {subject}: {acc:.2f}% ({len(rows)} examples)")


if __name__ == "__main__":
    args = get_arguments()
    evaluate_model(args)
