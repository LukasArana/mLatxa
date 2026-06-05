"""
Evaluate multimodal models on ocr-eu using vLLM offline inference.
MOST EFFICIENT approach - no server needed, direct model loading with batching.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import List
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm

PROMPT_EU = """Irudian agertzen den testua transkribatu. Ez gehitu azalpenik edo komentariorik, testu hutsa soilik itzuli.
Irudia:"""

def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a model on ocr-eu using vLLM offline inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for nucleus sampling")
    parser.add_argument("--min_p", type=float, default=0.0, help="Min-p for nucleus sampling")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k for sampling (-1 = disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--dataset_name", type=str, default="HiTZ/ocr-eu", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to evaluate")
    parser.add_argument("--output_file", type=str, default="results_ocr_offline.jsonl", help="Output file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to evaluate (for testing)")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace, normalize unicode."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate via edit distance."""
    ref = list(reference)
    hyp = list(hypothesis)
    return _edit_distance(ref, hyp) / max(len(ref), 1)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate via edit distance."""
    ref = reference.split()
    hyp = hypothesis.split()
    return _edit_distance(ref, hyp) / max(len(ref), 1)


def _edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein edit distance between two sequences."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[m]


def prepare_inputs_with_images(dataset, batch_indices: List[int]) -> List[List[dict]]:
    """Prepare batch of inputs with prompts and images using chat format."""
    inputs = []
    for idx in batch_indices:
        example = dataset[idx]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": example["image"]},
                    {"type": "text", "text": PROMPT_EU},
                ]
            }
        ]
        inputs.append(messages)
    return inputs


def evaluate_model(args):
    """Run evaluation using vLLM offline inference."""

    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        mm_encoder_tp_mode="data",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    print(f"Loading dataset: {args.dataset_name} (split={args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)

    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    total_cer = 0.0
    total_wer = 0.0
    total = 0

    print(f"Evaluating {len(dataset)} examples with batch size {args.batch_size}...")

    with open(output_path, "w") as f:
        for start_idx in tqdm(range(0, len(dataset), args.batch_size)):
            end_idx = min(start_idx + args.batch_size, len(dataset))
            batch_indices = list(range(start_idx, end_idx))

            try:
                messages_batch = prepare_inputs_with_images(dataset, batch_indices)

                outputs = llm.chat(
                    messages=messages_batch,
                    sampling_params=sampling_params,
                    chat_template_kwargs={"enable_thinking": False},
                )

                for idx, output in zip(batch_indices, outputs):
                    example = dataset[idx]
                    raw_text = output.outputs[0].text
                    # Strip any <think>...</think> blocks
                    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
                    prediction = raw_text.strip()
                    ground_truth = example["transcription"].strip()

                    pred_norm = normalize_text(prediction)
                    gt_norm = normalize_text(ground_truth)

                    cer = compute_cer(gt_norm, pred_norm)
                    wer = compute_wer(gt_norm, pred_norm)

                    total_cer += cer
                    total_wer += wer
                    total += 1

                    result = {
                        "idx": idx,
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "cer": cer,
                        "wer": wer,
                    }
                    results.append(result)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                f.flush()

            except Exception as e:
                print(f"\nError processing batch {start_idx}-{end_idx}: {e}")
                for idx in batch_indices:
                    f.write(json.dumps({"idx": idx, "error": str(e)}) + "\n")
                f.flush()

    mean_cer = (total_cer / total * 100) if total > 0 else 0
    mean_wer = (total_wer / total * 100) if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"Total examples: {total}")
    print(f"Mean CER: {mean_cer:.2f}%")
    print(f"Mean WER: {mean_wer:.2f}%")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = get_arguments()
    evaluate_model(args)
