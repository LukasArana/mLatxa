"""
Evaluate multimodal models on mmstar_eu using vLLM server with OpenAI client.
More efficient than the original script - sends images properly and uses the OpenAI client.
"""
import argparse
import base64
from io import BytesIO
from typing import List, Dict
import json
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from PIL import Image
from tqdm import tqdm


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a model on the mmstar_eu dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate")
    parser.add_argument("--server_url", type=str, required=True, help="URL of the vllm server")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    parser.add_argument("--dataset_name", type=str, default="HiTZ/mmstar_eu", help="Name of the dataset")
    parser.add_argument("--output_file", type=str, default="results.jsonl", help="Output file for results")
    parser.add_argument("--num_few_shot", type=int, default=0, help="Number of few-shot examples (0 for zero-shot)")
    return parser.parse_args()


def pil_to_data_uri(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI for vision model input"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def create_few_shot_messages(dataset, n: int = 5) -> List[Dict]:
    """Create few-shot examples as message history (only if n > 0)"""
    if n == 0:
        return []

    messages = []
    for i in range(min(n, len(dataset))):
        example = dataset[i]
        # Add user message with image
        image_uri = pil_to_data_uri(example["image"])
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": example["question"]}
            ]
        })
        # Add assistant answer
        messages.append({
            "role": "assistant",
            "content": example["answer"]
        })
    return messages


def evaluate_model(args):
    """Run evaluation on the dataset"""
    # Initialize OpenAI client for vLLM server
    client = OpenAI(
        base_url=args.server_url,
        api_key="token-abc123",  # vLLM doesn't validate this
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")

    # Create few-shot examples once (outside the loop)
    print(f"Creating {args.num_few_shot} few-shot examples...")
    few_shot_messages = create_few_shot_messages(dataset, args.num_few_shot)

    # Prepare output file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    correct = 0
    total = 0

    print(f"Evaluating {len(dataset)} examples...")

    # Evaluate each example
    with open(output_path, 'w') as f:
        for idx, example in enumerate(tqdm(dataset)):
            try:
                # Prepare messages for this query
                messages = few_shot_messages.copy()

                # Add current example
                image_uri = pil_to_data_uri(example["image"])
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": example["question"]}
                    ]
                })

                # Get model response
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                prediction = response.choices[0].message.content.strip()
                ground_truth = example["answer"].strip()

                # Simple exact match evaluation
                is_correct = prediction.lower() == ground_truth.lower()
                if is_correct:
                    correct += 1
                total += 1

                # Store result
                result = {
                    "idx": idx,
                    "question": example["question"],
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "correct": is_correct,
                }
                results.append(result)

                # Write to file incrementally
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

            except Exception as e:
                print(f"\nError processing example {idx}: {e}")
                result = {
                    "idx": idx,
                    "question": example.get("question", ""),
                    "error": str(e)
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

    # Print final results
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = get_arguments()
    evaluate_model(args)
