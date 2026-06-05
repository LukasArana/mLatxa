"""
Evaluate multimodal models on mmstar_eu using vLLM offline inference.
MOST EFFICIENT approach - no server needed, direct model loading with batching.
"""
import argparse
import json
import re
from pathlib import Path
from typing import List
from vllm import LLM, SamplingParams
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# NOTE: Updated to use Qwen's specific image token wrapper instead of <image>
PROMPT_EN = """You are a helpful assistant for answering questions about images. Use the provided image and question to generate a concise answer. Respond only by providing the letter of the option (A, B, C, or D) that you think is correct.
Question: %s"""
PROMPT_ES = """Eres un asistente útil para responder preguntas sobre imágenes. Utiliza la imagen y la pregunta proporcionadas para generar una respuesta concisa. Responde solo proporcionando la letra de la opción (A, B, C o D) que creas que es correcta.
Pregunta: %s"""
PROMPT_EU = """Irudiei buruzko galderak erantzuten dituen laguntzaile bat zara. Lau aukera emango zaizkizu eta letra bakarreko erantzun bat sortu behar duzu, aukera zuzenaren letra bakarra soilik sortuz (A, B, C edo D).
Galdera: %s"""

def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a model on mmstar_eu using vLLM offline inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Maximum number of new tokens") # Lowered from 100 since we only want 1 letter
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for nucleus sampling")
    parser.add_argument("--min_p", type=float, default=0.0, help="Min-p for nucleus sampling (new parameter to avoid very low-probability tokens)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k for sampling (added to further control randomness)")
    parser.add_argument("--presence_penalty", type=float, default=2.0, help="Presence penalty to encourage diversity in answers")
    parser.add_argument("--repetition_penalty", type=float, default=2.0, help="Repetition penalty to discourage repetition")
    parser.add_argument("--dataset_name", type=str, default="HiTZ/mmstar_eu", help="Dataset name")
    parser.add_argument("--output_file", type=str, default="results_offline.jsonl", help="Output file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--language", type=str, choices=["en", "eu", "eu_reviewed"], default="en", help="Language of the dataset (en, es, eu)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to evaluate (for testing)")
    return parser.parse_args()


def prepare_inputs_with_images(dataset, batch_indices: List[int], language: str) -> List[List[dict]]:
    """Prepare batch of inputs with prompts and images using chat format"""
    inputs = []

    for idx in batch_indices:
        example = dataset[idx]
        if language == "en":
            prompt = PROMPT_EN % (example['question'])
        elif language == "es":
            prompt = PROMPT_ES % (example['question'])
        else:  # language == "eu"
            prompt = PROMPT_EU % (example['question'])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": example["image"]},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        inputs.append(messages)

    return inputs


def evaluate_model(args):
    """Run evaluation using vLLM offline inference"""

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
        max_tokens=args.max_new_tokens,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        stop=["\n"],
        logprobs=20,
    )

    if args.language == "eu":
        split = "eu"
    elif args.language == "en":
        split = "en"
    elif args.language == "eu_reviewed":
        split = "eu_reviewed"
    else:
        print(f"Warning: The dataset does not have a separate {args.language} split, using 'en' split for {args.language} evaluation.")
        exit()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split=split)

    # Prepare output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    correct_exact = 0
    correct_logprobs = 0
    total = 0

    print(f"Evaluating {len(dataset)} examples with batch size {args.batch_size}...")

    # Process in batches
    with open(output_path, 'w') as f:
        for start_idx in tqdm(range(0, len(dataset), args.batch_size)):
            end_idx = min(start_idx + args.batch_size, len(dataset))
            batch_indices = list(range(start_idx, end_idx))

            try:
                # Prepare batch
                messages_batch = prepare_inputs_with_images(dataset, batch_indices, args.language)

                # Run inference as a batch
                outputs = llm.chat(
                    messages=messages_batch,
                    sampling_params=sampling_params,
                    chat_template_kwargs={"enable_thinking": False},
                )

                # Process results
                for i, (idx, output) in enumerate(zip(batch_indices, outputs)):
                    example = dataset[idx]
                    raw_text = output.outputs[0].text
                    # Mirror --reasoning-parser qwen3: strip any <think>...</think> blocks
                    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
                    prediction = raw_text.strip()
                    ground_truth = example["answer"].strip().upper()

                    # 1. Evaluating exact match
                    is_correct_exact = prediction.upper() == ground_truth
                    if is_correct_exact:
                        correct_exact += 1
                    total += 1

                    # 2. Evaluating log probabilities for each option
                    option_log_probs = {"A": float("-inf"), "B": float("-inf"), "C": float("-inf"), "D": float("-inf")}

                    log_probs_list = output.outputs[0].logprobs
                    if log_probs_list and log_probs_list[0] is not None:
                        first_token_logprobs = log_probs_list[0]

                        for token_id, logprob_obj in first_token_logprobs.items():
                            token_str = logprob_obj.decoded_token.strip().upper()
                            if token_str in option_log_probs:
                                option_log_probs[token_str] = max(option_log_probs[token_str], logprob_obj.logprob)

                        # Check if the ground truth had the highest logprob among A, B, C, D
                        best_option = max(option_log_probs, key=option_log_probs.get)
                        if best_option == ground_truth and option_log_probs[best_option] != float("-inf"):
                            correct_logprobs += 1

                    # Store result
                    result = {
                        "idx": idx,
                        "question": example["question"],
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "option_log_probs": option_log_probs,
                        "correct_exact_match": is_correct_exact,
                    }
                    results.append(result)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                f.flush()

            except Exception as e:
                print(f"\nError processing batch {start_idx}-{end_idx}: {e}")
                for idx in batch_indices:
                    f.write(json.dumps({"idx": idx, "error": str(e)}) + "\n")
                f.flush()

    # Print results (Fixed variable names here)
    accuracy_exact = (correct_exact / total * 100) if total > 0 else 0
    accuracy_logprobs = (correct_logprobs / total * 100) if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"Total: {total}")
    print(f"Correct (Exact Match): {correct_exact} ({accuracy_exact:.2f}%)")
    print(f"Correct (Highest Logprob): {correct_logprobs} ({accuracy_logprobs:.2f}%)")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = get_arguments()
    evaluate_model(args)