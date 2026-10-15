"""
Evaluate multimodal models on mmstar_eu using vLLM offline inference.
MOST EFFICIENT approach - no server needed, direct model loading with batching.
"""
import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import List
from vllm import LLM, SamplingParams
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import ast
from collections import Counter
import os
import ast
from PIL import Image


# NOTE: Updated to use Qwen's specific image token wrapper instead of <image>
PROMPT_EN =  """You are a helpful assistant for answering questions about images. Use the provided image and question to answer concisely. Respond only with the answer in Basque.

Question: %s

Answer:"""

PROMPT_ES = """Eres un asistente útil para responder preguntas sobre imágenes. Utiliza la imagen y la pregunta proporcionadas para generar una respuesta concisa. Responde solo con la respuesta.

Pregunta: %s

Respuesta:"""


PROMPT_EU = """Irudiei buruzko galderak erantzuten dituen laguntzaile bat zara. Irudia eta galdera erabiliz, eman erantzun labur eta zehatza. Erantzunarekin bakarrik erantzun

Galdera: %s

Erantzuna:"""

def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a model on mmstar_eu using vLLM offline inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Maximum number of new tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for nucleus sampling")
    parser.add_argument("--min_p", type=float, default=0.0, help="Min-p for nucleus sampling (new parameter to avoid very low-probability tokens)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k for sampling (added to further control randomness)")
    parser.add_argument("--presence_penalty", type=float, default=2.0, help="Presence penalty to encourage diversity in answers")
    parser.add_argument("--repetition_penalty", type=float, default=2.0, help="Repetition penalty to discourage repetition")
    parser.add_argument("--dataset_name", type=str, default="HiTZ/mmstar_eu", help="Dataset name")
    parser.add_argument("--image_path", type=str, default=None, help="Dataset image path (if not using HF dataset with images)")
    parser.add_argument("--num_sample_dataset", type=int, default=300, help="Number of samples to load from the dataset")
    parser.add_argument("--output_file", type=str, default="results_offline.jsonl", help="Output file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--language", type=str, choices=["en", "eu", "eu_reviewed"], default="en", help="Language of the dataset (en, es, eu)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to evaluate (for testing)")
    return parser.parse_args()


def select_valid_samples(dataset, image_dir):
    # Count image occurrences
    image_counts = Counter()

    for example in dataset:
        img = example["image"]

        if isinstance(img, dict) and "path" in img:
            img_path = img["path"]
        else:
            img_path = img

        image_counts[img_path] += 1

    def is_valid(example):
        # -------------------------
        # 0. Image must be unique
        # -------------------------
        img = example["image"]

        if isinstance(img, dict) and "path" in img:
            img_path = img["path"]
        else:
            img_path = img

        if image_counts[img_path] != 1:
            return False

        # -------------------------
        # 1. Check image can be opened
        # -------------------------
        full_path = os.path.join(image_dir, img_path)

        try:
            Image.open(full_path).convert("RGB")
        except Exception:
            return False

        # -------------------------
        # 2. Parse conversation
        # -------------------------
        conv = ast.literal_eval(example["conversations"])

        human_turn = next(
            (c for c in conv if c.get("from") == "human"),
            None
        )

        if human_turn is None:
            return False

        question = human_turn.get("value", "")

        # -------------------------
        # 3. Must contain <image>
        # -------------------------
        if "<image>" not in question:
            return False

        # -------------------------
        # 4. Exclude emoji questions
        # -------------------------
        if "emoji" in question.lower():
            return False

        return True

    return dataset.filter(is_valid)


def load_dataset_from_hf(dataset_name, subset="default", split="test", num_samples=None, sampling="random", seed=42, image_path=None):
    """
    Load a Hugging Face dataset with optional sampling.

    Args:
        dataset_name (str): HF dataset name or path.
        subset (str or None): Dataset subset/config.
        split (str): Dataset split (train/val/test).
        num_samples (int or None): Number of samples to load.
        sampling (str): "first" or "random".
        seed (int): Random seed for shuffling.
        image_path (str or None): Path to the image directory (if not using HF dataset with images).

    Returns:
        datasets.Dataset
    """
    print(f"Loading {dataset_name} ({split} split)...")

    if "json" in dataset_name:
        return None

    # Load dataset
    if subset is None:
        ds = load_dataset(dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name, subset, split=split)

    ds = select_valid_samples(ds, image_path) if image_path else ds

    # Optional sampling
    if num_samples is not None:
        num_samples = min(num_samples, len(ds))

        if sampling == "random":
            ds = ds.shuffle(seed=seed)
        elif sampling != "first":
            raise ValueError("sampling must be 'first' or 'random'")

        ds = ds.select(range(num_samples))
    return ds

def prepare_inputs_with_images(dataset, batch_indices: List[int], language: str, image_path: str) -> List[List[dict]]:
    """Prepare batch of inputs with prompts and images using chat format"""
    inputs = []

    for idx in batch_indices:
        example = dataset[idx]

        # Get the human question (first entry) and extract the text value
        
        # Parse conversations if stored as a JSON string
        conversations = ast.literal_eval(example['conversations'])
        
        question_entry = next(c for c in conversations if c['from'] == 'human')
        question = question_entry['value'].replace('\\n<image>', '').replace('\n<image>', '').replace("Emann", "Eman").replace("emann", "eman").strip()

        if language == "en":
            prompt = PROMPT_EN % (question)
        elif language == "es":
            prompt = PROMPT_ES % (question)
        else:  # language == "eu"
            prompt = PROMPT_EU % (question)
        
        img = Image.open(os.path.join(image_path, example["image"])).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": img},
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
    )

    
    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset_from_hf(args.dataset_name, split="eu", image_path=args.image_path, num_samples=args.num_sample_dataset)

    # Prepare output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"Evaluating {len(dataset)} examples with batch size {args.batch_size}...")

    # Process in batches
    with open(output_path, 'w') as f:
        for start_idx in tqdm(range(0, len(dataset), args.batch_size)):
            end_idx = min(start_idx + args.batch_size, len(dataset))
            batch_indices = list(range(start_idx, end_idx))

            try:
                # Prepare batch
                messages_batch = prepare_inputs_with_images(dataset, batch_indices, args.language, args.image_path)

    
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
                    match = re.search(r"(?:Erantzuna|Answer)\s*:\s*(.+)", raw_text, re.IGNORECASE)
                    if match:
                        prediction = match.group(1).strip()
                    else:
                        prediction = raw_text.strip()

                    conversations = ast.literal_eval(example['conversations'])
        
                    ground_entry = next(c for c in conversations if c['from'] == 'gpt')
                    ground_truth = ground_entry['value'].strip()
        
                    question_entry = next(c for c in conversations if c['from'] == 'human')
                    question = question_entry['value'].replace('\\n<image>', '').replace('\n<image>', '').replace("Emann", "Eman").replace("emann", "eman").strip()
                                
                    '''
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
                    ''' 
                    # Store result
                    result = {
                        "idx": idx,
                        "question_id": example["id"],
                        "question": question,
                        "image_path":  example["image"],
                        "ground_truth": ground_truth,
                        "prediction": prediction
                    }
                    results.append(result)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                f.flush()

            except Exception as e:
                print(f"\nError processing batch {start_idx}-{end_idx}: {e}")
                for idx in batch_indices:
                    f.write(json.dumps({"idx": idx, "error": str(e)}) + "\n")
                f.flush()

    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = get_arguments()
    evaluate_model(args)