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
"""
PROMPT_EN =  Your task is to generate a highly detailed, deep, and comprehensive description of the provided image, following the guidelines of the DOCCI (Description of Detailed and Complex Images) standard. Do not provide a simple summary; instead, detail all elements in the image meticulously.

Please follow these rules:

1. **Overall Scene:** Begin by clearly describing the main subject and context of the image.
2. **Spatial Layout:** Describe the placement of elements precisely. What is in the foreground, midground, and background? Use exact directions (on the left, on the right, top, bottom) to explain the relationships between objects.
3. **Visual Details:** Specify the attributes of each object: colors, shapes, sizes, textures, and materials. If there are people or animals, describe their postures, gestures, and clothing.
4. **Text Transcription (OCR):** If there is readable text, signs, or brands in the image, transcribe them word for word.
5. **Strict Objectivity:** Describe only what can be seen in the image. Do not invent information, do not make hypotheses, and avoid subjective opinions or metaphors.

Write continuous, natural, and well-structured paragraph in Basque.""



PROMPT_EU = Zure zeregina emandako irudiaren deskribapen oso zehatza, sakona eta zabala sortzea da, DOCCI (Description of Detailed and Complex Images) estandarraren gidalerroei jarraituz. Ez egin laburpen sinple bat; aitzitik, irudiko elementu guztiak zehatz-mehatz azaldu. 

Mesedez, jarraitu ondorengo arau hauek:

1. **Eszena Orokorra:** Hasi irudiaren gai nagusia eta testuingurua argi eta garbi deskribatuz.
2. **Banaketa Espaziala:** Deskribatu elementuen kokapena modu zehatzean. Zer dago aurrealdean, erdialdean eta atzealdean? Erabili norabide zehatzak (ezkerraldean, eskuinaldean, goian, behean) objektuen arteko erlazioak azaltzeko.
3. **Xehetasun Bisualak:** Zehaztu objektu bakoitzaren atributuak: koloreak, formak, tamainak, testurak eta materialak. Pertsonak edo animaliak badaude, deskribatu haien jarrerak, keinuak eta jantziak.
4. **Testuaren Transkripzioa (OCR):** Irudian irakur daitekeen testurik, seinalerik edo markarik badago, transkribatu hitzez hitz.
5. **Objektibotasun Zorrotza:** Deskribatu soilik irudian ikus daitekeena. Ez asmatu informaziorik, ez egin hipotesirik, eta saihestu iritzi subjektiboak edo metaforak.

Idatzi deskribapena parrafo jarraitu, natural eta ondo egituratu batean
"""

#PROMPT_EN = """Describe the provided image in detail in Basque."""

PROMPT_EN = """Describe in a paragraph the provided image."""

PROMPT_EU= """Deskribatu parrafo batean emandako irudia."""





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
    parser.add_argument("--num_sample_dataset", type=int, default=300, help="Number of samples to load from the dataset")
    parser.add_argument("--output_file", type=str, default="results_offline.jsonl", help="Output file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--language", type=str, choices=["en", "eu", "eu_reviewed"], default="en", help="Language of the dataset (en, es, eu)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to evaluate (for testing)")
    return parser.parse_args()


def load_dataset_from_hf(dataset_name, subset="default", split="test", num_samples=None, sampling="random", seed=42):
    """
    Load a Hugging Face dataset with optional sampling.

    Args:
        dataset_name (str): HF dataset name or path.
        subset (str or None): Dataset subset/config.
        split (str): Dataset split (train/val/test).
        num_samples (int or None): Number of samples to load.
        sampling (str): "first" or "random".
        seed (int): Random seed for shuffling.

    Returns:
        datasets.Dataset
    """
    print(f"Loading {dataset_name} ({split} split)...")

    if "json" in dataset_name:
        return None

    # Load dataset
    if subset is None:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    else:
        ds = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)

    # Optional sampling
    if num_samples is not None:
        num_samples = min(num_samples, len(ds))

        if sampling == "random":
            ds = ds.shuffle(seed=seed)
        elif sampling != "first":
            raise ValueError("sampling must be 'first' or 'random'")

        ds = ds.select(range(num_samples))
    return ds

def prepare_inputs_with_images(dataset, batch_indices: List[int], language: str) -> List[List[dict]]:
    """Prepare batch of inputs with prompts and images using chat format"""
    inputs = []

    for idx in batch_indices:
        example = dataset[idx]

        if language == "en":
            prompt = PROMPT_EN 
        else:  # language == "eu"
            prompt = PROMPT_EU 
        
        img = example["image"]

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
    dataset = load_dataset_from_hf(args.dataset_name, subset="docci", num_samples=args.num_sample_dataset)

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
                    prediction = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
        
                    if args.language == "en":
                        question = PROMPT_EN 
                    else: 
                        question = PROMPT_EU
                    
                    print(question)
                    print("----------")
                    print(prediction)
                    print("----------")

    
                    # Store result
                    result = {
                        "idx": idx,
                        "question_id": example["example_id"],
                        "question": question,
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