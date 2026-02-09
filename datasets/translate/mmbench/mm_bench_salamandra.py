from argparse import ArgumentParser
import json
import os
import re
from typing import Dict, List, Literal, Optional, Tuple

from vllm import LLM, SamplingParams
from pydantic import BaseModel, RootModel
import logging
from datasets import load_dataset
from examples import examples
import pandas as pd

lang_dict = {
        "gal": "Galician",
        "eu": "Basque",
        "cat": "Catalan",
        "es": "Spanish"
        }
OPTION_PREFIX_PATTERN = re.compile(
    r"^\s*\d+\s*(?:[\.\)]|º)?\s*(?:aukera|opcion|opción|opcio|opció|option|answer)?\s*[:.\-–—]?\s*",
    re.IGNORECASE,
)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
class Conversation(RootModel):
    root: List[Message]
def load_dataset_slice(dataset_path, dataset_slice=slice(0, None, 1)):
    try:
        dataset = load_dataset(dataset_path)
        dataset = dataset["train"]
    except:
        if "json" in dataset_path:
            #Load as json
            with open(dataset_path, "r") as f:
                dataset = json.load(f)
        elif "tsv" in dataset_path:
            dataset = pd.read_csv(dataset_path, sep="\t").to_dict("records")
    return dataset[dataset_slice]
def batch_generator(dataset, batch_size=1):
    #Return a generator that yields batches of the dataset
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def build_translation_prompt(source_text: str, target_lang: str, source_lang: str = "English") -> str:
    return (
        f"Translate the following text from {source_lang} into {target_lang}.\n\n"
        f"{source_lang}: {source_text}\n\n"
        f"{target_lang}:"
    )

def strip_option_prefix(text: str) -> str:
    """Remove numbering like `2. aukera:` from translated options."""
    return OPTION_PREFIX_PATTERN.sub("", text, count=1).strip()

def prepare_translation_tasks(
    batch: List[dict],
    example: List[dict],
    lang: str,
) -> Tuple[List[List[dict]], List[Dict[str, str]]]:
    target_lang = lang_dict[lang]
    option_keys = ["A", "B", "C", "D"]
    prompts: List[List[dict]] = []
    metadata: List[Dict[str, str]] = []

    for element_index, element in enumerate(batch):
        question_prompt = build_translation_prompt(
            source_text=f"Question: {element['question']}",
            target_lang=target_lang,
        )
        prompts.append(example + [{"role": "user", "content": question_prompt}])
        metadata.append({"element_index": element_index, "field": "question"})

        for idx, option_key in enumerate(option_keys, start=1):
            option_prompt = build_translation_prompt(
                source_text=f"Option {idx}: {element[option_key]}",
                target_lang=target_lang,
            )
            prompts.append(example + [{"role": "user", "content": option_prompt}])
            metadata.append(
                {
                    "element_index": element_index,
                    "field": "option",
                    "option_position": str(idx),
                }
            )

    return prompts, metadata

def run_llm_chat(
    llm: LLM,
    prompts: List[List[dict]],
    sampling_params: SamplingParams,
) -> List[str]:
    outputs = llm.chat(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    return [output.outputs[0].text.strip() for output in outputs]

def assemble_translations(
    batch: List[dict],
    responses: List[str],
    metadata: List[Dict[str, str]],
) -> Tuple[List[Optional[dict]], List[int]]:
    option_name = {"1": "1", "2": "2", "3": "3", "4": "4"}
    translated_batch = [dict(item) for item in batch]
    failures: List[int] = []

    # Initialize containers for each sample
    staged = [
        {"question": None, "options": {key: None for key in option_name}}
        for _ in batch
    ]

    for text, meta in zip(responses, metadata):
        element_idx = meta["element_index"]
        if meta["field"] == "question":
            staged[element_idx]["question"] = text
        else:
            option_position = meta["option_position"]
            staged[element_idx]["options"][option_position] = text

    for idx, (element, slots) in enumerate(zip(translated_batch, staged)):
        if not slots["question"] or None in slots["options"].values():
            failures.append(idx)
            translated_batch[idx] = None
            continue
        element["question"] = slots["question"]
        for option_key, option_value in slots["options"].items():
            element[option_name[option_key]] = strip_option_prefix(option_value)
        translated_batch[idx] = element

    return translated_batch, failures

def compute_dataset_slice(start: int, end: int, progress: int) -> slice:
    slice_start = start + progress
    slice_end = None if end == -1 else end
    return slice(slice_start, slice_end, 1)

def main(args):
    # Set up OpenAI client with custom base URL
    llm = LLM(
        model=args.model_path,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
        guided_decoding_backend="outlines"
    )
    output_dir = os.path.join(
        args.output_path,
        f"{os.path.basename(args.dataset_path).replace(".tsv", "")}_{args.prompt_lang}",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(
        output_dir,
        f"translated_{args.dataset_start}_{args.dataset_end}.jsonl"
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        stop="\n'''",
        frequency_penalty=args.frequency_penalty,
    )
    example = examples["mm_bench"][args.prompt_lang]
    conversation_id = 0
    progress = 0
    if os.path.exists(output_file_path):
        with open(output_file_path, "rt") as f:
            progress = sum(1 for _ in f)
            conversation_id = progress
    print(f"Progress: {progress}")
    print(f"Conversation ID: {conversation_id}")
    dataset_slice = compute_dataset_slice(
        args.dataset_start, args.dataset_end, progress
    )
    dataset = load_dataset_slice(args.dataset_path, dataset_slice)
    unable = []
    os.makedirs(args.output_path, exist_ok=True)
    with open(output_file_path, "a") as f:
        for i, batch in enumerate(batch_generator(dataset, batch_size=args.batch_size)):
            outputs = []
            metadata: List[Dict[str, str]] = []
            try:
                prompts, metadata = prepare_translation_tasks(
                    batch, example, args.prompt_lang
                )
                outputs = run_llm_chat(llm, prompts, sampling_params)
            except Exception as e:
                logger.error(f"Error calling API: {e}")
                outputs = []
            if not outputs:
                continue
            translated_batch, failed_indices = assemble_translations(
                batch, outputs, metadata
            )
            unable.extend([i * args.batch_size + idx for idx in failed_indices])

            for output_dict in translated_batch:
                if output_dict is None:
                    continue
                conversation_id += 1
                try:
                    print(json.dumps(output_dict, ensure_ascii=False), file=f)
                except UnicodeEncodeError: # Emojis raise this error
                    logger.error("Failed to encode output")
                    logger.error(output_dict)
                    fallback = {"conversation_id": output_dict["conversation_id"]}
                    print(json.dumps(fallback, ensure_ascii=False), file=f)
            logger.warning(f"Processed {progress + i * args.batch_size} examples")
            last_printed = translated_batch[-1]
            if last_printed:
                print(last_printed)
    print(unable)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="The model name to use with the OpenAI API.",
    )
    parser.add_argument(
        "--prompt_lang",
        type=str,
        default="eu",
        help="The prompt.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="HiTZ/Magpie-Llama-3.70B-Instruct-Filtered",
        help="The dataset path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/",
        help="The output path.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="The tensor parallel size.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.15)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    # Remove vLLM specific arguments
    parser.add_argument("--dataset_start", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=124)  # Reduced batch size for API calls
    parser.add_argument("--multiple_choice", type=bool, default=False, help="Use multiple choice questions")
    args = parser.parse_args()
    main(args)