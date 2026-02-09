from argparse import ArgumentParser
import json
import os
from typing import List, Literal
# Remove vLLM imports
import openai
from vllm import LLM, SamplingParams
from pydantic import BaseModel, RootModel
import logging
from time import sleep
from datasets import load_dataset
from vllm.sampling_params import SamplingParams, GuidedDecodingParams
import jinja2 as j2
import re # Add re import
from examples import examples
import pandas as pd
import io
import base64

lang_dict = {
        "gal": "Galician",
        "eu": "Basque",
        "cat": "Catalan",
        "es": "Spanish"
        }

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class Conversation(RootModel):
    root: List[Message]

def check_output_format(output):
    # This is the simplest "skeleton" check
    pattern = r"Aukerak:.*?A:.*?B:.*?C:.*?D:.*"

    # re.DOTALL is crucial so that '.*' matches newlines
    # re.IGNORECASE makes it catch 'aukerak' or 'AUKERAK'
    match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)

    return match

def load_dataset_slice(dataset_path, slice=slice(0, None, 1)):
    try:
        dataset = load_dataset(dataset_path)
        dataset = dataset["val"]
    except:
        if "json" in dataset_path:
            #Load as json
            with open(dataset_path, "r") as f:
                dataset = json.load(f)
        elif "tsv" in dataset_path:
            dataset = pd.read_csv(dataset_path, sep="\t").to_dict("records")

    return [dataset[i] for i in range(*slice.indices(len(dataset)))]


def batch_generator(dataset, batch_size=1):
    #Return a generator that yields batches of the dataset
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def prepare_example(element, example, lang):
    prompt = [{"role": "user", "content": f"Translate the following example to {lang_dict[lang]} \n\n\n {element['question']}"}]
    return example + prompt


def main(args):
    # Set up OpenAI client with custom base URL
    llm = LLM(
        model=args.model_path,
        enable_prefix_caching=True,
        max_model_len = 8192,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype
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
    # Regex rule: Output MUST contain both words in order
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        stop="\n'''",
        frequency_penalty=args.frequency_penalty,
    )

    example = examples[args.prompt_lang]

    conversation_id = 0
    progress = 0

    if os.path.exists(output_file_path):
        with open(output_file_path, "rt") as f:
            progress = sum(1 for _ in f)
            conversation_id = progress

    print(f"Progress: {progress}")
    print(f"Conversation ID: {conversation_id}")

    dataset = load_dataset_slice(
        args.dataset_path, slice(args.dataset_start + progress, args.dataset_end, 1)
    )

    unable = []
    os.makedirs(args.output_path, exist_ok=True)
    with open(output_file_path, "a") as f:
        for i, batch in enumerate(batch_generator(dataset, batch_size=args.batch_size)):
            questions = [i["question"] for i in batch]

            prompts = [
                prepare_example(question, example, args.prompt_lang)
                for question in batch
            ]

            outputs = []
            try:
                outputs = [
                    output.outputs[0].text.strip()
                    for output in llm.chat(
                        prompts,
                        sampling_params=sampling_params,
                        use_tqdm=True,
                    )
                ]
            except Exception as e:
                logger.error(f"Error calling API: {e}")
                outputs.append("[]")  # Empty JSON array as fallback
            for idx, (original, output) in enumerate(zip(batch, outputs)):

                buffered = io.BytesIO()
                batch[idx]["image"].save(buffered, format="JPEG")

                # 3. Encode the binary data to a Base64 string
                img_byte = buffered.getvalue()
                batch[idx]["image"] = base64.b64encode(img_byte).decode('utf-8')

                if not check_output_format(output):
                    idx = i * args.batch_size + idx
                    print(f"Unable to translate example {idx}")
                    unable.append(idx)
                    continue
                batch[idx]["question"] = output
                conversation_id += 1
                breakpoint()
                try:
                    print(json.dumps(batch[idx], ensure_ascii=False), file=f)
                except UnicodeEncodeError: # Emojis raise this error
                    logger.error("Failed to encode output")
                    logger.error(output)
                    output_dict = {"conversation_id": original["conversation_id"]}
                    print(json.dumps(output_dict, ensure_ascii=False), file=f)
            logger.warning(f"Processed {progress + i * args.batch_size} examples")

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
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    # Remove vLLM specific arguments
    parser.add_argument("--dataset_start", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=124)  # Reduced batch size for API calls
    parser.add_argument("--multiple_choice", type=bool, default=False, help="Use multiple choice questions")
    args = parser.parse_args()
    main(args)