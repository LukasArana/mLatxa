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
def load_dataset_slice(dataset_path, slice=slice(0, None, 1)):
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

    return dataset[slice]
def batch_generator(dataset, batch_size=1):
    #Return a generator that yields batches of the dataset
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def prepare_example(element, example, lang):
    answer = f'1.{element["A"]}, 2.{element["B"]}, 3.{element["C"]}, 4.{element["D"]}'
    element["conversations"] = [{"role": "user", "content": f"Translate the following question and answers to {lang_dict[lang]} \n\n\n Question: {element['question']} \n Answer: {answer}"}]
    return example + element["conversations"]

def split_response(response, option_name):

    # Build a regex that captures each numbered option and the text that follows it
    # It will capture text for each key up until the next key or end of string.
    keys = sorted(option_name.values())  # sort for consistent order (e.g. ['1','2','3','4'])
    # build tokens like '1\.' '2\.' ... for use in the regex
    key_tokens = [re.escape(k) + r"\." for k in keys]
    if not key_tokens:
        return []
    # pattern captures the key (with dot) and the following text lazily until the next key or end
    pattern = r"(?s)(" + "|".join(key_tokens) + r")\s*(.*?)(?=(?:" + "|".join(key_tokens) + r")|$)"
    matches = re.findall(pattern, response)
    if not matches:
        print(f"Emaitzak ez dira aurkitu: {response}")
        return []

    # matches is a list of tuples: (key_with_dot, text)
    # Convert to a mapping key -> text (strip trailing dot and whitespace)
    found = {}
    for key_with_dot, text in matches:
        key_plain = key_with_dot.strip().rstrip('.')
        found[key_plain] = text.strip()

    # Return values in the expected order (matching keys list). If a key is missing, return empty string
    ordered_texts = []
    for k in keys:
        ordered_texts.append(found.get(k, ""))
    return ordered_texts

def postprocess_output(output, option_name):
    try:
        instruction = output[(output.index("Question:") + len("Question:")):(output.index("Answer:") -3)]
        response = output[output.index("Answer:") + len("Answer:"):]
    except:
        return None
    #Remove leading spaces
    instruction = instruction.strip()
    response = response.strip()
    # Pass only the answer text to the splitter (not the whole output)
    choices = split_response(response, option_name)
    return instruction, choices

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
    example = examples["mm_bench"][args.prompt_lang]
    option_name = {"A": "1", "B": "2", "C": "3", "D": "4"}
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
            prompts = [
                prepare_example(element, example, args.prompt_lang)
                for element in batch
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
                results = postprocess_output(output, option_name)
                if not results:
                    idx = i * args.batch_size + idx
                    print(f"Unable to translate example {idx}")
                    unable.append(idx)
                    continue
                else:
                    question, choices = results
                output_dict = original
                output_dict["question"] = question
                for idx in range(len(choices)):
                    output_dict[list(option_name.values())[idx]] = choices[idx]
                conversation_id += 1
                try:
                    print(json.dumps(output_dict, ensure_ascii=False), file=f)
                except UnicodeEncodeError: # Emojis raise this error
                    logger.error("Failed to encode output")
                    logger.error(output)
                    output_dict = {"conversation_id": original["conversation_id"]}
                    print(json.dumps(output_dict, ensure_ascii=False), file=f)
            logger.warning(f"Processed {progress + i * args.batch_size} examples")

            print(output_dict)
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
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    # Remove vLLM specific arguments
    parser.add_argument("--dataset_start", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=124)  # Reduced batch size for API calls
    parser.add_argument("--multiple_choice", type=bool, default=False, help="Use multiple choice questions")
    args = parser.parse_args()
    main(args)