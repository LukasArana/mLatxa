
import argparse
import requests

from datasets import load_dataset


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a model on the mmstar_eu dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate")
    parser.add_argument("--server_url", type=str, required=True, help="URL of the vllm server")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    parser.add_argument("--dataset_name", nargs="+", default=["\n"], help="Name of the dataset to evaluate on")
    return parser.parse_args()

def create_few_shot(dataset, n=5):
    few_shot_examples = []
    for i in range(n):
        example = dataset[i]
        question = example["question"]
        answer = example["answer"]
        few_shot_examples.append(f"Question: {question}\nAnswer: {answer}")
    return "\n\n".join(few_shot_examples)

if __name__ == "__main__":
  #Ask the user for model name, server adress and port using program arguments
  args = get_arguments()

  dataset_name = args.dataset_name[0]
  dataset = load_dataset(dataset_name, split="train")

  for i in dataset:
      print(i)
      img = i["image"]
      question = i["question"]
      ground_truth = i["answer"]

      model_name = args.model_name
      few_shot_examples = create_few_shot(dataset)
      prompt = f"{few_shot_examples}\nQuestion: {question}\nAnswer:"

      url = f"{args.server_url}/v1/completions"
      payload = {
          "model": model_name,
          "prompt": prompt,
          "max_tokens": args.max_new_tokens,
          "temperature": args.temperature,
          "top_p": args.top_p,
          "stop": ["\n"]
      }

      response = requests.post(url, json=payload)

      print("Model's answer:", response.json()["choices"][0]["text"])
      print("Ground truth:", ground_truth)
