"""
prepare_batch.py

Reads a predictions JSONL and the allenai/pixmo-ask-model-anything HuggingFace
dataset, pairs each prediction with the matching ground-truth question, and
writes an OpenAI Batch API JSONL file ready for submission.
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

JUDGE_PROMPT_TEMPLATE = """\
You are a fair judge assistant tasked with evaluating model-generated answers in Basque, providing clear and objective scores based on specific criteria.

###Task Description:
A question, a model-generated answer in Basque to evaluate, a reference answer in English, and two score rubrics are given.
1. For each rubric, write a brief feedback assessing the answer strictly based on the criteria.
2. After writing the feedback for each rubric, write the corresponding score.
3. The output format should look as follows:
{{
  "correctness_feedback": (write a feedback for correctness),
  "correctness": (0 or 1),
  "basque_quality_feedback": (write a feedback for Basque quality),
  "basque_quality": (0, 1, or 2)
}}
4. Please do not generate any other opening, closing, or explanations. Respond only with a JSON object containing the feedback and scores as specified above.

###Question to evaluate:
{question}

###Answer to evaluate:
{prediction}

###Reference Answer:
{ground_truth}

###Score Rubric (Correctness):
[Does the prediction correctly answer the question, consistent with the reference answer?]
Score 0: The prediction does not answer the question, contradicts the reference answer, or contains major factual errors.
Score 1: The prediction correctly answers the question and is semantically consistent with the reference answer. Minor wording differences, paraphrases, or small grammatical errors are acceptable.

###Score Rubric (Basque Language Quality):
[Is the prediction written in fluent, natural, grammatically correct Basque?]
Score 0: Poor. Mostly ungrammatical, incoherent, unnatural, appears garbled or invented, or is not written in Basque.
Score 1: Understandable. Meaning is clear, but there are noticeable grammatical, lexical, spelling, or fluency issues.
Score 2: Good. Fluent, natural, grammatically correct Basque with only minor or no errors.

###Feedback:"""

 
JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "correctness_feedback": {"type": "string"},
                "correctness": {"type": "integer", "enum": [0, 1]},
                "basque_quality_feedback": {"type": "string"},
                "basque_quality": {"type": "integer", "enum": [0, 1, 2]},
            },
            "required": [
                "correctness_feedback",
                "correctness",
                "basque_quality_feedback",
                "basque_quality",
            ],
            "additionalProperties": False,
        },
    },
}

 


def load_hf_ground_truths(split: str, dataset_name: str) -> dict[int, str]:
    """
    Returns a dict mapping question_id -> ground_truth answer string
    from the specified HuggingFace dataset.
    The dataset field used as ground truth is 'question'.
    """
    print(f"Loading {dataset_name} (split={split}) from HuggingFace...")
    ds = load_dataset(dataset_name, split=split)

    # Build lookup: question_id (int) -> answer text
    # The dataset uses 'question_id' and 'question' fields.
    gt_map: dict[int, str] = {}
    for row in ds:
        image_path = row.get("image_url").split("/")[-1]
        answer = row.get("answer")  # field requested by user
        if image_path is not None and answer is not None:
            gt_map[image_path] = answer

    print(f"  Loaded {len(gt_map)} ground-truth entries.")
    return gt_map


def build_batch_request(custom_id: str, model: str, prompt: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "max_completion_tokens": 1500,
            "response_format": JUDGE_SCHEMA,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", required=True, help="Path to the predictions JSONL file.")
    parser.add_argument("--batch_file", required=True, help="Output path for the OpenAI batch JSONL.")
    parser.add_argument("--model_name", default="gpt-4o", help="OpenAI model to use for judging.")
    parser.add_argument("--hf_split", default="train", help="HuggingFace dataset split to load (default: train).")
    parser.add_argument("--hf_dataset", default=None, help="HuggingFace dataset name to load (default: allenai/pixmo-ask-model-anything).")
    parser.add_argument("--ground_in_eus", action="store_true", help="Whether to include the 'ground_truth' field from predictions in basque as-is in the prompt (instead of looking up from HuggingFace).")

    args = parser.parse_args()

    predictions_path = Path(args.predictions_file)
    if not predictions_path.exists():
        sys.exit(f"ERROR: predictions file not found: {predictions_path}")

    # Load ground truths from HuggingFace
    gt_map = load_hf_ground_truths(args.hf_split, args.hf_dataset)

    # Read predictions
    predictions = []
    with open(predictions_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping malformed line {lineno}: {e}")

    print(f"Read {len(predictions)} predictions from {predictions_path}")

    # Build batch requests
    batch_path = Path(args.batch_file)
    batch_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0
    with open(batch_path, "w") as out:
        for entry in predictions:
            idx = entry.get("idx")
            image_path = entry.get("image_path")
            question = entry.get("question", "").strip()
            prediction_eus = entry.get("ground_truth", "").strip()
            prediction = entry.get("prediction", "").strip()

            # Look up ground truth from HuggingFace dataset
            ground_truth = gt_map.get(image_path if image_path is not None else None)
            if ground_truth is None:
                print(f"  WARNING: no ground truth found for image_path={image_path} (idx={idx}), skipping.")
                skipped += 1
                continue

            prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=question,
                prediction=prediction_eus if args.ground_in_eus else prediction,
                ground_truth=ground_truth,
            )

            custom_id = f"judge-idx{idx}-forImage{image_path}"
            request = build_batch_request(custom_id, args.model_name, prompt)
            out.write(json.dumps(request, ensure_ascii=False) + "\n")
            written += 1

    print(f"Batch file written to {batch_path}")
    print(f"  Requests written : {written}")
    print(f"  Entries skipped  : {skipped}")


if __name__ == "__main__":
    main()