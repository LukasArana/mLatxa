"""
download_batch.py

Downloads the completed OpenAI Batch API results and parses the judge
JSON output into a clean output JSONL, one entry per prediction.

The first line of the output JSONL is a summary record:
  {"_type": "metrics", "correctness_accuracy": ..., "quality_mean": ...,
   "quality_distribution": {"0": ..., "1": ..., "2": ..., "3": ..., "4": ..., "5": ...},
   "quality_good_rate": ..., "n_scored": ..., "n_missing": ...}

Subsequent lines are per-prediction records:
  idx, question_id, prediction,
  linguistic_correctness_feedback, linguistic_correctness,
  style_naturalness_feedback, style_naturalness,
  raw_response
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI


def parse_judge_response(text: str) -> dict:
    """
    Parses the judge JSON response. With json_schema response_format enforced
    at request time, this should always succeed cleanly.
    """
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        print(f"  WARNING: failed to parse judge response as JSON: {e}")
        return {}


def compute_metrics(rows: list[dict]) -> dict:
    correctness_scores = [r["linguistic_correctness"] for r in rows if r.get("linguistic_correctness") is not None]
    quality_scores     = [r["style_naturalness"] for r in rows if r.get("style_naturalness") is not None]

    n_correctness = len(correctness_scores)
    n_quality     = len(quality_scores)

    if n_correctness:
        correctness_mean      = sum(correctness_scores) / n_correctness
        correctness_good_rate = sum(1 for s in correctness_scores if s >= 4) / n_correctness
        correctness_dist      = {
            str(k): correctness_scores.count(k) / n_correctness
            for k in range(6)
        }
    else:
        correctness_mean = correctness_good_rate = correctness_dist = None

    if n_quality:
        quality_mean      = sum(quality_scores) / n_quality
        quality_good_rate = sum(1 for s in quality_scores if s >= 4) / n_quality
        quality_dist      = {
            str(k): quality_scores.count(k) / n_quality
            for k in range(6)
        }
    else:
        quality_mean = quality_good_rate = quality_dist = None

    return {
        "_type": "metrics",
        "linguistic_correctness_mean": correctness_mean,
        "linguistic_correctness_distribution": correctness_dist,
        "linguistic_correctness_good_rate": correctness_good_rate,
        "naturalness_quality_mean": quality_mean,
        "naturalness_quality_distribution": quality_dist,
        "naturalness_quality_good_rate": quality_good_rate,
        "n_scored": n_correctness,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_id_file", required=True,
                        help="Path to the file containing the batch ID.")
    parser.add_argument("--predictions_file", required=True,
                        help="Original predictions JSONL (to restore metadata).")
    parser.add_argument("--out_file", required=True,
                        help="Output JSONL path for judged results.")
    args = parser.parse_args()

    client = OpenAI()

    batch_id = Path(args.batch_id_file).read_text().strip()
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        raise RuntimeError(f"Batch is not completed yet. Current status: {batch.status}")

    if not batch.output_file_id:
        raise RuntimeError("Batch has no output_file_id. It may have failed entirely.")

    print(f"Downloading results for batch {batch_id} ...")
    content = client.files.content(batch.output_file_id).text

    # Build lookup: custom_id -> parsed judge output
    results: dict[str, dict] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = record["custom_id"]

        if record.get("error"):
            print(f"  WARNING: request-level error for {custom_id}: {record['error']}")
            results[custom_id] = {"raw_response": "", "error": record["error"]}
            continue

        try:
            choice = record["response"]["body"]["choices"][0]
            raw_text = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason", "")
        except (KeyError, IndexError, TypeError) as e:
            print(f"  WARNING: unexpected response structure for {custom_id}: {e}")
            results[custom_id] = {"raw_response": ""}
            continue

        if finish_reason != "stop":
            print(f"  WARNING: unexpected finish_reason='{finish_reason}' for {custom_id}")

        parsed = parse_judge_response(raw_text)
        results[custom_id] = {"raw_response": raw_text, **parsed}

    print(f"Downloaded {len(results)} result(s).")

    # Read original predictions; key by custom_id (must match prepare_batch.py)
    predictions: dict[str, dict] = {}
    with open(args.predictions_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            idx = entry.get("idx")
            question_id = entry.get("question_id", "")
            custom_id = f"judge-idx{idx}-{question_id}"
            predictions[custom_id] = entry

    # Build merged rows
    rows = []
    missing = 0
    for custom_id, pred in predictions.items():
        judge = results.get(custom_id)
        if judge is None:
            print(f"  WARNING: no result for {custom_id}")
            missing += 1
            judge = {}

        rows.append({
            "idx": pred.get("idx"),
            "question_id": pred.get("question_id"),
            "prediction": pred.get("prediction"),
            "linguistic_correctness_feedback": judge.get("linguistic_correctness_feedback", ""),
            "linguistic_correctness": judge.get("linguistic_correctness", None),
            "style_naturalness_feedback": judge.get("style_naturalness_feedback", ""),
            "style_naturalness": judge.get("style_naturalness", None),
            "raw_response": judge.get("raw_response", ""),
        })

    # Compute metrics across all rows
    metrics = compute_metrics(rows)
    metrics["n_missing"] = missing

    # Write output: metrics first, then per-prediction rows
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as out:
        out.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Results written to: {out_path}")
    print(f"  Written : {len(rows)}")
    print(f"  Missing : {missing}")


if __name__ == "__main__":
    main()