"""
submit_batch.py

Uploads the prepared batch JSONL to OpenAI and submits a Batch API job.
Saves the resulting batch ID to a .txt file for later polling.
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_file", required=True,
                        help="Path to the prepared batch JSONL file.")
    parser.add_argument("--batch_id_file", required=True,
                        help="Path where the batch ID will be saved as plain text.")
    args = parser.parse_args()

    client = OpenAI()  # reads OPENAI_API_KEY from environment

    batch_path = Path(args.batch_file)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    print(f"Uploading batch file: {batch_path} ...")
    with open(batch_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    print(f"File uploaded. File ID: {file_obj.id}")

    print("Submitting batch job...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"Batch submitted. Batch ID: {batch.id}")
    print(f"Status: {batch.status}")

    id_path = Path(args.batch_id_file)
    id_path.parent.mkdir(parents=True, exist_ok=True)
    id_path.write_text(batch.id)
    print(f"Batch ID saved to: {id_path}")


if __name__ == "__main__":
    main()