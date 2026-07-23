"""
check_batch.py

Checks the current status of an OpenAI Batch API job.
Prints "Status: <status>" so the shell runner can parse it with grep/awk.
"""

import argparse
from pathlib import Path

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_id_file", required=True,
                        help="Path to the file containing the batch ID.")
    args = parser.parse_args()

    id_path = Path(args.batch_id_file)
    if not id_path.exists():
        raise FileNotFoundError(f"Batch ID file not found: {id_path}")

    batch_id = id_path.read_text().strip()
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    print(f"Status: {batch.status}")
    print(f"Batch ID: {batch.id}")
    if batch.request_counts:
        print(f"Total    : {batch.request_counts.total}")
        print(f"Completed: {batch.request_counts.completed}")
        print(f"Failed   : {batch.request_counts.failed}")


if __name__ == "__main__":
    main()