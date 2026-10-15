"""Expand environment variables in a `--custom_dataset_info` JSON and print the rendered file's path.

The configs in configs/ and configs_mn5/ use ${WORK}, ${SCRATCH} and ${EUHPC_WORK} instead of
user-specific absolute paths. JSON can't expand them, so export scripts render the config first:

    CFG=$(python $HOME/mLatxa/datasets/export/render_config.py configs_mn5/v1.json)
    swift export --custom_dataset_info "$CFG" ...
"""
import json
import os
import sys
import tempfile

# Shared Leonardo project that holds the raw pretraining/instruct JSONL (not the user's $WORK).
os.environ.setdefault("EUHPC_WORK", "/leonardo_work/EUHPC_E04_042")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} CONFIG.json")
    src = sys.argv[1]
    with open(src) as f:
        entries = json.load(f)

    for entry in entries:
        path = os.path.expanduser(os.path.expandvars(entry["dataset_path"]))
        if "$" in path:
            raise SystemExit(f"{src}: unset variable in dataset_path of '{entry['dataset_name']}': {path}")
        entry["dataset_path"] = path

    fd, dst = tempfile.mkstemp(prefix=os.path.basename(src).removesuffix(".json") + ".", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(entries, f, indent=2)
    print(dst)


if __name__ == "__main__":
    main()
