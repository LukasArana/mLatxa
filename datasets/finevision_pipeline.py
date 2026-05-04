#!/usr/bin/env python
"""
FineVision pipeline: load from HuggingFace, quality-filter, optionally sample to a
target token count, and write MS-Swift JSONL format.

Output format per line:
  {"messages": [{"role": "user", "content": "<image>\n..."}, {"role": "assistant", "content": "..."}],
   "images": ["/abs/path/to/image.png"]}

Token-based sampling (optional):
  --tokenizer_path enables token counting per sample.
  --target_tokens  stops after accumulating N tokens (text + image).
  --instruct_path  counts tokens from a reference JSONL to set target_tokens automatically.
"""

import json
import os
import argparse
import glob as glob_module
import hashlib
import random
import multiprocessing as mp
from collections import deque
from io import BytesIO
from typing import Optional, List, Dict, Tuple

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Worker-process globals
# ---------------------------------------------------------------------------

_tokenizer = None
_image_processor = None
_images_dir = None


def _init_worker(tokenizer_path: str):
    """Initializer for instruct token-counting workers (tokenizer only)."""
    global _tokenizer
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _init_sample_worker(tokenizer_path: Optional[str], images_dir: str):
    """Initializer for sample-processing workers (tokenizer + image processor)."""
    global _tokenizer, _image_processor, _images_dir
    _images_dir = images_dir
    if tokenizer_path:
        from transformers import AutoTokenizer, Qwen2VLImageProcessorFast
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        _image_processor = Qwen2VLImageProcessorFast.from_pretrained(tokenizer_path, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------

def _count_text_tokens(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return 0


def _count_image_tokens(image_processor, img) -> int:
    """Compute actual visual token count using Qwen2VLImageProcessorFast."""
    try:
        processed = image_processor(images=img, return_tensors="pt")
        t, h, w = processed["image_grid_thw"][0].tolist()
        merge_size = image_processor.merge_size  # typically 2
        return t * h * w // (merge_size * merge_size)
    except Exception:
        # Fallback: patch-based estimate
        merge = 2
        return max(1, -(-img.height // (16 * merge))) * max(1, -(-img.width // (16 * merge))) + 2


# ---------------------------------------------------------------------------
# Instruct dataset token counting (parallel, text-only)
# ---------------------------------------------------------------------------

def _count_instruct_line(line: str) -> int:
    global _tokenizer
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return 0
    total = 0
    if "conversations" in data:
        for turn in data["conversations"]:
            if isinstance(turn, dict):
                total += _count_text_tokens(_tokenizer, turn.get("content", ""))
                total += _count_text_tokens(_tokenizer, turn.get("user", ""))
                total += _count_text_tokens(_tokenizer, turn.get("assistant", ""))
    elif "text" in data:
        total += _count_text_tokens(_tokenizer, data["text"])
    elif "prompt" in data and "response" in data:
        total += _count_text_tokens(_tokenizer, data["prompt"])
        total += _count_text_tokens(_tokenizer, data["response"])
    return total


def count_instruct_tokens(tokenizer_path: str, instruct_path: str, num_workers: int) -> int:
    print(f"\nCounting tokens in Instruct dataset: {instruct_path}")
    with open(instruct_path, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  {len(lines):,} lines, using {num_workers} workers...")
    with mp.Pool(num_workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
        counts = list(tqdm(
            pool.imap(_count_instruct_line, lines, chunksize=500),
            total=len(lines), desc="Tokenizing Instruct dataset",
        ))
    total = sum(counts)
    print(f"  {len(lines):,} examples, {total:,} tokens")
    return total


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _open_image(img_data) -> Optional[Image.Image]:
    """Accept PIL Image, bytes, or HF-style dict {bytes, path}. Returns PIL Image or None."""
    if img_data is None:
        return None
    if isinstance(img_data, Image.Image):
        return img_data
    if isinstance(img_data, dict):
        img_bytes = img_data.get("bytes")
        if img_bytes is None:
            return None
    elif isinstance(img_data, (bytes, bytearray)):
        img_bytes = img_data
    else:
        return None
    try:
        return Image.open(BytesIO(img_bytes))
    except Exception:
        return None


def _save_image(img: Image.Image, img_data, images_dir: str, fallback_name: str) -> str:
    """Save image to disk. Uses md5 name for byte data, fallback_name for PIL-only."""
    if isinstance(img_data, dict):
        img_bytes = img_data.get("bytes")
    elif isinstance(img_data, (bytes, bytearray)):
        img_bytes = img_data
    else:
        img_bytes = None

    if img_bytes is not None:
        digest = hashlib.md5(img_bytes).hexdigest()
        fmt = (img.format or "JPEG").lower().replace("jpeg", "jpg")
        filepath = os.path.join(images_dir, f"{digest}.{fmt}")
        if not os.path.exists(filepath):
            with open(filepath, "wb") as f:
                f.write(img_bytes)
    else:
        filepath = os.path.join(images_dir, fallback_name)
        if not os.path.exists(filepath):
            img.save(filepath)
    return os.path.abspath(filepath)


# ---------------------------------------------------------------------------
# Sample processing worker
# ---------------------------------------------------------------------------

def _process_sample(args: Tuple) -> Tuple[List[Dict], int, int]:
    """
    Worker: convert one raw sample to output records.
    Returns (list_of_jsonl_dicts, text_tokens, image_tokens).
    """
    sample, global_idx, min_image_corr, min_visual_dep = args

    images_raw = sample.get("images") or []
    texts = sample.get("texts") or []
    image_corr_ratings = sample.get("image_correspondence_ratings") or []
    visual_dep_ratings = sample.get("visual_dependency_ratings") or []

    if not images_raw or not texts:
        return [], 0, 0

    image_paths = []
    sample_image_tokens = 0
    for img_idx, img_data in enumerate(images_raw):
        img = _open_image(img_data)
        if img is None:
            continue
        fallback_name = f"sample_{global_idx:06d}_img_{img_idx}.png"
        path = _save_image(img, img_data, _images_dir, fallback_name)
        image_paths.append(path)
        if _image_processor is not None:
            sample_image_tokens += _count_image_tokens(_image_processor, img)

    if not image_paths:
        return [], 0, 0

    image_prefix = "<image>" * len(image_paths)
    records = []
    text_tokens = 0

    for text_idx, text_pair in enumerate(texts):
        if not isinstance(text_pair, dict):
            continue
        user_text = str(text_pair.get("user", "")).strip()
        assistant_text = str(text_pair.get("assistant", "")).strip()
        if not user_text or not assistant_text:
            continue
        if image_corr_ratings and text_idx < len(image_corr_ratings):
            if image_corr_ratings[text_idx] < min_image_corr:
                continue
        if visual_dep_ratings and text_idx < len(visual_dep_ratings):
            if visual_dep_ratings[text_idx] < min_visual_dep:
                continue
        if _tokenizer is not None:
            text_tokens += _count_text_tokens(_tokenizer, user_text)
            text_tokens += _count_text_tokens(_tokenizer, assistant_text)
        records.append({
            "messages": [
                {"role": "user", "content": f"{image_prefix}\n{user_text}"},
                {"role": "assistant", "content": assistant_text},
            ],
            "images": image_paths,
        })

    return records, text_tokens, sample_image_tokens


# ---------------------------------------------------------------------------
# Subset discovery and dataset loading
# ---------------------------------------------------------------------------

def _get_local_subsets(path: str) -> List[str]:
    """Return sorted list of subdirectory names (excluding text_*) that contain parquet files."""
    subsets = []
    for name in sorted(os.listdir(path)):
        subdir = os.path.join(path, name)
        if os.path.isdir(subdir) and not name.startswith("text_"):
            if glob_module.glob(os.path.join(subdir, "*.parquet")):
                subsets.append(name)
    return subsets


def _parquet_iter(parquet_files: List[str]):
    """Yield rows from local parquet files as plain dicts, bypassing datasets schema parsing."""
    import pyarrow.parquet as pq
    for pf in parquet_files:
        try:
            table = pq.read_table(pf)
            d = table.to_pydict()
            keys = list(d.keys())
            for i in range(len(table)):
                yield {k: d[k][i] for k in keys}
        except Exception as e:
            tqdm.write(f"Warning: skipping {pf}: {e}")


def _load_subset_datasets(dataset_name: str, split: str, subset: Optional[str]) -> Dict:
    """
    Returns {subset_name: iterable} for all discovered subsets.
    If subset is given, loads only that one.
    Local paths use pyarrow directly (avoids datasets schema compatibility issues).
    HF repo names use load_dataset with streaming=True.
    """
    is_local = os.path.isdir(dataset_name)

    if subset:
        names = [subset]
    elif is_local:
        names = _get_local_subsets(dataset_name)
        if not names:
            raise ValueError(f"No parquet subdirectories found in {dataset_name!r}")
    else:
        from datasets import get_dataset_config_names
        all_configs = get_dataset_config_names(dataset_name)
        names = [c for c in all_configs if not c.startswith("text_")]
        if not names:
            raise ValueError(f"No non-text configs found for {dataset_name!r}")

    print(f"\nFound {len(names)} subset(s): {names}")

    result = {}
    for name in names:
        if is_local:
            parquet_files = sorted(glob_module.glob(os.path.join(dataset_name, name, "*.parquet")))
            result[name] = _parquet_iter(parquet_files)
        else:
            result[name] = load_dataset(dataset_name, name, split=split, streaming=True)
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset_name: str,
    split: str,
    subset: Optional[str],
    output_path: str,
    images_dir: str,
    tokenizer_path: Optional[str],
    target_tokens: Optional[int],
    instruct_path: Optional[str],
    min_image_correspondence: int,
    min_visual_dependency: int,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    os.makedirs(images_dir, exist_ok=True)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Step 1: resolve target token count
    if tokenizer_path and instruct_path and target_tokens is None:
        target_tokens = count_instruct_tokens(tokenizer_path, instruct_path, num_workers)
        print(f"Target token count from Instruct dataset: {target_tokens:,}")
    elif target_tokens is not None:
        print(f"Using provided target token count: {target_tokens:,}")

    # Step 2: discover subsets and build round-robin iterators
    print(f"\nLoading dataset {dataset_name!r} (split={split})")
    subset_datasets = _load_subset_datasets(dataset_name, split, subset)
    random.seed(seed)

    iterators: deque = deque((name, iter(ds)) for name, ds in subset_datasets.items())

    # Step 3: process samples in parallel batches
    total_written = 0
    total_skipped = 0
    total_text_tokens = 0
    total_image_tokens = 0
    global_idx = 0

    pbar_kwargs = {}
    if target_tokens:
        pbar_kwargs = {"total": target_tokens, "desc": "Tokens processed", "unit": "tok"}
    else:
        pbar_kwargs = {"desc": "Samples processed", "unit": "sample"}
    pbar = tqdm(**pbar_kwargs)

    pool = mp.Pool(
        num_workers,
        initializer=_init_sample_worker,
        initargs=(tokenizer_path, images_dir),
    )

    try:
        with open(output_path, "w", encoding="utf-8") as f_out:
            while iterators:
                if target_tokens is not None and (total_text_tokens + total_image_tokens) >= target_tokens:
                    break

                # Collect one batch round-robin
                batch: List[Tuple] = []
                exhausted = []
                while len(batch) < batch_size and iterators:
                    subset_name, it = iterators.popleft()
                    try:
                        sample = next(it)
                        iterators.append((subset_name, it))
                        batch.append((sample, global_idx, min_image_correspondence, min_visual_dependency))
                        global_idx += 1
                    except StopIteration:
                        tqdm.write(f"Subset '{subset_name}' exhausted")

                if not batch:
                    break

                results = pool.map(_process_sample, batch)

                for records, text_tok, img_tok in results:
                    if target_tokens is not None and (total_text_tokens + total_image_tokens) >= target_tokens:
                        break
                    if not records:
                        total_skipped += 1
                        continue
                    for rec in records:
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_written += len(records)
                    prev_tokens = total_text_tokens + total_image_tokens
                    total_text_tokens += text_tok
                    total_image_tokens += img_tok
                    if target_tokens is not None:
                        pbar.update((total_text_tokens + total_image_tokens) - prev_tokens)
                    else:
                        pbar.update(len(records))
    finally:
        pool.close()
        pool.join()

    pbar.close()

    # Summary
    total_tokens = total_text_tokens + total_image_tokens
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    if target_tokens is not None:
        print(f"Target tokens:      {target_tokens:>15,}")
        print(f"Total tokens:       {total_tokens:>15,}")
        print(f"  - Text tokens:    {total_text_tokens:>15,}")
        print(f"  - Image tokens:   {total_image_tokens:>15,}")
    print(f"Written samples:    {total_written:>15,}")
    print(f"Skipped samples:    {total_skipped:>15,}")
    print(f"Output file:        {output_path}")
    print(f"Images dir:         {images_dir}")

    # Metadata
    stats_path = output_path.replace(".jsonl", "_metadata.json")
    meta = {
        "dataset_name": dataset_name,
        "split": split,
        "subsets": list(subset_datasets.keys()),
        "target_tokens": target_tokens,
        "total_tokens": total_tokens,
        "text_tokens": total_text_tokens,
        "image_tokens": total_image_tokens,
        "written_samples": total_written,
        "skipped_samples": total_skipped,
        "min_image_correspondence": min_image_correspondence,
        "min_visual_dependency": min_visual_dependency,
        "seed": seed,
    }
    with open(stats_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved:     {stats_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="FineVision pipeline: HF streaming → quality filter → MS-Swift JSONL"
    )
    parser.add_argument("--dataset_name", type=str, default="/leonardo_work/AIFAC_5C0_261/datasets/train/finevision",
        help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset", type=str, default=None,
        help="Load only this subset/config; by default all non-text subsets are loaded round-robin")
    parser.add_argument("--output_path", type=str,
        default="/leonardo_work/AIFAC_5C0_261/datasets/train/finevisionProcessed/train.jsonl",
        help="Output JSONL path")
    parser.add_argument("--images_dir", type=str,
        default="/leonardo_work/AIFAC_5C0_261/datasets/train/finevisionProcessed/images",
        help="Directory to save extracted images")
    parser.add_argument("--tokenizer_path", type=str, default="/leonardo_work/AIFAC_5C0_261/baseModels/Qwen3.5-9B",
        help="Path to tokenizer; enables token counting (required for --target_tokens / --instruct_path)")
    parser.add_argument("--target_tokens", type=int, default=None,
        help="Stop after accumulating N tokens (requires --tokenizer_path)")
    parser.add_argument("--instruct_path", type=str, default="/leonardo_work/AIFAC_5C0_261/datasets/datasets/InstructDatasets/magpie.qwen3.32b.en.noreasoning.onlyconver.jsonl",
        help="Count tokens from this JSONL to set target_tokens (requires --tokenizer_path)")
    parser.add_argument("--min_image_correspondence", type=int, default=0,
        help="Minimum image_correspondence_rating to include a text pair")
    parser.add_argument("--min_visual_dependency", type=int, default=0,
        help="Minimum visual_dependency_rating to include a text pair")
    parser.add_argument("--batch_size", type=int, default=64,
        help="Batch size for tqdm progress reporting (unused in processing)")
    parser.add_argument("--num_workers", type=int, default=max(1, mp.cpu_count() - 1),
        help="Number of workers for instruct dataset token counting")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.target_tokens is not None and args.tokenizer_path is None:
        raise ValueError("--target_tokens requires --tokenizer_path")
    if args.instruct_path is not None and args.tokenizer_path is None:
        raise ValueError("--instruct_path requires --tokenizer_path")

    run_pipeline(
        dataset_name=args.dataset_name,
        split=args.split,
        subset=args.subset,
        output_path=args.output_path,
        images_dir=args.images_dir,
        tokenizer_path=args.tokenizer_path,
        target_tokens=args.target_tokens,
        instruct_path=args.instruct_path,
        min_image_correspondence=args.min_image_correspondence,
        min_visual_dependency=args.min_visual_dependency,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # required for transformers/CUDA
    main()
