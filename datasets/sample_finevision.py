#!/usr/bin/env python
"""
Sample examples from FineVision dataset to match the token count of a text-only Instruct dataset.

Steps:
1. Count tokens in the Instruct dataset (text-only, target)
2. Sample equally from all FineVision subsets (excluding text_* subsets)
3. Count text tokens + actual image tokens via Qwen2VLImageProcessorFast per sample
4. Save images to disk, write output JSONL with <image> tags and image paths
5. Stop when total tokens reach or exceed the target

Output format per line:
  {"messages": [{"role": "user", "content": "<image>..."}, {"role": "assistant", "content": "..."}],
   "images": ["/path/to/image.jpg"]}
"""

import os
import json
import argparse
import glob
import random
import hashlib
import multiprocessing as mp
from io import BytesIO
from typing import Optional, List, Dict, Tuple

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Worker-process globals (initialized once per worker)
# ---------------------------------------------------------------------------

_tokenizer = None
_image_processor = None
_images_dir = None


def _init_worker(tokenizer_path: str, images_dir: str):
    global _tokenizer, _image_processor, _images_dir
    from transformers import AutoTokenizer, Qwen2VLImageProcessorFast
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _image_processor = Qwen2VLImageProcessorFast.from_pretrained(tokenizer_path, trust_remote_code=True)
    _images_dir = images_dir


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------

def _count_text_tokens(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return 0


def _count_image_tokens(image_processor, img: Image.Image) -> int:
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


def _save_image(img_bytes: bytes, images_dir: str) -> Tuple[str, Image.Image]:
    """Save image bytes to disk (skip if exists). Returns (path, PIL image)."""
    digest = hashlib.md5(img_bytes).hexdigest()
    img = Image.open(BytesIO(img_bytes))
    fmt = (img.format or "JPEG").lower()
    if fmt == "jpeg":
        fmt = "jpg"
    filepath = os.path.join(images_dir, f"{digest}.{fmt}")
    if not os.path.exists(filepath):
        with open(filepath, "wb") as f:
            f.write(img_bytes)
    return filepath, img


# ---------------------------------------------------------------------------
# Worker function: process one sample
# ---------------------------------------------------------------------------

def _process_sample(args: Tuple[Dict, str]) -> Tuple[Dict, int, int]:
    """
    Worker: convert a raw parquet sample to output format.
    Returns (output_dict, text_tokens, image_tokens).
    """
    sample, subset_name = args
    breakpoint()
    texts = sample.get("texts") or []
    raw_images = sample.get("images") or []

    image_paths = []
    image_tokens = 0
    for img_data in raw_images:
        if img_data is None:
            continue
        try:
            img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
            if img_bytes is None:
                continue
            path, img = _save_image(img_bytes, _images_dir)
            image_paths.append(path)
            image_tokens += _count_image_tokens(_image_processor, img)
        except Exception:
            image_tokens += 256  # fallback

    image_prefix = "<image>" * len(image_paths)
    messages = []
    text_tokens = 0
    first_user = True
    for turn in texts:
        if not isinstance(turn, dict):
            continue
        user_text = str(turn.get("user", "")).strip()
        assistant_text = str(turn.get("assistant", "")).strip()
        if user_text:
            content = (image_prefix + user_text) if first_user else user_text
            messages.append({"role": "user", "content": content})
            text_tokens += _count_text_tokens(_tokenizer, user_text)
            first_user = False
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
            text_tokens += _count_text_tokens(_tokenizer, assistant_text)

    output = {"messages": messages, "images": image_paths}
    return output, text_tokens, image_tokens


# ---------------------------------------------------------------------------
# Instruct dataset token counting (parallel)
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
    with mp.Pool(num_workers, initializer=_init_worker, initargs=(tokenizer_path, "")) as pool:
        counts = list(tqdm(
            pool.imap(_count_instruct_line, lines, chunksize=500),
            total=len(lines), desc="Tokenizing Instruct dataset",
        ))
    total = sum(counts)
    print(f"  {len(lines):,} examples, {total:,} tokens")
    return total


# ---------------------------------------------------------------------------
# FineVision subset discovery
# ---------------------------------------------------------------------------

def get_finevision_subsets(finevision_path: str) -> List[str]:
    subsets = []
    for name in sorted(os.listdir(finevision_path)):
        path = os.path.join(finevision_path, name)
        if os.path.isdir(path) and not name.startswith("text_"):
            if glob.glob(os.path.join(path, "*.parquet")):
                subsets.append(name)
    return subsets


# ---------------------------------------------------------------------------
# Lazy parquet iterator (main process only)
# ---------------------------------------------------------------------------

class SubsetIterator:
    def __init__(self, subset_path: str, seed: int = 42):
        self.parquet_files = sorted(glob.glob(os.path.join(subset_path, "*.parquet")))
        random.seed(seed)
        random.shuffle(self.parquet_files)
        self.file_idx = 0
        self.rows: List[Dict] = []
        self.row_idx = 0
        self.exhausted = False

    def _load_next(self) -> bool:
        while self.file_idx < len(self.parquet_files):
            path = self.parquet_files[self.file_idx]
            self.file_idx += 1
            try:
                df = pq.read_table(path).to_pandas()
                self.rows = [row.to_dict() for _, row in df.iterrows()]
                random.shuffle(self.rows)
                self.row_idx = 0
                if self.rows:
                    return True
            except Exception:
                continue
        self.exhausted = True
        return False

    def get_next(self) -> Optional[Dict]:
        if self.exhausted:
            return None
        if self.row_idx >= len(self.rows):
            if not self._load_next():
                return None
        sample = self.rows[self.row_idx]
        self.row_idx += 1
        return sample


# ---------------------------------------------------------------------------
# Main sampling loop
# ---------------------------------------------------------------------------

def sample_finevision(
    finevision_path: str,
    subsets: List[str],
    target_tokens: int,
    tokenizer_path: str,
    images_dir: str,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 4,
) -> Tuple[List[Dict], int, int]:
    random.seed(seed)
    print(f"\nSampling from {len(subsets)} subsets, target {target_tokens:,} tokens...")
    print(f"Using {num_workers} workers, batch size {batch_size}")

    iterators = {s: SubsetIterator(os.path.join(finevision_path, s), seed) for s in subsets}
    active = list(iterators.keys())

    sampled: List[Dict] = []
    total_text_tokens = 0
    total_image_tokens = 0
    total_tokens = 0

    pool = mp.Pool(num_workers, initializer=_init_worker, initargs=(tokenizer_path, images_dir))
    pbar = tqdm(total=target_tokens, desc="Sampling tokens")

    try:
        while total_tokens < target_tokens and active:
            # Collect a batch of (sample, subset_name) pairs round-robin across subsets
            random.shuffle(active)
            batch: List[Tuple[Dict, str]] = []
            exhausted = []
            for subset in active:
                sample = iterators[subset].get_next()
                if sample is None:
                    exhausted.append(subset)
                    tqdm.write(f"Subset '{subset}' exhausted")
                else:
                    batch.append((sample, subset))
                    if len(batch) >= batch_size:
                        break
            for s in exhausted:
                active.remove(s)

            if not batch:
                continue

            results = pool.map(_process_sample, batch)

            for output, text_tok, img_tok in results:
                if total_tokens >= target_tokens:
                    break
                sampled.append(output)
                prev = total_tokens
                total_text_tokens += text_tok
                total_image_tokens += img_tok
                total_tokens = total_text_tokens + total_image_tokens
                pbar.update(total_tokens - prev)

    finally:
        pool.close()
        pool.join()
        pbar.close()

    return sampled, total_text_tokens, total_image_tokens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Sample FineVision dataset to match Instruct dataset token count")
    parser.add_argument("--finevision_path", type=str,
        default="/leonardo_work/AIFAC_5C0_261/datasets/train/finevision")
    parser.add_argument("--instruct_path", type=str,
        default="/leonardo_work/AIFAC_5C0_261/datasets/datasets/InstructDatasets/magpie.qwen3.32b.en.noreasoning.onlyconver.jsonl")
    parser.add_argument("--tokenizer_path", type=str,
        default=os.path.expandvars("$WORK/baseModels/Qwen3.5-9B"))
    parser.add_argument("--output_path", type=str, default="./finevision_sampled.jsonl")
    parser.add_argument("--images_dir", type=str, default="./finevision_images",
        help="Directory where extracted images are saved")
    parser.add_argument("--target_tokens", type=int, default=None,
        help="Override target token count (otherwise counts from Instruct dataset)")
    parser.add_argument("--batch_size", type=int, default=64,
        help="Number of samples per worker batch")
    parser.add_argument("--num_workers", type=int, default=max(1, mp.cpu_count() - 1),
        help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.target_tokens is not None:
        target_tokens = args.target_tokens
        print(f"Using provided target token count: {target_tokens:,}")
    else:
        target_tokens = count_instruct_tokens(
            args.tokenizer_path, args.instruct_path, args.num_workers
        )

    subsets = get_finevision_subsets(args.finevision_path)
    print(f"\nFound {len(subsets)} subsets (excluding text_*): {subsets}")

    os.makedirs(args.images_dir, exist_ok=True)
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    samples, text_tokens, image_tokens = sample_finevision(
        finevision_path=args.finevision_path,
        subsets=subsets,
        target_tokens=target_tokens,
        tokenizer_path=args.tokenizer_path,
        images_dir=args.images_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    print(f"\nSaving {len(samples):,} samples to {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    total_tokens = text_tokens + image_tokens
    multimodal = sum(1 for s in samples if s["images"])
    text_only  = len(samples) - multimodal

    print("\n" + "=" * 60)
    print("SAMPLING COMPLETE")
    print("=" * 60)
    print(f"Target tokens:      {target_tokens:>15,}")
    print(f"Total tokens:       {total_tokens:>15,}")
    print(f"  - Text tokens:    {text_tokens:>15,}")
    print(f"  - Image tokens:   {image_tokens:>15,}")
    print(f"Total samples:      {len(samples):>15,}")
    print(f"  - Multimodal:     {multimodal:>15,}")
    print(f"  - Text-only:      {text_only:>15,}")
    print(f"Output file:        {args.output_path}")
    print(f"Images dir:         {args.images_dir}")

    stats_path = args.output_path.replace(".jsonl", "_metadata.json")
    with open(stats_path, "w") as f:
        json.dump({
            "target_tokens": target_tokens,
            "total_tokens": total_tokens,
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "total_samples": len(samples),
            "multimodal_samples": multimodal,
            "text_only_samples": text_only,
            "subsets": subsets,
            "seed": args.seed,
        }, f, indent=2)
    print(f"Metadata saved:     {stats_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # required for transformers/CUDA
    main()
