import os
import argparse
from datasets import load_from_disk
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Path to the dataset saved with save_to_disk")
parser.add_argument("max_image_tokens", type=int, help="Max tokens per image (must match the ms-swift export)")
parser.add_argument("--model", default=os.path.join(os.environ.get("SCRATCH", os.path.expanduser("~/scratch")), "basemodels/Qwen3.5-9B"),
                    help="Model whose processor was used for the export")
args = parser.parse_args()
print(args.dataset)
ds = load_from_disk(args.dataset)
processor = AutoProcessor.from_pretrained(args.model)
tok, ip = processor.tokenizer, processor.image_processor

MIN_IMAGE_TOKENS = 4                            # IMAGE_MIN_TOKEN_NUM = 4 for Qwen3.5 in ms-swift
MAX_IMAGE_TOKENS = args.max_image_tokens
FACTOR = ip.patch_size * ip.merge_size          # 32
MAX_PIXELS = MAX_IMAGE_TOKENS * FACTOR ** 2
MIN_PIXELS = MIN_IMAGE_TOKENS * FACTOR ** 2

def to_content(s):
    parts = []
    for i, chunk in enumerate(s.split("<image>")):
        if i > 0: parts.append({"type": "image"})
        if chunk: parts.append({"type": "text", "text": chunk})
    return parts

def to_messages(msgs):
    return [{"role": m["role"], "content": to_content(m["content"])} for m in msgs]

def count(batch):
    texts, img_tokens = [], []
    for msgs, imgs in zip(batch["messages"], batch["images"]):
        assert sum(m["content"].count("<image>") for m in msgs) == len(imgs)
        texts.append(processor.apply_chat_template(to_messages(msgs), tokenize=False))
        n = 0
        for img in imgs:
            with Image.open(img["path"]) as im:     # reads header only, no decoding
                w, h = im.size
            rh, rw = smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
            n += (rh // FACTOR) * (rw // FACTOR)
        img_tokens.append(n)
    # minus one <|image_pad|> per image
    other = [len(ids) - len(imgs) for ids, imgs in zip(tok(texts).input_ids, batch["images"])]
    return {"img_tokens": img_tokens, "other_tokens": other}

# --- Verification (remove later) ---
N_CHECK = min(1000, len(ds))
fast = count(ds[:N_CHECK])
for i in range(N_CHECK):
    sample = ds[i]
    images = [Image.open(img["path"]).convert("RGB") for img in sample["images"]]
    text = processor.apply_chat_template(to_messages(sample["messages"]), tokenize=False)
    ids = processor(text=[text], images=images or None, return_tensors="pt",
                    size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS})["input_ids"]
    slow_img = (ids == processor.image_token_id).sum().item()
    slow_other = ids.shape[1] - slow_img

    assert (fast["img_tokens"][i], fast["other_tokens"][i]) == (slow_img, slow_other), \
        f"Sample {i}: fast=({fast['img_tokens'][i]}, {fast['other_tokens'][i]}) slow=({slow_img}, {slow_other})"
    assert slow_img + slow_other == sample["lengths"][0], \
        f"Sample {i}: ours={slow_img + slow_other} ms-swift={sample['lengths'][0]}"
print(f"Verification passed on {N_CHECK} samples")
# --- end verification ---

res = ds.map(count, batched=True, num_proc=os.cpu_count(), remove_columns=ds.column_names, load_from_cache_file=False)

total_img, total_other = sum(res["img_tokens"]), sum(res["other_tokens"])
print(f"Image tokens: {total_img}\nOther tokens: {total_other}")
print(f"Percentage image tokens {total_img / (total_img + total_other):.4f}")