"""
Convert the finevision dataset to MS-Swift standard JSONL format.

Input format (finevision):
{
    'images': [<PIL Image>],
    'texts': [
        {'user': '...', 'assistant': '...'},
        {'user': '...', 'assistant': '...'},
        ...
    ],
    'source': '...',
    ...rating fields...
}

Output format (MS-Swift standard messages + images):
{"messages": [{"role": "user", "content": "<image>\nHow many hydroxyl groups..."}, {"role": "assistant", "content": "By examining..."}], "images": ["/abs/path/to/image.png"]}
{"messages": [{"role": "user", "content": "<image>\nWhat is the molecular..."}, {"role": "assistant", "content": "The molecular..."}], "images": ["/abs/path/to/image.png"]}

Each text pair becomes its own JSONL row, all referencing the same saved image.
The <image> tag in the user content tells MS-Swift where to insert image features.
"""

import json
import os
import argparse
from pathlib import Path

# Requires: pip install datasets Pillow
from datasets import load_dataset


def convert_finevision(
    dataset_name: str = "finevision",          # HF dataset name or local path
    output_dir: str = "./finevision_swift",     # Where to save JSONL + images
    split: str = "train",
    min_image_correspondence: int = 0,         # Filter: minimum image_correspondence rating
    min_visual_dependency: int = 0,            # Filter: minimum visual_dependency rating
    subset: str = None,                        # HF dataset subset/config if needed
):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Load dataset - adjust this to match how you load finevision
    if subset:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    output_path = os.path.join(output_dir, "train.jsonl")
    total_written = 0
    total_skipped = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        for idx, sample in enumerate(ds):
            images = sample.get("images", [])
            texts = sample.get("texts", [])
            image_corr_ratings = sample.get("image_correspondence_ratings", [])
            visual_dep_ratings = sample.get("visual_dependency_ratings", [])

            if not images or not texts:
                total_skipped += 1
                continue

            # Save the image(s) to disk - MS-Swift needs file paths, not PIL objects
            image_paths = []
            for img_idx, img in enumerate(images):
                img_filename = f"sample_{idx:06d}_img_{img_idx}.png"
                img_path = os.path.join(images_dir, img_filename)
                if not os.path.exists(img_path):
                    img.save(img_path)
                image_paths.append(os.path.abspath(img_path))

            # Each text pair becomes a separate training example
            # All pairs reference the same image(s)
            for text_idx, text_pair in enumerate(texts):
                user_text = text_pair.get("user", "")
                assistant_text = text_pair.get("assistant", "")

                if not user_text or not assistant_text:
                    total_skipped += 1
                    continue

                # Optional: filter by quality ratings
                if image_corr_ratings and text_idx < len(image_corr_ratings):
                    if image_corr_ratings[text_idx] < min_image_correspondence:
                        total_skipped += 1
                        continue

                if visual_dep_ratings and text_idx < len(visual_dep_ratings):
                    if visual_dep_ratings[text_idx] < min_visual_dependency:
                        total_skipped += 1
                        continue

                # Build the <image> tags - one per image in the sample
                image_tags = "".join(["<image>"] * len(image_paths))

                # MS-Swift standard messages format
                swift_sample = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{image_tags}\n{user_text}"
                        },
                        {
                            "role": "assistant",
                            "content": assistant_text
                        }
                    ],
                    "images": image_paths
                }

                f_out.write(json.dumps(swift_sample, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"Conversion complete!")
    print(f"  Written: {total_written} examples")
    print(f"  Skipped: {total_skipped} examples")
    print(f"  Output:  {output_path}")
    print(f"  Images:  {images_dir}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert finevision to MS-Swift format")
    parser.add_argument("--dataset_name", type=str, default="finevision",
                        help="HF dataset name or local path")
    parser.add_argument("--output_dir", type=str, default="./finevision_swift",
                        help="Output directory for JSONL and images")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset", type=str, default=None,
                        help="HF dataset config/subset name")
    parser.add_argument("--min_image_correspondence", type=int, default=0,
                        help="Minimum image_correspondence_rating to include a pair")
    parser.add_argument("--min_visual_dependency", type=int, default=0,
                        help="Minimum visual_dependency_rating to include a pair")
    args = parser.parse_args()

    convert_finevision(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        split=args.split,
        subset=args.subset,
        min_image_correspondence=args.min_image_correspondence,
        min_visual_dependency=args.min_visual_dependency,
    )