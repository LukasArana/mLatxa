import json
import base64
from io import BytesIO
from PIL import Image
from datasets import load_dataset
import os
def convert_to_base64(pil_image):
    """Converts a PIL Image object to a base64 string."""
    buffered = BytesIO()
    # FineVision/Geo3k often use JPEG/PNG; we'll save as JPEG for compatibility
    pil_image.convert("RGB").save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpg;base64,{img_str}"

def process_geo_to_mswift(dataset_path):
    # Load the dataset from the provided path
    if "jsonl" in dataset_path:
        # If it's a JSONL file, we can read it directly into a list of dicts
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f][0]
    else:
        dataset = load_dataset(dataset_path)

    # We'll assume we are processing the 'train' split based on your output
    split = 'train' if 'train' in dataset else list(dataset.keys())[0]

    mswift_data = []

    for idx, item in enumerate(dataset[split]):
        # 1. Handle Images
        # Geo3k 'images' feature is usually a list of PIL Image objects
        base64_images = []
        if 'images' in item:
            multimodal = True
            texts = item.get('texts', [])
            for img in item['images']:
                if isinstance(img, Image.Image):
                    base64_images.append(convert_to_base64(img))
                else:
                    # If it's already a path or string, handle accordingly
                    base64_images.append(img)
        else:
            multimodal = False
            texts = dataset
        # 2. Handle Messages
        # FineVision/Geo3k 'texts' is often a list of strings or a conversation list
        # Based on typical MSWIFT structure:
        messages = []
        roles = ['user', 'assistant']  # Assuming alternating roles; adjust if your dataset has explicit roles
        # Construct the conversation. If 'texts' is a simple list of [User, Assistant]
        for i, turn in enumerate(texts): # Assuming texts[0] is a list of alternating User and Assistant messages
            print(idx)
            if isinstance(turn, str): # Data is pretraining style without explicit roles
                turn = texts[turn]
                messages.append({
                    "role": "assistant",
                    "content": turn
                })
                continue

            for role in roles:
                if not role in turn:
                    continue
                content = turn[role]
                # Ensure the <image> tag is present for the first message
                # If multiple images, repeat <image> tags
                if i == 0 and role == 'user' and multimodal:
                    image_tags = "<image>" * len(base64_images)
                    content = f"{image_tags}{content}"

                messages.append({
                    "role": role,
                    "content": content
                })

        mswift_entry = {
            "messages": messages
        }
        if multimodal:
            mswift_entry["images"] = base64_images

        mswift_data.append(mswift_entry)
    # Save as JSONL
    output_dir = "/leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/"
    output_file = os.path.join(output_dir, dataset_path.split('/')[-1])
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in mswift_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Done! Processed {len(mswift_data)} rows into {output_file}")

# Usage
if __name__ == "__main__":
    # Replace "geo3k" with your local path if it's not the hub version
    process_geo_to_mswift("/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/euscrawl_v2/train.04_clean-01.onlytext.jsonl")