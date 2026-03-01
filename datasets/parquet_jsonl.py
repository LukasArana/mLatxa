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
    print("Processing dataset:", dataset_path)
    # Load the dataset from the provided path
    if "jsonl" in dataset_path:
        # If it's a JSONL file, we can read it directly into a list of dicts
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
    else:
        dataset = load_dataset(dataset_path)

    if not isinstance(dataset, list):
        assert len(dataset) == 1, "Expected only one split in the dataset. Please check the dataset structure."
        dataset = dataset[0]
        # We'll assume we are processing the 'train' split based on your output
        split = 'train' if 'train' in dataset else list(dataset.keys())[0]
        dataset = dataset[split]
    mswift_data = []
    breakpoint()
    for idx, item in enumerate(dataset):
        # 1. Handle Images
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
            texts = dataset[idx]
        messages = []
        roles = ["system", "user", "assistant"]
        for i, turn in enumerate(texts):

            if isinstance(turn, str) or "text" in turn: # Data is pretraining style without explicit roles
                messages.append({
                    "role": "assistant",
                    "content": texts[turn]
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
    breakpoint()
    # Save as JSONL
    output_dir = "/leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/"
    output_file = os.path.join(output_dir, dataset_path.split('/')[-2], dataset_path.split('/')[-1])
    print(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in mswift_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Done! Processed {len(mswift_data)} rows into {output_file}")

# Usage
if __name__ == "__main__":
    datasets = [
    #    "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/aldizkariak/full.train.04_clean-01.onlytext.jsonl",
     #   "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/berria/berria-202509.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/bog/bog_euskera_18_09.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/booktegi/booktegi-bsc.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/bopv/bopv_eu_18_09.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/botha/botha_eu_18_09.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/cc-bsc/colossal_oscar_2023-14_eu.train.part-0001-of-0001.shuffled.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/colossal-oscar/05-06-23_eu_meta.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/colossal-oscar/06-07-22_eu_meta.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/cultura-x/eu.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/egunkaria_1999-2006/2001-2006.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/euscrawl_v1/euscrawl-v1.train.full.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/euscrawl_v1.202311/euscrawl-v1-2023.train.full.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/euscrawl_v1.202509_/euscrawl-v1-202509.train.full.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/euscrawl_v2/train.04_clean-01.onlytext.jsonl",
##        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/finepdf/train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/fineweb/full.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/hplt_v1/full.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/hplt_v2/full.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/OpenSubtitles/eu.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/parleus/parlamentu_db_final_eu.train.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/wikipedia/wikipedia.train.eu.04_clean-01.onlytext.jsonl",
#        "/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/iter-4/zelaihandi/zelaiHandi.train.trimmed.04_clean-01.onlytext.jsonl",
        "/leonardo_work/EUHPC_E04_042/datasets/InstructDatasets/magpie.qwen3.32b.en.noreasoning.jsonl",
        "/leonardo_work/EUHPC_E04_042/datasets/InstructDatasets/Magpie-Llama-3.1-70B-Instruct-Filtered-1M.jsonl"
    ]
    #/leonardo_work/EUHPC_E04_042/datasets/InstructDatasets/magpie.qwen3.32b.en.noreasoning.jsonl
    #
    # Replace "geo3k" with your local path if it's not the hub version
    for dataset in datasets:
        process_geo_to_mswift(dataset)