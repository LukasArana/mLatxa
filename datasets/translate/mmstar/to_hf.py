import base64
from datasets import Dataset, Features, Value, Image

def b64_to_bytes(b64_str):
    return base64.b64decode(b64_str)

# Load JSONL
dataset = Dataset.from_json("/home/larana/mLatxa/datasets/translate/output/MMStar_eu/translated_0_-1.jsonl")

# Map: convert base64 string to bytes (Image will wrap as {"bytes": ..., "path": None})
dataset = dataset.map(
    lambda ex: {"image": b64_to_bytes(ex["image"])},
    num_proc=8
)

# Set column type explicitly
features = dataset.features.copy()
features["image"] = Image()  # dtype: image
dataset = dataset.cast(features)
breakpoint()
# Push to Hub
dataset.push_to_hub("lukasArana/mmstar_eu")
