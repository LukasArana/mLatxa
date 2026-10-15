import os
import time
from huggingface_hub import snapshot_download, constants

# 1. FORCE enable the Rust accelerator
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Verify it is enabled
print(f"HF_HUB_ENABLE_HF_TRANSFER: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")

# 2. Download (DATA_DIR is the HiTZ data root, e.g. /hitz_data/<user>)
FINEVISION_DIR = os.path.join(os.environ["DATA_DIR"], "finevision")
print("🚀 Starting accelerated download...")
start_time = time.time()

snapshot_download(
    cache_dir=FINEVISION_DIR,
    repo_id="HuggingFaceM4/FineVision",
    repo_type="dataset",
    local_dir=FINEVISION_DIR,
    max_workers=32

)

print(f"✅ Done! Total time: {time.time() - start_time:.2f}s")