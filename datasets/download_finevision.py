import os
import time
from huggingface_hub import snapshot_download, constants

# 1. FORCE enable the Rust accelerator
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Verify it is enabled
print(f"HF_HUB_ENABLE_HF_TRANSFER: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")

# 2. Download
print("🚀 Starting accelerated download...")
start_time = time.time()

snapshot_download(
    cache_dir = "${DATA_DIR}/finevision/.",
    repo_id="HuggingFaceM4/FineVision",
    repo_type="dataset",
    local_dir="${DATA_DIR}/finevision/.",
    local_dir_use_symlinks=False, # Critical for speed on clusters
    resume_download=True
)

print(f"✅ Done! Total time: {time.time() - start_time:.2f}s")