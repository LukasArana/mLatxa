#!/usr/bin/env python
"""Rewrite MN5 image paths in cached multimodal datasets copied to Leonardo.

The MN5 caches store absolute image paths under
/home/ehu/$MN5_USER/scratch/data/leonardo/images_v1/, which were copied from
Leonardo's finevisionProcessed/images/ (same file names). This points them back
at the Leonardo copy. Idempotent: a dataset with a .leonardo_paths marker is skipped.

    MN5_USER=<mn5 user> python fix_image_paths.py <cache_dir/train> [<cache_dir/train> ...]
"""
import json
import os
import random
import shutil
import sys

from datasets import load_from_disk
from datasets.utils.logging import enable_progress_bar
from tqdm.auto import tqdm

enable_progress_bar()  # map / save_to_disk show their own per-step bars

OLD = f"/home/ehu/{os.environ['MN5_USER']}/scratch/data/leonardo/images_v1/"  # MN5 home of whoever exported the caches
NEW = "/leonardo_work/AIFAC_5C0_261/datasets/train/finevisionProcessed/images/"
MARKER = ".leonardo_paths"


def fix(batch):
    out = []
    for imgs in batch["images"]:
        if imgs:
            imgs = [dict(i, path=i["path"].replace(OLD, NEW, 1)) if i.get("path") else i for i in imgs]
        out.append(imgs)
    return {"images": out}


overall = tqdm(sys.argv[1:], desc="datasets", unit="ds", position=0, dynamic_ncols=True)
for path in overall:
    path = path.rstrip("/")
    overall.set_postfix_str(os.path.basename(os.path.dirname(path)))
    if os.path.exists(os.path.join(path, MARKER)):
        tqdm.write(f"{path}: already rewritten, skipping")
        continue
    tqdm.write(f"{path}: loading")
    ds = load_from_disk(path)
    n_shards = len(json.load(open(os.path.join(path, "state.json")))["_data_files"])
    new = ds.map(fix, batched=True, batch_size=10000, num_proc=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), features=ds.features,
                 desc=f"rewrite {os.path.basename(os.path.dirname(path))}")

    # Every path must now be on Leonardo, and a random sample must exist.
    paths = [i["path"] for x in new.select(random.sample(range(len(new)), min(5000, len(new))))["images"]
             if x for i in x if i.get("path")]
    bad_prefix = [p for p in paths if not p.startswith(NEW)]
    missing = [p for p in tqdm(paths, desc="check images exist", unit="img", leave=False)
               if p.startswith(NEW) and not os.path.exists(p)]
    if bad_prefix or missing or not paths:
        sys.exit(f"{path}: {len(paths)} sampled paths, {len(bad_prefix)} unexpected prefix "
                 f"(e.g. {bad_prefix[:2]}), {len(missing)} missing (e.g. {missing[:2]}) — NOT replacing")

    tmp = path + ".tmp_leonardo"
    shutil.rmtree(tmp, ignore_errors=True)
    tqdm.write(f"{path}: checks passed, saving {n_shards} shards")
    new.save_to_disk(tmp, num_shards=n_shards)
    old = path + ".mn5_paths"
    os.rename(path, old)
    os.rename(tmp, path)
    open(os.path.join(path, MARKER), "w").write(f"{OLD} -> {NEW}\n")
    shutil.rmtree(old)
    tqdm.write(f"{path}: {len(new)} rows rewritten, {len(paths)} sampled image paths exist")
