# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

`mLatxa/train/` holds the SLURM launch scripts and training configs for continued-pretraining / SFT of Latxa (Basque-focused) LLMs and Qwen3-VL multimodal variants on **CINECA Leonardo** (Boost partition). It is not a Python package — every file is either a `sbatch` script or a YAML/JSON config consumed by **ms-swift** (vendored at `~/ms-swift`, not in this repo).

The wider repo lives at `mLatxa/` and is split:
- `mLatxa/datasets/` — dataset construction: FineVision multimodal pipeline + `swift export --to_cached_dataset` to produce the cached datasets the train scripts consume.
- `mLatxa/train/` — this directory.
- `mLatxa/models/` — post-training: HF↔mcore conversion (`export_qwen-32_megatron.sh`) and rsync of checkpoints to `xirimiri.ixa.eus` (`move/rsync.sh`).

## Running training

All entry points are `sbatch` scripts. There is no test suite, build step, or linter.

```bash
# Primary (Megatron backend, current path):
sbatch scripts/megatron/Qwen3.5-VL-9B-Instruct_v2.sh
sbatch scripts/megatron/Qwen3.5-VL-27B-Instruct_v2.sh
sbatch scripts/megatron/Qwen3-VL-32B-Instruct_v2.sh

# Single-node smoke test on the debug QoS (`boost_qos_dbg`):
sbatch scripts/megatron/train_single_node_multimodal.sh

# Legacy (HF Trainer + DeepSpeed ZeRO-3, kept for reference, uses configs/*.yaml):
sbatch scripts/train_multinode.sh
```

Logs land in `logs/%j.out` / `logs/%j.err` relative to the submission cwd.

## Architecture you need to know to be productive

### Two training backends, do not mix them
- **Megatron (current)** — scripts in `scripts/megatron/`. Invoke `accelerate launch` over `~/ms-swift/swift/cli/_megatron/sft.py` with Megatron flags (`tensor_model_parallel_size`, `pipeline_model_parallel_size`, `recompute_*`, `attention_backend flash`, `cross_entropy_loss_fusion`, `sequence_parallel`). Reads **mcore-format weights** when `--mcore_model <ckpt>` is set; otherwise auto-converts from `--model <HF path>` on first run. Outputs distributed mcore checkpoints.
- **HF/DeepSpeed (legacy)** — `scripts/train_multinode.sh` + `configs/train_32B.yaml` + `configs/train_32B_cache.yaml` + `scripts/deepspeed_conf.json`. Uses `swift/cli/sft.py` (not `_megatron/sft.py`), ZeRO-3, `attn_impl flash_attention_2`, `use_liger_kernel`. `configs/train_32B_megatron.yaml` is a half-migrated YAML form of the Megatron flags and is not currently wired into any script.

When editing or copying a script, keep the backend's flag dialect consistent: `attention_backend` (Megatron) vs `attn_impl` (HF), `cached_dataset` vs `dataset`, `lr` vs `learning_rate`, `micro_batch_size` vs `per_device_train_batch_size`, `lr_warmup_fraction` vs `warmup_ratio`.

### Datasets are always pre-cached
Train scripts never load raw JSONL. They point at an already-tokenized `cached_dataset` directory under `/leonardo_work/AIFAC_5C0_261/datasets/train/preprocessed/...`. The cache is built one-off by `mLatxa/datasets/export/scripts/export_v*.sh`, which runs `swift export --to_cached_dataset` against a `custom_dataset_info` JSON in `mLatxa/datasets/export/configs/`. Dataset versions in use:
- `v1/train` — text-only Basque corpora + magpie (Qwen3.5-9B tokenizer).
- `multimodal_v1/train` — v1 plus the FineVision-derived multimodal data.
- `v2/qwen32b/train`, `latxa_v2/qwen32b/train` — expanded corpora, Qwen3-VL-32B tokenizer.

The multimodal source JSONL is produced by `mLatxa/datasets/finevision_pipeline.py` (FineVision → quality filter → token-budgeted sample → MS-Swift JSONL with `messages` + `images` paths).

### Leonardo-specific launch boilerplate (do not strip)
Every multi-node script repeats the same prologue and you must keep it when adding new scripts:
- Env activation: `source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9_megratron/bin/activate` (Megatron) or `.../env_torch_2_9/...` (HF path for dataset export).
- Caches relocated off `$HOME` (small quota): `TMPDIR`, `HF_HOME`, `MODELSCOPE_CACHE`, `WANDB_DIR` under `/leonardo_work/AIFAC_5C0_261/`. `WANDB_MODE=offline` — compute nodes have no internet; wandb is rsynced later.
- NCCL/IB tuning required to avoid hangs on Boost: `NCCL_IB_HCA=mlx5_0..3`, `NCCL_IB_GID_INDEX=3`, `NCCL_NET_GDR_LEVEL=2`, `NCCL_NET_DISABLE_INTRA=1`, `GLOO_SOCKET_IFNAME=ib0`. Multimodal jobs raise `NCCL_TIMEOUT` / `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` to 28800–57600 because dataset warm-up takes a long time.
- ms-swift Qwen3-VL knobs: `SWIFT_PATCH_CONV3D=1` (always), `SWIFT_USE_MCORE_GDN=1` for Megatron, `video_min_token_num=0`, `video_max_token_num=0`, `MAX_PIXELS=1003520`.
- Rendezvous: derive `MAIN_PROCESS_IP` from `scontrol show hostnames`, fixed `MASTER_PORT=9327`, `--rdzv_backend c10d`, `--machine_rank $SLURM_NODEID`.
- SLURM header: `--account=AIFAC_5C0_261`, `--partition=boost_usr_prod`, `--gres=gpu:4`, `--exclusive`, `--ntasks-per-node=1` (the single task fans out via `accelerate launch --num_processes`). Single-node debug scripts add `--qos=boost_qos_dbg`.

### Base models and outputs
- Base weights: `/leonardo_work/AIFAC_5C0_261/baseModels/` (Qwen3.5-9B, Qwen3.5-27B, gemma-4-E4B-it) and `/leonardo_work/EUHPC_E04_042/BaseModels/` (Qwen3-VL-32B-Instruct, Qwen3.5-9B-Instruct).
- Checkpoints land in `/leonardo_work/AIFAC_5C0_261/multimodalModels/v{N}-{YYYYMMDD-HHMMSS}/checkpoint-{step}`. To resume / continue training, pass that path via `--mcore_model` (Megatron) — `--model` still points at the original HF weights for tokenizer/config.
- After training, convert mcore→HF with `mLatxa/models/export_qwen-32_megatron.sh` (calls `swift export --to_mcore`/the inverse) before rsyncing with `mLatxa/models/move/rsync.sh`.

## Conventions when adding a new training script

Copy the closest existing `scripts/megatron/<Model>_v{N}.sh` rather than starting from scratch — the NCCL/IB block, env activation, and rendezvous setup are load-bearing. Typical things that change between scripts: `--nodes` (4/16/32), `--model`, `--mcore_model` (only when resuming), `--cached_dataset`, `tensor_model_parallel_size`, `micro_batch_size`, `recompute_num_layers`, `freeze_llm` / `freeze_vit` / `freeze_aligner`, and a fresh `WANDB_PROJECT` if it is a new model family.
