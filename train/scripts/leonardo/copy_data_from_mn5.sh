#!/bin/bash
# Copy the cached datasets used by the MN5-ported scripts in scripts/leonardo/ from MN5 to Leonardo.
# Run on your LOCAL machine (it relays: MN5 -> local staging -> Leonardo), since
# neither cluster can ssh to the other. Uses the data-transfer nodes on both ends
# (no CPU-time limit like the login nodes). Re-running resumes; finished datasets
# are skipped by rsync.
#
#   LEO_USER=<leonardo user> bash copy_data_from_mn5.sh              # all datasets
#   LEO_USER=<leonardo user> bash copy_data_from_mn5.sh v2_ca v2_gl  # only some
#
# Needs ~110 GB free in $STAGING (each dataset is removed from staging once it
# is verified on Leonardo; set KEEP_STAGING=1 to keep it).
set -euo pipefail

MN5_SRC="transfer:/gpfs/scratch/ehu26/data"   # = ~/scratch/data on MN5
LEO_HOST="${LEO_USER:?set LEO_USER (Leonardo username)}@data.leonardo.cineca.it"
LEO_DST="/leonardo_work/AIFAC_S07_127/lukas/data_mn5"   # = $DATA_DIR in the sbatch scripts
STAGING="${STAGING:-$HOME/mn5_to_leonardo_staging}"
LEO_SSH="ssh -i $HOME/.ssh/cineca -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# sizes on MN5: v2 23G, v2_ca 15G, v2_gl 15G, v2_no 26G, multimodal_v1 12G,
#               v1_multimodal_eu_en 13G, multimodal_124 1.3G, multimodal_eu_124 1.2G
DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(v2 v2_ca v2_gl v2_no multimodal_v1 v1_multimodal_eu_en multimodal_124 multimodal_eu_124)

mkdir -p "$STAGING"
for d in "${DATASETS[@]}"; do
    echo "=== $d: MN5 -> local"
    mkdir -p "$STAGING/$d"
    rsync -a --partial --info=progress2 "$MN5_SRC/$d/train/" "$STAGING/$d/train/"

    echo "=== $d: local -> Leonardo"
    # Leonardo has rsync 3.1.3 (no --mkpath) and the data node only allows
    # rsync/scp (no mkdir), so --relative with the "/./" anchor makes rsync
    # create $LEO_DST/$d/train itself.
    rsync -a --relative --partial --info=progress2 -e "$LEO_SSH" \
        "$STAGING/./$d/train/" "$LEO_HOST:$LEO_DST/"

    # Verify: a checksum dry-run must report nothing left to transfer.
    pending=$(rsync -a --checksum --dry-run --itemize-changes -e "$LEO_SSH" \
        "$STAGING/$d/train/" "$LEO_HOST:$LEO_DST/$d/train/" | grep -c '^[<>c]' || true)
    if [ "$pending" != "0" ]; then
        echo "!!! $d: $pending files differ on Leonardo after copy, keeping staging" >&2
        exit 1
    fi
    echo "=== $d: verified"
    [ "${KEEP_STAGING:-0}" = "1" ] || rm -rf "${STAGING:?}/$d"
done
echo "All done: ${DATASETS[*]}"
echo "NEXT (on Leonardo): the multimodal caches still hold MN5 image paths. Run, in the training env:"
echo "  MN5_USER=<mn5 user> python ~/mLatxa/train/scripts/leonardo/fix_image_paths.py $LEO_DST/{multimodal_v1,v1_multimodal_eu_en,multimodal_124,multimodal_eu_124}/train"
