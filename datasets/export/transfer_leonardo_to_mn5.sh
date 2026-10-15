#!/bin/bash
#SBATCH --job-name=transfer_leonardo_to_mn5
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=%u_transfer_%j.out
#SBATCH --error=%u_transfer_%j.err

set -euo pipefail

echo "================================"
echo "Leonardo to MN5 Data Transfer"
echo "Started: $(date)"
echo "================================"
echo ""

# Transfer iter-4 (Basque dataset)
echo "=== Transferring iter-4 (Basque datasets) ==="
rsync -avP --checksum ${EUHPC_WORK:-/leonardo_work/EUHPC_E04_042}/datasets/PretrainDatasets/iter-4/ ${MN5_USER:?set MN5_USER}@login.mn5.bsc.es:/gpfs/scratch/ehpc678/leonardo_data/iter-4/
echo "✓ iter-4 complete"
echo ""

# Transfer gl-2b (Galician dataset)
echo "=== Transferring gl-2b (Galician datasets) ==="
rsync -avP --checksum ${EUHPC_WORK:-/leonardo_work/EUHPC_E04_042}/datasets/PretrainDatasets/gl-2b/ ${MN5_USER:?set MN5_USER}@login.mn5.bsc.es:/gpfs/scratch/ehpc678/leonardo_data/gl-2b/
echo "✓ gl-2b complete"
echo ""

# Transfer ca-2b (Catalan dataset)
echo "=== Transferring ca-2b (Catalan datasets) ==="
rsync -avP --checksum ${EUHPC_WORK:-/leonardo_work/EUHPC_E04_042}/datasets/PretrainDatasets/ca-2b/ ${MN5_USER:?set MN5_USER}@login.mn5.bsc.es:/gpfs/scratch/ehpc678/leonardo_data/ca-2b/
echo "✓ ca-2b complete"
echo ""

echo "================================"
echo "Transfer completed: $(date)"
echo "================================"
