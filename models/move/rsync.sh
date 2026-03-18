#!/bin/bash
#SBATCH -A EUHPC_E04_042       # account name
#SBATCH -p lrd_all_serial
#SBATCH --time 4:00:00
#SBATCH --job-name=rsync_checkpoint_weights
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --output=.slurm/rsync_checkpoint_weights.out
#SBATCH --error=.slurm/rsync_checkpoint_weights.err
#SBATCH --mem=30800MB
#SBATCH --gres=tmpfs:200g

BASE_PATH="/leonardo_work/AIFAC_5C0_261/multimodalModels"
# REMOTE_PATH="igarcia945@xirimiri.ixa.eus:/proiektuak/ilenia-scratch/models-instruct"
REMOTE_PATH="larana@xirimiri.ixa.eus:/proiektuak/ilenia-scratch/models-instruct"
# MODELS=(
#     "llama-3.1-8B_1M_1epoch"
#     "llama-3.1-8B_1M_4epoch"
#     "llama-3.1-8B_Full_1epoch"
#     "llama-3.1-8B_Full_4epoch"
# )
# CHECKPOINTS=(
#     # 286
#     # 572
#     # 858
#     # 1144
#     # 1430
#     # 1716
#     # 2002
#     # 2288
#     2574
#     2860
# )

# MODEL="exp_0_010-norepeat_large"

CHECKPOINTS=(
    # 202
    # 404
    # 606
    # 808
    # 1010
    # # 1212
    # # 1414
    # # 1616
    # # 1818
    # 2020
    # 2222
    # 2424
    # 2626
    # 2828
    checkpoint-3900
)
MODEL="v30-20260308-195244"

echo "Starting rsync process..."
for CHKPT in "${CHECKPOINTS[@]}"; do
    #LOCAL_MODEL_PATH="${BASE_PATH}/${MODEL}/merged_checkpoint-${CHKPT}"
    LOCAL_MODEL_PATH="${BASE_PATH}/${MODEL}/${CHKPT}"
    REMOTE_MODEL_PATH="${REMOTE_PATH}/${MODEL}-${CHKPT}"

    echo "----------------------------------------"
    echo "Syncing model: ${MODEL}_checkpoint-${CHKPT}"
    echo "From: ${LOCAL_MODEL_PATH}"
    echo "To: ${REMOTE_MODEL_PATH}"

    # Create the remote directory structure
    # ssh igarcia945@xirimiri.ixa.eus "mkdir -p ${REMOTE_MODEL_PATH}"
    REMOTE_DIR="${REMOTE_MODEL_PATH#*:}"
    echo "Creating remote directory: ${REMOTE_DIR}"
    ssh larana@xirimiri.ixa.eus "mkdir -p ${REMOTE_DIR}"
    echo "a"
    # Rsync with optimized parameters for large files
    rsync -PravzHS --info=progress2 \
        --compress-level=9 \
        --exclude iter_0005750 \
        --partial-dir=/leonardo_work/AIFAC_5C0_261/multimodalModels/.rsync-partial \
        ${LOCAL_MODEL_PATH}/ \
        ${REMOTE_MODEL_PATH}/

    if [ $? -eq 0 ]; then
        echo "Successfully synced ${MODEL}_checkpoint-${CHKPT}"
    else
        echo "Error syncing ${MODEL}_checkpoint-${CHKPT}"
        exit 1
    fi
done

echo "----------------------------------------"
echo "All models synced successfully!"