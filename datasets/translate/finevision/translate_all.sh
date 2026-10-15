#!/bin/bash
set -euo pipefail

NJOBS="${1:?usage: ./translate_all.sh NJOBS LANGUAGE [INPUT]}"
LANGUAGE="${2:?usage: ./translate_all.sh NJOBS LANGUAGE [INPUT]}"
INPUT="${3:-$WORK/datasets/train/finevisionProcessed/train.jsonl}"
echo $LANGUAGE
case "$LANGUAGE" in
    eu)
        MODEL="$WORK/baseModels/Latxa70B"
        ;;
    cat)
        MODEL="$WORK/baseModels/Qwen3.5-27B/"
        ;;
    gal)
        MODEL="$WORK/baseModels/Qwen3.5-27B/"
        ;;
    nor)
        MODEL="$WORK/baseModels/Qwen3.5-27B/"
        ;;
    *)
        echo "Unsupported language: $LANGUAGE" >&2
        echo "Supported languages: eu, cat, gal, nor" >&2
        exit 1
        ;;
esac

TOTAL=$(wc -l < "$INPUT")
PER=$(( (TOTAL + NJOBS - 1) / NJOBS ))

echo "Language: $LANGUAGE"
echo "Model: $MODEL"
echo "Input: $INPUT"
echo "Total: $TOTAL | Jobs: $NJOBS | Per job: $PER"
mkdir -p logs

SUBMITTED=0
for (( i = 0; i < NJOBS; i++ )); do
    START=$(( i * PER ))
    [ "$START" -ge "$TOTAL" ] && break
    END=$(( START + PER < TOTAL ? START + PER : TOTAL ))

    JOB_ID=$(sbatch --parsable \
        --nodes=1 \
        --gres=gpu:4 \
        --ntasks=1 \
        --cpus-per-task=32 \
        --partition=boost_usr_prod \
        --qos=normal \
        --time=12:00:00 \
        --account=AIFAC_5C0_261 \
        --job-name="mt_${START}_${END}" \
        --output="logs/mt_${START}_${END}_${LANGUAGE}.out" \
        --error="logs/mt_${START}_${END}_${LANGUAGE}.err" \
        "$HOME/mLatxa/datasets/translate/finevision/translate.sh" \
        "$START" "$END" "$MODEL" "$LANGUAGE" "$INPUT")

    SUBMITTED=$((SUBMITTED + 1))
    echo "  submitted job ${JOB_ID} for [$START, $END)"
done

echo "Done: submitted ${SUBMITTED} jobs to Slurm queue."