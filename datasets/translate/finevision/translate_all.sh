#!/bin/bash
set -euo pipefail

NJOBS="${1:?usage: ./translate_all.sh NJOBS LANGUAGE [INPUT]}"
LANGUAGE="${2:?usage: ./translate_all.sh NJOBS LANGUAGE [INPUT]}"
INPUT="${3:-$WORK/datasets/train/finevisionProcessed/train.jsonl}"
echo $LANGUAGE
case "$LANGUAGE" in
    eu)
        MODEL="/leonardo_work/EUHPC_E04_042/LatxaTxat_models/InstructionModels/Latxa-Llama-3.1-70B-Instruct-exp_2_101_v2/"
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

# ...existing code...
for (( i = 0; i < NJOBS; i++ )); do
    START=$(( i * PER ))
    [ "$START" -ge "$TOTAL" ] && break
    END=$(( START + PER < TOTAL ? START + PER : TOTAL ))

    srun \
        --nodes=1 \
        --gres=gpu:4 \
        --exclusive \
        --partition=boost_usr_prod \
        --qos=normal \
        --time=12:00:00 \
        --cpus-per-task=32 \
        --account=AIFAC_5C0_261 \
        --job-name="mt_${START}_${END}" \
        bash -lc "$HOME/mLatxa/datasets/translate/finevision/translate.sh $START $END \"$MODEL\" \"$LANGUAGE\" \"$INPUT\"" \
        >"logs/mt_${START}_${END}_${LANGUAGE}.out" \
        2>"logs/mt_${START}_${END}_${LANGUAGE}.err" &

    echo "  submitted [$START, $END)"
done
wait