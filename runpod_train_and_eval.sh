#!/usr/bin/env bash
# Train all 5 CV folds + fold_all in parallel across 6x A100, predict on the
# held-out test set, evaluate, then shut the pod down -- whether this
# succeeds, fails, or is interrupted (see the trap below).
set -uo pipefail

# --- Config: adjust these for your pod ---
export nnUNet_raw=/workspace/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnUNet_preprocessed
export nnUNet_results=/workspace/nnUNet_results

DATASET=4               # Dataset004_cSDH_Abstract -- the one with imagesTs/labelsTs
CONFIG=3d_fullres
OUT=/workspace/eval_$(date +%Y-%m-%d)

# Shut the pod down on ANY exit path (success, failure, or Ctrl-C) so a
# crash mid-script doesn't leave the meter running.
trap 'echo "[shutdown] stopping pod $RUNPOD_POD_ID"; runpodctl stop pod "$RUNPOD_POD_ID"' EXIT

# --- 1. Preprocess (one-time; safe to rerun, nnU-Net skips finished steps) ---
nnUNetv2_plan_and_preprocess -d "$DATASET" --verify_dataset_integrity

# --- 2. Train all 5 folds + fold_all, one per A100, fully in parallel ---
pids=()
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train "$DATASET" "$CONFIG" 0   & pids+=($!)
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train "$DATASET" "$CONFIG" 1   & pids+=($!)
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train "$DATASET" "$CONFIG" 2   & pids+=($!)
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train "$DATASET" "$CONFIG" 3   & pids+=($!)
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train "$DATASET" "$CONFIG" 4   & pids+=($!)
CUDA_VISIBLE_DEVICES=5 nnUNetv2_train "$DATASET" "$CONFIG" all & pids+=($!)

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

if [ "$fail" -ne 0 ]; then
    echo "[error] one or more training jobs failed -- skipping predict/eval" >&2
    exit 1
fi

# --- 3. Predict on the held-out test set (CV ensemble + fold_all separately) ---
mkdir -p "$OUT/predictions_cv" "$OUT/predictions_foldall"
IMAGES_TS="$nnUNet_raw/Dataset004_cSDH_Abstract/imagesTs"
LABELS_TS="$nnUNet_raw/Dataset004_cSDH_Abstract/labelsTs"

nnUNetv2_predict -i "$IMAGES_TS" -o "$OUT/predictions_cv"      -d "$DATASET" -c "$CONFIG" -f 0 1 2 3 4 || exit 1
nnUNetv2_predict -i "$IMAGES_TS" -o "$OUT/predictions_foldall" -d "$DATASET" -c "$CONFIG" -f all       || exit 1

# --- 4. Evaluate (assumes nnunet-eval is installed on the pod) ---
nnunet-eval compare \
    -m cv_ensemble "$OUT/predictions_cv" \
    -m fold_all "$OUT/predictions_foldall" \
    --labels "$LABELS_TS" \
    --dataset-json "$nnUNet_raw/Dataset004_cSDH_Abstract/dataset.json" \
    --metric dice --metric hd95 --metric assd --metric cl_dice \
    -o "$OUT/comparison_results.csv" \
    -s "$OUT/comparison_stats.csv" || exit 1

echo "[done] results in $OUT"
# pod shutdown happens via the trap above
