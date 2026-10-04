#!/usr/bin/env bash
set -e

split_file="${SPLIT_FILE:-data/splits/CWoLa_LHCO_nsig1000.npz}"
sample="${SAMPLE:-SR}"                 # SR, reference
model="${MODEL:-edgeconv}"             # edgeconv, gladc, netge, sdm_nat
reg_weight="${REG_WEIGHT:-0}"          # 0 = baseline, 1 = EB3
run_dir="${RUN_DIR:-runs/edgeconv_sr}"  # use a separate folder for each run
device="${DEVICE:-cpu}"                # cpu, cuda:0, ...
resume="${RESUME:-}"                   # optional last.pt; keeps saved settings

cd "$(dirname "$0")/.."

if [[ -n "$resume" ]]; then
    exec python -m scripts.train --resume "$resume" "$@"
fi

exec python -m scripts.train \
    --split-file "$split_file" \
    --sample "$sample" \
    --model "$model" \
    --reg-weight "$reg_weight" \
    --output-dir "$run_dir" \
    --device "$device" "$@"
