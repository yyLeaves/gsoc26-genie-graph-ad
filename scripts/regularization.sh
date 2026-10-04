#!/usr/bin/env bash
set -euo pipefail

n_signal="${NSIG:-1000}"             # 0, 50, 100, 500, 1000
device="${DEVICE:-cuda:0}"           # cpu, cuda:0, cuda:1, ...
num_workers="${NUM_WORKERS:-8}"
runs_dir="${RUNS_DIR:-runs}"
signals=("$@")                       # default: all five signals
if (( ${#signals[@]} == 0 )); then
    signals=(LHCO XtoWRto3W XtoYYprime ZpToTpTp YtoHHto4T)
fi

cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for signal in "${signals[@]}"; do
    split="data/splits/CWoLa_${signal}_nsig${n_signal}.npz"
    output="$runs_dir/reg_calibration_cwola_nsig${n_signal}/seed42/$signal/reg"

    python -u -m scripts.train \
        --split-file "$split" --graph-dir data/graphs --sample SR \
        --model edgeconv --objective reconstruction --score total \
        --epochs 50 --eval-interval 5 --batch-size 256 --lr 0.003 \
        --edge-weight 1 --reg-weight 1 --seed 42 \
        --device "$device" --num-workers "$num_workers" --output-dir "$output"

    python -u -m scripts.inference \
        --checkpoint "$output/last.pt" --split-file "$split" \
        --graph-dir data/graphs --split test --score total \
        --batch-size 256 --device "$device" --num-workers "$num_workers" \
        --output "$output/test_scores.npz"
done
