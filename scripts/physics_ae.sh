#!/usr/bin/env bash
set -euo pipefail

# Physics-informed AE trained on LHCO, frozen, then a CWoLa tree readout for each signal.
seed="${SEED:-42}"
device="${DEVICE:-cpu}"                # cpu, cuda:0, ...
run_dir="${RUN_DIR:-runs/physics_ae}"
signals=("$@")                         # default: all five signals
if (( ${#signals[@]} == 0 )); then
    signals=(LHCO XtoWRto3W XtoYYprime ZpToTpTp YtoHHto4T)
fi

cd "$(dirname "$0")/.."

for signal in LHCO "${signals[@]}"; do
    if [[ ! -f "data/packed/$signal/test.pt" ]]; then
        python -m scripts.physics_ae pack --signal "$signal"
    fi
done

# ae.pt is written only after the last epoch; an interrupted run leaves last.pt and is retrained.
if [[ ! -f "$run_dir/seed$seed/ae.pt" ]]; then
    python -u -m scripts.physics_ae train --seed "$seed" --device "$device" --run-dir "$run_dir"
fi

for signal in "${signals[@]}"; do
    python -u -m scripts.physics_ae evaluate --signal "$signal" --seed "$seed" --device "$device" --run-dir "$run_dir"
done
