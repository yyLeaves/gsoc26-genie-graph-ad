#!/usr/bin/env bash
# Best-result pipeline: EdgeGraphAE edge-reconstruction anomaly detection.
#
# Reproduces our current strongest LHCO W'→XY event-level result
# (AUC ~0.928, max-SIC ~2.80) with the Araz et al. (2506.19920)
# reference-style recipe:
#   jets pT>1.2 TeV → 30 exclusive-kT subjets → unique-k graph →
#   LOG edge features (lnΔR, ln k_T, ln z) → EdgeGraphAE (node+edge recon, no BN)
#   → AdamW(lr=3e-3, wd=0.01) + linear OneCycleLR, 50 epochs, no early stop
#   → event score = sum of selected-jet reconstruction errors.
#
# Usage:
#   bash develop/scripts/pipeline_relae.sh                 # full pipeline
#   EPOCHS=50 BKG_EVENTS=100000 bash develop/scripts/pipeline_relae.sh
#   K=8 bash develop/scripts/pipeline_relae.sh             # unique-8 graph
#   EDGE_PT_SCALE=raw bash develop/scripts/pipeline_relae.sh
#   EVENT_SCORE_AGG=max SKIP_DATA=1 bash develop/scripts/pipeline_relae.sh
#   PYTHON=/path/to/python bash develop/scripts/pipeline_relae.sh
#
# Set SKIP_DATA=1 to reuse existing shards and only (re)train.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '1,18p' "$0"
    cat <<'EOF'

Environment overrides:
  PYTHON=/path/to/python
  BKG_EVENTS=100000
  SIG_EVENTS=40000
  EPOCHS=50
  SKIP_DATA=1
  K=6
  EDGE_PT_SCALE=normalized|raw
  EVENT_SCORE_AGG=sum|mean|max|min|pt_weighted
  DATA_DIR=...
  RUN_DIR=...
EOF
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/develop${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"                       # override: PYTHON=... bash ...
LHCO_H5="develop/dataset/lhco/events_anomalydetection.h5"

# tunables (env-overridable)
BKG_EVENTS="${BKG_EVENTS:-100000}"
SIG_EVENTS="${SIG_EVENTS:-40000}"
EPOCHS="${EPOCHS:-50}"
SKIP_DATA="${SKIP_DATA:-0}"
K="${K:-6}"
EDGE_PT_SCALE="${EDGE_PT_SCALE:-normalized}"      # normalized | raw
EVENT_SCORE_AGG="${EVENT_SCORE_AGG:-sum}"

[[ "$EDGE_PT_SCALE" == "normalized" || "$EDGE_PT_SCALE" == "raw" ]] || {
    echo "EDGE_PT_SCALE must be normalized or raw, got $EDGE_PT_SCALE"; exit 1; }

suffix="unique${K}_logef"
build_extra=()
if [[ "$EDGE_PT_SCALE" == "raw" ]]; then
    suffix="unique${K}_logef_rawpt"
    build_extra+=(--raw_edge_pt)
fi

DATA_DIR="${DATA_DIR:-develop/dataset/processed/lhco_relae_${suffix}}"
if [[ -z "${RUN_DIR:-}" ]]; then
    if [[ "$K" == "6" && "$EDGE_PT_SCALE" == "normalized" ]]; then
        RUN_DIR="develop/runs/relae"
    elif [[ "$K" == "6" && "$EDGE_PT_SCALE" == "raw" ]]; then
        RUN_DIR="develop/runs/relae_rawpt"
    elif [[ "$EDGE_PT_SCALE" == "raw" ]]; then
        RUN_DIR="develop/runs/relae_unique${K}_rawpt"
    else
        RUN_DIR="develop/runs/relae_unique${K}"
    fi
fi

echo "======================================================"
echo " EdgeGraphAE pipeline (best config)"
echo "   python   : $PYTHON"
echo "   data dir : $DATA_DIR"
echo "   run dir  : $RUN_DIR   epochs=$EPOCHS"
echo "   graph    : unique-$K  edge_pt_scale=$EDGE_PT_SCALE"
echo "   score agg: $EVENT_SCORE_AGG"
echo "======================================================"

# Step 0: repair the LHCO HDF5 if it was written by old pandas (idempotent —
# a no-op once the byte-encoded attrs have been decoded to str).
if [[ -f "$LHCO_H5" ]]; then
    "$PYTHON" -m src.data.fix_lhco_h5 "$LHCO_H5"
fi

# Step 1: background+signal subset → subjet-30 → unique-k → log edge features
if [[ "$SKIP_DATA" -eq 0 ]]; then
    "$PYTHON" -m src.data.build_subset \
        --h5_path     "$LHCO_H5" \
        --output_dir  "$DATA_DIR" \
        --bkg_events  "$BKG_EVENTS" \
        --sig_events  "$SIG_EVENTS" \
        --min_jet_pt  1200 --n_subjets 30 \
        --strategy unique --k "$K" --edge_features log "${build_extra[@]}"
else
    echo "SKIP_DATA=1 → reusing $DATA_DIR"
fi

# Step 2: train EdgeGraphAE with the Araz-style schedule
"$PYTHON" -m develop.scripts.train_graph_ae \
    --data_dir "$DATA_DIR" \
    --output   "$RUN_DIR" \
    --model edgeae --node_features pt \
    --scheduler onecycle --lr 3e-3 --weight_decay 0.01 \
    --event_score_agg "$EVENT_SCORE_AGG" \
    --epochs "$EPOCHS" --no_early_stop --eval_interval 5

echo ""
echo "Done. Latest checkpoint under $RUN_DIR/<timestamp>/  (best.pt / last.pt)"
echo "Zero-shot the 4 Kitchen Sink signals with:"
echo "    bash develop/scripts/zeroshot_kitchensink.sh $RUN_DIR/<timestamp>/last.pt"
