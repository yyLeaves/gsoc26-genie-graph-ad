#!/usr/bin/env bash
# Zero-shot Kitchen Sink inference.
#
# Takes an EdgeGraphAE checkpoint trained ONLY on LHCO QCD background and scores
# the 4 unseen Kitchen Sink BSM signals against a held-out background — no
# retraining. Each signal is processed through the SAME pipeline the model was
# trained on (pT>1.2 TeV → 30 subjets → unique-k → log edges), then infer.py
# computes honest AUC / max-SIC / ε_S for each.
#
# Usage:
#   bash develop/scripts/zeroshot_kitchensink.sh <checkpoint.pt> [bkg_graph_dir]
#
#   <checkpoint.pt>  e.g. develop/runs/relae/<timestamp>/last.pt
#   [bkg_graph_dir]  held-out background graph shards (default follows
#                    K/EDGE_PT_SCALE, e.g. lhco_relae_unique6_logef)
#
# Env:
#   EDGE_PT_SCALE=normalized|raw     kT pT scale for newly built KS graphs
#   K=6                             unique-k graph parameter
#   EVENT_SCORE_AGG=sum|mean|max|min|pt_weighted
#                                    event score aggregation for infer.py
#   FORCE_DATA=1                    rebuild KS graphs even if compatible

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '1,20p' "$0"
    cat <<'EOF'

Usage:
  bash develop/scripts/zeroshot_kitchensink.sh <checkpoint.pt> [bkg_graph_dir]

Environment overrides:
  PYTHON=/path/to/python
  K=6
  EDGE_PT_SCALE=normalized|raw
  EVENT_SCORE_AGG=sum|mean|max|min|pt_weighted
  FORCE_DATA=1
EOF
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/develop${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
CKPT="${1:?usage: zeroshot_kitchensink.sh <checkpoint.pt> [bkg_graph_dir]}"
KS_DIR="develop/dataset/kitchensink"
PROC="develop/dataset/processed"
MODELS=(XtoWRto3W XtoYYprime ZpToTpTp YtoHHto4T)
K="${K:-6}"
EDGE_PT_SCALE="${EDGE_PT_SCALE:-normalized}"
EVENT_SCORE_AGG="${EVENT_SCORE_AGG:-sum}"
FORCE_DATA="${FORCE_DATA:-0}"

[[ "$EDGE_PT_SCALE" == "normalized" || "$EDGE_PT_SCALE" == "raw" ]] || {
    echo "EDGE_PT_SCALE must be normalized or raw, got $EDGE_PT_SCALE"; exit 1; }

suffix="unique${K}_logef"
build_extra=()
if [[ "$EDGE_PT_SCALE" == "raw" ]]; then
    suffix="unique${K}_logef_rawpt"
    build_extra+=(--raw_edge_pt)
fi
BKG_DIR="${2:-develop/dataset/processed/lhco_relae_${suffix}}"

[[ -f "$CKPT" ]]    || { echo "checkpoint not found: $CKPT"; exit 1; }
[[ -d "$BKG_DIR" ]] || { echo "background dir not found: $BKG_DIR"; exit 1; }

echo "======================================================"
echo " Kitchen Sink zero-shot   checkpoint=$CKPT"
echo " background reference=$BKG_DIR"
echo " graph=unique-$K   edge pt scale=$EDGE_PT_SCALE"
echo " event score agg=$EVENT_SCORE_AGG"
echo "======================================================"

for m in "${MODELS[@]}"; do
    h5="$KS_DIR/$m/events.h5"
    jet_dir="$PROC/ks_${m}_jets_event"
    graph_dir="$PROC/ks_${m}_${suffix}_event"
    [[ -f "$h5" ]] || { echo "skip $m: $h5 missing"; continue; }

    needs_build=0
    if [[ "$FORCE_DATA" -eq 1 || ! -f "$graph_dir/metadata.pt" ]]; then
        needs_build=1
    else
        if ! "$PYTHON" - "$graph_dir" "$EDGE_PT_SCALE" "$K" <<'PY'
import sys, torch
from pathlib import Path
meta = torch.load(Path(sys.argv[1]) / "metadata.pt", weights_only=False)
expected_scale = sys.argv[2]
expected_k = int(sys.argv[3])
ok = (
    "event_ids" in meta
    and meta.get("edge_pt_scale", "normalized") == expected_scale
    and meta.get("strategy") == "unique"
    and int(meta.get("k", -1)) == expected_k
    and meta.get("edge_feats") == "log"
)
sys.exit(0 if ok else 1)
PY
        then
            needs_build=1
        fi
    fi

    # build graph shards for this signal if missing/incompatible (same recipe)
    if [[ "$needs_build" -eq 1 ]]; then
        echo "── building $m graphs ──"
        needs_preprocess=0
        if [[ "$FORCE_DATA" -eq 1 || ! -f "$jet_dir/metadata.pt" ]]; then
            needs_preprocess=1
        elif ! "$PYTHON" - "$jet_dir" <<'PY'
import sys, torch
from pathlib import Path
meta = torch.load(Path(sys.argv[1]) / "metadata.pt", weights_only=False)
sys.exit(0 if "event_ids" in meta else 1)
PY
        then
            needs_preprocess=1
        fi
        if [[ "$needs_preprocess" -eq 1 ]]; then
            "$PYTHON" -m src.data.preprocess \
                --h5_path "$h5" --output_dir "$jet_dir" \
                --min_jet_pt 1200 --n_subjets 30
        else
            echo "   reusing point-cloud shards: $jet_dir"
        fi
        "$PYTHON" -m src.data.build_graph \
            --input_dir "$jet_dir" --output_dir "$graph_dir" \
            --strategy unique --k "$K" --edge_features log "${build_extra[@]}"
    else
        echo "── reusing $m graphs: $graph_dir ──"
    fi

    echo "── $m zero-shot ──"
    "$PYTHON" -m develop.scripts.infer \
        --checkpoint "$CKPT" --data_dir "$graph_dir" --bkg_dir "$BKG_DIR" \
        --event_score_agg "$EVENT_SCORE_AGG" \
        --output_dir "develop/runs/zeroshot/${m}_${suffix}_${EVENT_SCORE_AGG}"
done

echo ""
echo "Per-model metrics under develop/runs/zeroshot/<model>/metrics.json"
