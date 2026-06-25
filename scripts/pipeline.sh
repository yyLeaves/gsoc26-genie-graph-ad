#!/usr/bin/env bash
# Preprocessing pipeline: HDF5 → point-cloud shards → graph shards
#
# Step 1: jet clustering + node features (src.data.preprocess)
# Step 2: add graph edges + optional edge features (src.data.build_graph)
#
# Datasets:
#   lhco         LHCO 2020 R&D (1.1M events, background + W'→XY)
#   kitchensink  the 4 Kitchen Sink signal models (eval only)
#   all          both (default)
#
# Usage:
#   bash develop/scripts/pipeline.sh                          # everything
#   bash develop/scripts/pipeline.sh --dataset lhco           # one dataset
#   bash develop/scripts/pipeline.sh --skip_step1             # Step 2 only
#   bash develop/scripts/pipeline.sh --k 8                    # unique-8 variant
#   bash develop/scripts/pipeline.sh --raw_edge_pt            # raw-pT edge kT
#   bash develop/scripts/pipeline.sh --dry_run                # print paths only

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"          # override with PYTHON=/path/to/python
PROCESSED="$ROOT/develop/dataset/processed"
export PYTHONPATH="$ROOT/develop${PYTHONPATH:+:$PYTHONPATH}"

LHCO_H5="$ROOT/develop/dataset/lhco/events_anomalydetection.h5"
KS_DIR="$ROOT/develop/dataset/kitchensink"
KS_MODELS=(XtoYYprime XtoWRto3W YtoHHto4T ZpToTpTp)

# ── Defaults ──────────────────────────────────────────────────────────────────
DATASET="all"
FEATURES="log_phys"
MIN_JET_PT=1200
MIN_PARTICLES=3
MAX_NODES=""
SHARD_SIZE=8192
K=6
STRATEGY="unique"
EDGE_FEATURES="log"
RAW_EDGE_PT=0
N_SUBJETS="30"
SKIP_STEP1=0
SKIP_STEP2=0
DRY_RUN=0

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            sed -n '1,18p' "$0"
            cat <<'EOF'

Options:
  --dataset lhco|kitchensink|all
  --features raw|normalized|log_phys
  --min_jet_pt FLOAT
  --min_particles INT
  --max_nodes INT
  --shard_size INT
  --strategy knn|laman|unique|fully_connected
  --k INT
  --edge_features none|linear|log
  --raw_edge_pt
  --n_subjets INT
  --skip_step1
  --skip_step2
  --dry_run
EOF
            exit 0
            ;;
        --dataset)        DATASET="$2";        shift 2 ;;
        --features)       FEATURES="$2";       shift 2 ;;
        --min_jet_pt)     MIN_JET_PT="$2";     shift 2 ;;
        --min_particles)  MIN_PARTICLES="$2";  shift 2 ;;
        --max_nodes)      MAX_NODES="$2";      shift 2 ;;
        --shard_size)     SHARD_SIZE="$2";     shift 2 ;;
        --k)              K="$2";              shift 2 ;;
        --strategy)       STRATEGY="$2";       shift 2 ;;
        --edge_features)  EDGE_FEATURES="$2";  shift 2 ;;
        --raw_edge_pt)    RAW_EDGE_PT=1;       shift   ;;
        --n_subjets)      N_SUBJETS="$2";      shift 2 ;;
        --skip_step1)     SKIP_STEP1=1;        shift   ;;
        --skip_step2)     SKIP_STEP2=1;        shift   ;;
        --dry_run)        DRY_RUN=1;           shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$DATASET" in lhco|kitchensink|all) ;; *)
    echo "ERROR: --dataset must be lhco | kitchensink | all"; exit 1 ;;
esac

# ── Config ────────────────────────────────────────────────────────────────────
echo "======================================================"
echo " Preprocessing Pipeline"
echo "======================================================"
echo "  Dataset        : $DATASET"
echo "  Features       : $FEATURES"
echo "  min_jet_pt     : ${MIN_JET_PT} GeV"
echo "  min_particles  : $MIN_PARTICLES"
echo "  strategy       : $STRATEGY   (k=$K)"
echo "  edge_features  : $EDGE_FEATURES"
echo "  edge_pt_scale  : $([[ $RAW_EDGE_PT -eq 1 ]] && echo raw || echo normalized)"
echo "  shard_size     : $SHARD_SIZE"
[[ -n "$MAX_NODES" ]]  && echo "  max_nodes      : $MAX_NODES"
[[ -n "$N_SUBJETS" ]]  && echo "  n_subjets      : $N_SUBJETS"
echo "  skip_step1     : $SKIP_STEP1   skip_step2 : $SKIP_STEP2"
echo "======================================================"

command -v "$PYTHON" >/dev/null || { echo "ERROR: Python not found: $PYTHON"; exit 1; }
cd "$ROOT"

# ── One dataset: HDF5 → point clouds → graphs ─────────────────────────────────
run_one() {
    local name="$1" h5="$2"
    local pc_dir="$PROCESSED/${name}_jets"
    local gtag="${STRATEGY}${K}"
    [[ "$EDGE_FEATURES" != "none" ]] && gtag="${gtag}_${EDGE_FEATURES}ef"
    [[ "$RAW_EDGE_PT" -eq 1 ]] && gtag="${gtag}_rawpt"
    local graph_dir="$PROCESSED/${name}_graphs_${gtag}"

    echo ""
    echo "── $name ─────────────────────────────────────────────"
    echo "  H5     : $h5"
    echo "  Clouds : $pc_dir"
    echo "  Graphs : $graph_dir"
    [[ $DRY_RUN -eq 1 ]] && return 0

    if [[ $SKIP_STEP1 -eq 0 ]]; then
        [[ ! -f "$h5" ]] && echo "ERROR: H5 not found at $h5" && exit 1
        local cmd=(
            "$PYTHON" -m src.data.preprocess
            --h5_path       "$h5"
            --output_dir    "$pc_dir"
            --features      "$FEATURES"
            --min_jet_pt    "$MIN_JET_PT"
            --min_particles "$MIN_PARTICLES"
            --shard_size    "$SHARD_SIZE"
        )
        [[ -n "$MAX_NODES" ]] && cmd+=(--max_nodes "$MAX_NODES")
        [[ -n "$N_SUBJETS" ]] && cmd+=(--n_subjets "$N_SUBJETS")
        "${cmd[@]}"
    else
        [[ ! -f "$pc_dir/metadata.pt" ]] \
            && echo "ERROR: $pc_dir/metadata.pt not found" && exit 1
    fi

    if [[ $SKIP_STEP2 -eq 0 ]]; then
        local graph_cmd=(
            "$PYTHON" -m src.data.build_graph
            --input_dir      "$pc_dir" \
            --output_dir     "$graph_dir" \
            --strategy       "$STRATEGY" \
            --k              "$K" \
            --edge_features  "$EDGE_FEATURES"
        )
        [[ "$RAW_EDGE_PT" -eq 1 ]] && graph_cmd+=(--raw_edge_pt)
        "${graph_cmd[@]}"
    fi
}

# ── Run ───────────────────────────────────────────────────────────────────────
if [[ "$DATASET" == "lhco" || "$DATASET" == "all" ]]; then
    run_one lhco "$LHCO_H5"
fi
if [[ "$DATASET" == "kitchensink" || "$DATASET" == "all" ]]; then
    for model in "${KS_MODELS[@]}"; do
        run_one "$model" "$KS_DIR/$model/events.h5"
    done
fi

echo ""
echo "======================================================"
echo " Pipeline complete → $PROCESSED/"
echo ""
echo " Train with:"
echo "   python -m develop.scripts.train_graph_ae \\"
echo "       --data_dir develop/dataset/processed/lhco_graphs_${STRATEGY}${K}_${EDGE_FEATURES}ef \\"
echo "       --output   develop/runs/relae_${STRATEGY}${K} \\"
echo "       --model edgeae --node_features pt --scheduler onecycle \\"
echo "       --lr 3e-3 --weight_decay 0.01 --epochs 50 --no_early_stop"
echo "======================================================"
