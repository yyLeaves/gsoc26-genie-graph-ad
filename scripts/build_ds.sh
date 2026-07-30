#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
CANONICAL_JETS="${CANONICAL_JETS:-dataset/processed/lhco_canonical_leadingpt_jets}"
GRAPH_DIR="${GRAPH_DIR:-dataset/processed/lhco_canonical_leadingpt_sj30_unique6_logef}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_GRAPH="${RUN_GRAPH:-1}"
RUN_SPLIT="${RUN_SPLIT:-1}"

SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_BKG_EVENTS="${TRAIN_BKG_EVENTS:-80000}"
TRAIN_SIG_EVENTS="${TRAIN_SIG_EVENTS:-}"
TRAIN_S_OVER_B="${TRAIN_S_OVER_B:-0.0}"
VAL_BKG_EVENTS="${VAL_BKG_EVENTS:-20000}"
MONITOR_SIG_EVENTS="${MONITOR_SIG_EVENTS:-0}"
TEST_BKG_EVENTS="${TEST_BKG_EVENTS:-340000}"
TEST_SIG_EVENTS="${TEST_SIG_EVENTS:-20000}"

if [[ -z "${SPLIT_MANIFEST:-}" ]]; then
    contam_tag=""
    if [[ -n "$TRAIN_SIG_EVENTS" ]]; then
        contam_tag="_trainsig${TRAIN_SIG_EVENTS}"
    elif [[ "$TRAIN_S_OVER_B" != "0" && "$TRAIN_S_OVER_B" != "0.0" && "$TRAIN_S_OVER_B" != "0.00" ]]; then
        contam_tag="_sbr${TRAIN_S_OVER_B//./p}"
    fi
    monitor_tag=""
    if [[ "$MONITOR_SIG_EVENTS" != "0" ]]; then
        monitor_tag="_monsig${MONITOR_SIG_EVENTS}"
    fi
    SPLIT_MANIFEST="dataset/processed/splits/lhco_leadingpt_train${TRAIN_BKG_EVENTS}b_val${VAL_BKG_EVENTS}b_test${TEST_BKG_EVENTS}b_${TEST_SIG_EVENTS}s${contam_tag}${monitor_tag}_seed${SPLIT_SEED}.npz"
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '1,15p' "$0"
    cat <<'EOF'

Environment overrides:
  PYTHON=/path/to/python
  CANONICAL_JETS=dataset/processed/lhco_canonical_leadingpt_jets
  GRAPH_DIR=dataset/processed/lhco_canonical_leadingpt_sj30_unique6_logef
  SPLIT_MANIFEST=...   # optional; defaults include split sizes/contamination/seed

  RUN_PREPROCESS=1|0
  RUN_GRAPH=1|0
  RUN_SPLIT=1|0

  # passed through to preprocess.sh / build_graph.sh:
  FORCE=1
  STRATEGY=unique
  K=6
  EDGE_FEATURES=log
  EDGE_PT_SCALE=normalized|raw

  # split sizes:
  SPLIT_SEED=42
  TRAIN_BKG_EVENTS=80000
  TRAIN_S_OVER_B=0.0       # optional train signal contamination ratio
  TRAIN_SIG_EVENTS=        # optional explicit override for train signal count
  VAL_BKG_EVENTS=20000
  MONITOR_SIG_EVENTS=0     # optional disjoint signal for epoch monitoring
  TEST_BKG_EVENTS=340000
  TEST_SIG_EVENTS=20000
EOF
    exit 0
fi

command -v "$PYTHON" >/dev/null || {
    echo "ERROR: Python not found: $PYTHON"; exit 1;
}

echo "Build selected-jet dataset stack"
echo "  selected jets : $CANONICAL_JETS"
echo "  graph dataset  : $GRAPH_DIR"
echo "  split manifest : $SPLIT_MANIFEST"
echo "  steps          : preprocess=$RUN_PREPROCESS graph=$RUN_GRAPH split=$RUN_SPLIT"

if [[ "$RUN_PREPROCESS" == "1" ]]; then
    OUT_DIR="$CANONICAL_JETS" "$SCRIPT_DIR/preprocess.sh"
fi

if [[ "$RUN_GRAPH" == "1" ]]; then
    CANONICAL_DIR="$CANONICAL_JETS" OUT_DIR="$GRAPH_DIR" "$SCRIPT_DIR/build_graph.sh"
fi

if [[ "$RUN_SPLIT" == "1" ]]; then
    if [[ ! -f "$CANONICAL_JETS/metadata.pt" ]]; then
        echo "ERROR: missing selected-jet metadata: $CANONICAL_JETS/metadata.pt"
        exit 1
    fi
    if [[ -n "$TRAIN_SIG_EVENTS" ]]; then
        "$PYTHON" -u scripts/make_event_split.py \
            --data_dir "$CANONICAL_JETS" \
            --output "$SPLIT_MANIFEST" \
            --seed "$SPLIT_SEED" \
            --train_bkg_events "$TRAIN_BKG_EVENTS" \
            --train_sig_events "$TRAIN_SIG_EVENTS" \
            --val_bkg_events "$VAL_BKG_EVENTS" \
            --monitor_sig_events "$MONITOR_SIG_EVENTS" \
            --test_bkg_events "$TEST_BKG_EVENTS" \
            --test_sig_events "$TEST_SIG_EVENTS"
    else
        "$PYTHON" -u scripts/make_event_split.py \
            --data_dir "$CANONICAL_JETS" \
            --output "$SPLIT_MANIFEST" \
            --seed "$SPLIT_SEED" \
            --train_bkg_events "$TRAIN_BKG_EVENTS" \
            --train_s_over_b "$TRAIN_S_OVER_B" \
            --val_bkg_events "$VAL_BKG_EVENTS" \
            --monitor_sig_events "$MONITOR_SIG_EVENTS" \
            --test_bkg_events "$TEST_BKG_EVENTS" \
            --test_sig_events "$TEST_SIG_EVENTS"
    fi
fi

echo ""
echo "Dataset stack ready."
echo "Train with:"
echo "  PYTHONPATH=. $PYTHON -m scripts.train_graph_ae \\"
echo "    --data_dir $GRAPH_DIR \\"
echo "    --split_manifest $SPLIT_MANIFEST \\"
echo "    --output runs/leadingpt_ksfixed \\"
echo "    --model edge_graph --node_features pt --scheduler onecycle \\"
echo "    --lr 3e-3 --weight_decay 0.01 --event_score_agg sum \\"
echo "    --epochs 50 --no_early_stop --eval_interval 0"
