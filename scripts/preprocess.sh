#!/usr/bin/env bash
# raw HDF5 → selected-jet constituent point-cloud shards.
#
# This script produces the graph-agnostic selected-jet dataset.  Representation
# choices and event-level splits are downstream steps.
#
# Defaults:
#   anti-kT R=1 is fixed in src.data.extractor
#   leading_pt jet selection
#   require only leading jet pT > 1.2 TeV
#   keep events with two usable selected jets
#   store selected jet constituents for downstream representations

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
H5_PATH="${H5_PATH:-dataset/lhco/events_anomalydetection.h5}"
LABELS_PATH="${LABELS_PATH:-}"
OUT_DIR="${OUT_DIR:-dataset/processed/lhco_canonical_leadingpt_jets}"
FEATURES="${FEATURES:-log_phys}"
MIN_JET_PT="${MIN_JET_PT:-1200}"
MIN_PARTICLES="${MIN_PARTICLES:-3}"
JET_SELECTION="${JET_SELECTION:-leading_pt}"
REQUIRE_TWO_JETS="${REQUIRE_TWO_JETS:-1}"
SHARD_SIZE="${SHARD_SIZE:-8192}"
FORCE="${FORCE:-0}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '1,16p' "$0"
    cat <<'EOF'

Environment overrides:
  PYTHON=/path/to/python
  H5_PATH=dataset/lhco/events_anomalydetection.h5
  LABELS_PATH=       # optional masterkey/truth file for unlabeled BB HDF5
  OUT_DIR=dataset/processed/lhco_canonical_leadingpt_jets
  FEATURES=log_phys
  MIN_JET_PT=1200
  MIN_PARTICLES=3
  JET_SELECTION=leading_pt
  REQUIRE_TWO_JETS=1
  SHARD_SIZE=8192
  FORCE=1       remove OUT_DIR before rebuilding
EOF
    exit 0
fi

[[ -f "$H5_PATH" ]] || { echo "ERROR: H5 not found: $H5_PATH"; exit 1; }
if [[ -n "$LABELS_PATH" && ! -f "$LABELS_PATH" ]]; then
    echo "ERROR: LABELS_PATH not found: $LABELS_PATH"
    exit 1
fi
command -v "$PYTHON" >/dev/null || {
    echo "ERROR: Python not found: $PYTHON"; exit 1;
}
[[ "$REQUIRE_TWO_JETS" == "0" || "$REQUIRE_TWO_JETS" == "1" ]] || {
    echo "ERROR: REQUIRE_TWO_JETS must be 0 or 1, got $REQUIRE_TWO_JETS";
    exit 1;
}
if [[ -n "${N_SUBJETS:-}" ]]; then
    echo "ERROR: N_SUBJETS does not belong in preprocessing."
    echo "Run scripts/build_graph.sh for subjet or other graph representations."
    exit 1
fi

if [[ -e "$OUT_DIR/metadata.pt" && "$FORCE" != "1" ]]; then
    echo "Reusing existing selected-jet preprocessing: $OUT_DIR"
    echo "Set FORCE=1 to rebuild."
    exit 0
fi
if [[ -d "$OUT_DIR" && ! -e "$OUT_DIR/metadata.pt" && "$FORCE" != "1" ]]; then
    echo "ERROR: $OUT_DIR exists but has no metadata.pt; likely partial/in-progress."
    echo "Set FORCE=1 to remove it and rebuild."
    exit 1
fi

if [[ "$FORCE" == "1" ]]; then
    rm -rf "$OUT_DIR"
fi

cmd=(
    "$PYTHON" -u -m src.data.preprocess
    --h5_path "$H5_PATH"
    --output_dir "$OUT_DIR"
    --features "$FEATURES"
    --min_jet_pt "$MIN_JET_PT"
    --min_particles "$MIN_PARTICLES"
    --jet_selection "$JET_SELECTION"
    --shard_size "$SHARD_SIZE"
)
if [[ -n "$LABELS_PATH" ]]; then
    cmd+=(--labels_path "$LABELS_PATH")
fi
if [[ "$REQUIRE_TWO_JETS" == "1" ]]; then
    cmd+=(--require_two_jets)
else
    cmd+=(--allow_single_jet)
fi

echo "Selected-jet preprocessing"
echo "  input : $H5_PATH"
if [[ -n "$LABELS_PATH" ]]; then
    echo "  labels: $LABELS_PATH"
fi
echo "  output: $OUT_DIR"
echo "  jet_selection=$JET_SELECTION  min_jet_pt=$MIN_JET_PT"
echo "  require_two_jets=$REQUIRE_TWO_JETS"
"${cmd[@]}"
