#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
CANONICAL_DIR="${CANONICAL_DIR:-dataset/processed/lhco_canonical_leadingpt_jets}"
INPUT_DIR_WAS_SET=0
if [[ -n "${INPUT_DIR+x}" ]]; then
    INPUT_DIR_WAS_SET=1
fi
INPUT_DIR="${INPUT_DIR:-$CANONICAL_DIR}"
N_SUBJETS="${N_SUBJETS:-30}"
STRATEGY="${STRATEGY:-unique}"
K="${K:-6}"
EDGE_FEATURES="${EDGE_FEATURES:-log}"
EDGE_PT_SCALE="${EDGE_PT_SCALE:-normalized}"   # normalized | raw
FORCE="${FORCE:-0}"

derive_prefix() {
    local path="${1%/}"
    if [[ "$path" == *_jets ]]; then
        echo "${path%_jets}"
    else
        echo "$path"
    fi
}

if [[ -z "${OUT_DIR:-}" ]]; then
    suffix="${STRATEGY}${K}"
    [[ "$EDGE_FEATURES" != "none" ]] && suffix="${suffix}_${EDGE_FEATURES}ef"
    [[ "$EDGE_PT_SCALE" == "raw" ]] && suffix="${suffix}_rawpt"
    repr_tag=""
    [[ -n "$N_SUBJETS" ]] && repr_tag="_sj${N_SUBJETS}"
    if [[ "$INPUT_DIR_WAS_SET" == "1" ]]; then
        prefix="$(derive_prefix "$INPUT_DIR")"
    else
        prefix="$(derive_prefix "$CANONICAL_DIR")"
    fi
    OUT_DIR="${prefix}${repr_tag}_${suffix}"
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '1,12p' "$0"
    cat <<'EOF'

Environment overrides:
  PYTHON=/path/to/python
  CANONICAL_DIR=dataset/processed/lhco_canonical_leadingpt_jets
  INPUT_DIR=...        # optional; overrides CANONICAL_DIR/N_SUBJETS derivation
  SUBJET_DIR=...       # optional; output for N_SUBJETS representation
  OUT_DIR=dataset/processed/lhco_canonical_leadingpt_sj30_unique6_logef
  N_SUBJETS=30         # set empty to build graph directly on constituents
  STRATEGY=unique      # knn|laman|unique|fully_connected
  K=6
  EDGE_FEATURES=log    # none|linear|log
  EDGE_PT_SCALE=normalized|raw
  FORCE=1              # remove OUT_DIR before rebuilding
EOF
    exit 0
fi

command -v "$PYTHON" >/dev/null || {
    echo "ERROR: Python not found: $PYTHON"; exit 1;
}
[[ "$EDGE_PT_SCALE" == "normalized" || "$EDGE_PT_SCALE" == "raw" ]] || {
    echo "ERROR: EDGE_PT_SCALE must be normalized or raw, got $EDGE_PT_SCALE";
    exit 1;
}

if [[ -e "$OUT_DIR/metadata.pt" && "$FORCE" != "1" ]]; then
    echo "Reusing existing graph dataset: $OUT_DIR"
    echo "Set FORCE=1 to rebuild."
    exit 0
fi
if [[ -d "$OUT_DIR" && ! -e "$OUT_DIR/metadata.pt" && "$FORCE" != "1" ]]; then
    echo "ERROR: $OUT_DIR exists but has no metadata.pt; likely partial/in-progress."
    echo "Set FORCE=1 to remove it and rebuild."
    exit 1
fi

if [[ -n "$N_SUBJETS" && "$INPUT_DIR_WAS_SET" == "0" ]]; then
    if [[ -z "${SUBJET_DIR:-}" ]]; then
        SUBJET_DIR="$(derive_prefix "$CANONICAL_DIR")_sj${N_SUBJETS}_jets"
    fi
    if [[ -e "$SUBJET_DIR/metadata.pt" && "$FORCE" != "1" ]]; then
        echo "Reusing existing subjet representation: $SUBJET_DIR"
    else
        if [[ -d "$SUBJET_DIR" && ! -e "$SUBJET_DIR/metadata.pt" && "$FORCE" != "1" ]]; then
            echo "ERROR: $SUBJET_DIR exists but has no metadata.pt; likely partial/in-progress."
            echo "Set FORCE=1 to remove it and rebuild."
            exit 1
        fi
        if [[ "$FORCE" == "1" ]]; then
            rm -rf "$SUBJET_DIR"
        fi
        [[ -f "$CANONICAL_DIR/metadata.pt" ]] || {
            echo "ERROR: missing canonical metadata: $CANONICAL_DIR/metadata.pt"; exit 1;
        }
        "$PYTHON" -u -m src.data.build_subjets \
            --input_dir "$CANONICAL_DIR" \
            --output_dir "$SUBJET_DIR" \
            --n_subjets "$N_SUBJETS"
    fi
    INPUT_DIR="$SUBJET_DIR"
fi

[[ -f "$INPUT_DIR/metadata.pt" ]] || {
    echo "ERROR: missing input metadata: $INPUT_DIR/metadata.pt"; exit 1;
}

if [[ "$FORCE" == "1" ]]; then
    rm -rf "$OUT_DIR"
fi

cmd=(
    "$PYTHON" -u -m src.data.build_graph
    --input_dir "$INPUT_DIR"
    --output_dir "$OUT_DIR"
    --strategy "$STRATEGY"
    --k "$K"
    --edge_features "$EDGE_FEATURES"
    --edge_pt_scale "$EDGE_PT_SCALE"
)

echo "Build graph dataset"
echo "  input : $INPUT_DIR"
echo "  output: $OUT_DIR"
echo "  graph : $STRATEGY  k=$K  edge_features=$EDGE_FEATURES  edge_pt_scale=$EDGE_PT_SCALE  n_subjets=${N_SUBJETS:-none}"
"${cmd[@]}"
