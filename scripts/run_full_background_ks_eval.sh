#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/user/lyeyang/miniconda3/envs/genie/bin/python}"
TRAIN_ROOT="runs/full_background_architecture"
OUTPUT="runs/full_background_architecture_ks_eval"
BKG="dataset/processed/lhco_canonical_leadingpt_sj30_unique6_logef"
SPLIT="dataset/processed/splits/lhco_canonical_leadingpt_sj30_trainallb_val20000b_test340000b_20000s_monsig20000_preserveeval_seed42.npz"

signals=(
  "XtoWRto3W=dataset/processed/ks_XtoWRto3W_leadingpt_sj30_unique6_logef"
  "XtoYYprime=dataset/processed/ks_XtoYYprime_leadingpt_sj30_unique6_logef"
  "ZpToTpTp=dataset/processed/ks_ZpToTpTp_leadingpt_sj30_unique6_logef"
  "YtoHHto4T=dataset/processed/ks_YtoHHto4T_leadingpt_sj30_unique6_logef"
)

mkdir -p "${OUTPUT}/logs"

if ! "${PYTHON}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "ERROR: CUDA is unavailable; refusing to run the full KS evaluation on CPU." >&2
    exit 1
fi

run_eval() {
    local name="$1"
    local gpu="$2"
    if [[ -s "${OUTPUT}/${name}/summary.json" ]]; then
        echo "SKIP: completed ${name}"
        return 0
    fi
    local checkpoint
    checkpoint="$(find "${TRAIN_ROOT}/${name}" -mindepth 2 -maxdepth 2 \
        -name last.pt -print | sort | tail -1)"
    if [[ -z "${checkpoint}" || ! -s "$(dirname "${checkpoint}")/metrics_last.json" ]]; then
        echo "SKIP: completed checkpoint not ready for ${name}"
        return 0
    fi
    local args=(
        --checkpoint "${checkpoint}"
        --bkg_dir "${BKG}"
        --split_manifest "${SPLIT}"
        --output_dir "${OUTPUT}/${name}"
        --n_signal 20000 --bkg_to_signal 17
        --batch_size 2048 --event_score_agg sum
    )
    for signal in "${signals[@]}"; do
        args+=(--signal "${signal}")
    done
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m scripts.eval_ks_ratio \
        "${args[@]}" >"${OUTPUT}/logs/${name}.log" 2>&1
}

names=(
  reference_1ep_seed123 reference_3ep_seed123
  gcn_edge3_1ep_seed123 gcn_edge3_3ep_seed123
  gcn_edge3_1ep_seed42 gcn_edge3_3ep_seed42
)
pids=()
for gpu in "${!names[@]}"; do
    run_eval "${names[$gpu]}" "${gpu}" & pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "${pid}"
done
echo "Full-background Kitchen Sink evaluations completed: ${OUTPUT}"
