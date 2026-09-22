#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/user/lyeyang/miniconda3/envs/genie/bin/python}"
TRAIN_ROOT="${TRAIN_ROOT:-runs/gcn_edge3_full_training_50ep_no_bn}"
OUTPUT="${OUTPUT:-runs/gcn_edge3_full_training_50ep_no_bn_ks_eval}"
BKG="dataset/processed/lhco_canonical_leadingpt_sj30_unique6_logef_trainpack_seed42"
SPLIT="dataset/processed/splits/lhco_canonical_leadingpt_sj30_train80000b_val20000b_test340000b_20000s_monsig20000_seed42.npz"

signals=(
  "XtoWRto3W=dataset/processed/ks_XtoWRto3W_leadingpt_sj30_unique6_logef"
  "XtoYYprime=dataset/processed/ks_XtoYYprime_leadingpt_sj30_unique6_logef"
  "ZpToTpTp=dataset/processed/ks_ZpToTpTp_leadingpt_sj30_unique6_logef"
  "YtoHHto4T=dataset/processed/ks_YtoHHto4T_leadingpt_sj30_unique6_logef"
)

mkdir -p "${OUTPUT}/logs"
"${PYTHON}" -c \
    'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; refusing CPU fallback"'

run_eval() {
    local name="$1"
    local gpu="$2"
    local train_name="$3"
    local checkpoint_name="$4"
    if [[ -s "${OUTPUT}/${name}/summary.json" ]]; then
        echo "SKIP: completed ${name}"
        return 0
    fi
    local run_dir checkpoint
    run_dir="$(find "${TRAIN_ROOT}/${train_name}" -mindepth 1 -maxdepth 1 \
        -type d | sort | tail -1)"
    checkpoint="${run_dir}/${checkpoint_name}.pt"
    if [[ ! -s "${checkpoint}" ]]; then
        echo "ERROR: missing checkpoint ${checkpoint}" >&2
        return 1
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

pids=()
run_eval pure_val_best 0 pure_background_seed42 best & pids+=("$!")
run_eval pure_monitor_best_oracle 1 pure_background_seed42 monitor_best & pids+=("$!")
run_eval contamination_3pct_val_best 2 contamination_3pct_seed42 best & pids+=("$!")
run_eval contamination_3pct_monitor_best_oracle 3 contamination_3pct_seed42 monitor_best & pids+=("$!")
for pid in "${pids[@]}"; do
    wait "${pid}"
done
echo "GCN-edge Kitchen Sink evaluations completed: ${OUTPUT}"
