#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA="dataset/processed/lhco_canonical_leadingpt_sj30_unique6_logef_trainpack_seed42"
PURE_SPLIT="dataset/processed/splits/lhco_canonical_leadingpt_sj30_train80000b_val20000b_test340000b_20000s_monsig20000_seed42.npz"
CONTAM_SPLIT="dataset/processed/splits/lhco_canonical_leadingpt_sj30_train80000b_val20000b_test340000b_20000s_sbr0p03_sameeval_seed42.npz"
BB1="dataset/processed/bb1_canonical_leadingpt_sj30_unique6_logef"
RUN_ROOT="runs/gladc/formal_seed123"
LOG_ROOT="${RUN_ROOT}/logs"

signals=(
  "XtoWRto3W=dataset/processed/ks_XtoWRto3W_leadingpt_sj30_unique6_logef"
  "XtoYYprime=dataset/processed/ks_XtoYYprime_leadingpt_sj30_unique6_logef"
  "ZpToTpTp=dataset/processed/ks_ZpToTpTp_leadingpt_sj30_unique6_logef"
  "YtoHHto4T=dataset/processed/ks_YtoHHto4T_leadingpt_sj30_unique6_logef"
)

mkdir -p "${LOG_ROOT}"

train_one() {
  local gpu="$1"
  local tag="$2"
  local split="$3"
  local cycle_weight="$4"
  local out="${RUN_ROOT}/${tag}"
  local log="${LOG_ROOT}/${tag}.log"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m scripts.train_graph_ae \
    --data_dir "${DATA}" \
    --split_manifest "${split}" \
    --output "${out}" \
    --model edge_graph \
    --backbone edgeconv \
    --epochs 50 \
    --batch_size 512 \
    --hidden_dim 64 \
    --latent_dim 2 \
    --no_bn \
    --scheduler onecycle \
    --edge_weight 1 \
    --cycle_weight "${cycle_weight}" \
    --cycle_component total \
    --contrast_weight 1 \
    --perturb_scale 1 \
    --contrast_temperature 0.2 \
    --contrast_projection_dim 64 \
    --anomaly_score reconstruction \
    --eval_interval 5 \
    --no_early_stop \
    --event_score_agg sum \
    --cache_shards 32 \
    --seed 123 \
    >"${log}" 2>&1
}

eval_one() {
  local gpu="$1"
  local tag="$2"
  local split="$3"
  local out="${RUN_ROOT}/${tag}"
  local log="${LOG_ROOT}/${tag}.log"
  local checkpoint
  checkpoint="$(find "${out}" -mindepth 2 -maxdepth 2 \
    -name last.pt -print | sort | tail -n 1)"
  if [[ -z "${checkpoint}" ]]; then
    echo "No last.pt produced for ${tag}" >&2
    return 1
  fi

  local eval_args=(
    --checkpoint "${checkpoint}"
    --bkg_dir "${DATA}"
    --split_manifest "${split}"
    --bb1_dir "${BB1}"
    --output_dir "${out}/full_suite"
    --n_signal 20000
    --batch_size 2048
    --cache_shards 32
  )
  for signal in "${signals[@]}"; do
    eval_args+=(--signal "${signal}")
  done
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    -m scripts.eval_cycle_suite "${eval_args[@]}" >>"${log}" 2>&1
}

train_one 0 contrast_only_sbr0 "${PURE_SPLIT}" 0 & p0=$!
train_one 1 contrast_only_sbr3 "${CONTAM_SPLIT}" 0 & p1=$!
train_one 2 full_sbr0 "${PURE_SPLIT}" 0.05 & p2=$!
train_one 3 full_sbr3 "${CONTAM_SPLIT}" 0.05 & p3=$!
failed=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "One or more jobs failed; stopping this queue." >&2
  exit 1
fi

eval_one 0 contrast_only_sbr0 "${PURE_SPLIT}" & e0=$!
eval_one 1 contrast_only_sbr3 "${CONTAM_SPLIT}" & e1=$!
eval_one 2 full_sbr0 "${PURE_SPLIT}" & e2=$!
eval_one 3 full_sbr3 "${CONTAM_SPLIT}" & e3=$!
failed=0
for pid in "${e0}" "${e1}" "${e2}" "${e3}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "One or more jobs failed; stopping this queue." >&2
  exit 1
fi

echo "Full GLADC training and six-dataset evaluation completed."
