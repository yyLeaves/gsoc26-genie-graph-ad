#!/usr/bin/env bash
set -euo pipefail

# Reference Node+Edge AE trained on pure LHCO background outside
# 3600 <= mJJ < 4000 GeV.  The manifest keeps the standard 80k/20k split and
# the exact same monitor/final-test events as the all-mass reference run.

PYTHON="${PYTHON:-/home/user/lyeyang/miniconda3/envs/genie/bin/python}"
GPU="${GPU:-0}"
DATA_DIR="${DATA_DIR:-dataset/processed/lhco_leadingpt_sj30_unique6_mjj_exclude3600_4000_trainpack_seed42}"
SPLIT="${SPLIT:-dataset/processed/splits/lhco_leadingpt_sj30_train80000b_val20000b_test340000b_20000s_monsig20000_mjj_exclude3600_4000_seed42.npz}"
OUTPUT="${OUTPUT:-runs/mjj_window_exclude3600_4000/reference_pure_seed123}"

test -s "${DATA_DIR}/metadata.pt"
test -s "${SPLIT}"
mkdir -p "${OUTPUT}/logs"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -c \
    'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m scripts.train_graph_ae \
    --data_dir "${DATA_DIR}" \
    --split_manifest "${SPLIT}" \
    --output "${OUTPUT}" \
    --model edge_graph --backbone edgeconv --node_features pt \
    --aggr mean --edge_weight 1 \
    --epochs 50 --batch_size 512 --cache_shards 32 \
    --hidden_dim 64 --latent_dim 2 --no_bn \
    --lr 3e-3 --weight_decay 0.01 --scheduler onecycle \
    --eval_interval 5 --save_monitor_best --no_early_stop \
    --event_score_agg sum --seed 123 \
    2>&1 | tee "${OUTPUT}/logs/train.log"
