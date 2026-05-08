#!/bin/bash

domain='Rest-TA'
DATE=$(date +"%Y%m%d_%H%M%S")
IS_PREPROCESS=1

echo "=== Run ${domain^^} ==="

DATA_PATH=../Data/${domain}/
LOG_DIR=../logs/${domain}/
SAVE_DIR=../results/${domain}/
ADG_DIR=../Data/data_info/${domain}/

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

# Train
echo "=== Train ${domain^^} ==="
CUDA_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
PYTHONDONTWRITEBYTECODE=1 \
python -u main.py \
    --data_path "${DATA_PATH}" \
    --ADG_path "${ADG_DIR}" \
    --domain "${domain}" \
    --is_preprocess "${IS_PREPROCESS}" \
    --is_train 1 \
    --save_path "${SAVE_DIR}" \
    2>&1 | tee "${LOG_DIR}/log_${DATE}.log"

# Test
echo "=== Test ${domain^^} ==="
CUDA_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
PYTHONDONTWRITEBYTECODE=1 \
python -u main.py \
    --data_path "${DATA_PATH}" \
    --ADG_path "${ADG_DIR}" \
    --domain "${domain}" \
    --is_preprocess 0 \
    --is_train 0 \
    --save_path "${SAVE_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/log_${DATE}.log"

echo -e "\n=== Finished ${domain^^} ===\n"