#!/bin/bash

# Hugging Face model ID (or a local path whose folder name matches the ID's last part).
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B}"
MODEL_NAME=$(basename "$MODEL_PATH")

DATA_NAME="${DATA_NAME:-STAT-Sel}" # or "STAT-Syn"

# The paper picks the learning rate per method from {5e-4, 2e-4, 1e-4, 8e-5, 2e-5}
# and reports a weight decay of 0.1; the values below are what this repo shipped with.
LR="${LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

OUTPUT_DIR="./output/finetuned_adapters/${MODEL_NAME}/sft_${DATA_NAME}_${MODEL_NAME}"


echo "Training $MODEL_NAME on $DATA_NAME data"

python sft/finetune_base_model.py \
    --model_id "${MODEL_PATH}" \
    --dataset_path "./STAT_data/${DATA_NAME}/${DATA_NAME}_${MODEL_NAME}_9.5k.json" \
    --output_dir "${OUTPUT_DIR}" \
    --learning_rate "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --num_train_epochs 3
