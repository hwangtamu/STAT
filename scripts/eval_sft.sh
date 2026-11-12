#!/bin/bash


NUM_TEST_SAMPLES="${NUM_TEST_SAMPLES:-500}"          # first N rows of the test file; -1 evaluates all of it
NUM_GENERATED_SAMPLES="${NUM_GENERATED_SAMPLES:-1}"  # >1 reports pass@k and average@k
TEMPERATURE="${TEMPERATURE:-0.1}"
SEED="${SEED:-0}"

# Hugging Face model ID (or a local path whose folder name matches the ID's last part):
# "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", or "Qwen/Qwen2.5-3B"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"

TRAIN_DATA_NAME="${TRAIN_DATA_NAME:-STAT-Sel}" # or "STAT-Syn"

# List of test data names: ["math500", "math2", "gsm8k", "math_perturb_simple", "math_perturb_hard", "amc23", "aime"]
TEST_DATA_NAME="${TEST_DATA_NAME:-math500}"
TEST_DATA_PATH="./sft/eval_data/${TEST_DATA_NAME}.jsonl"

# Set to "true" to evaluate the untrained base model instead of the fine-tuned adapter.
EVAL_BASE_MODEL="${EVAL_BASE_MODEL:-false}"


BASE_MODEL_NAME=$(basename "$BASE_MODEL_PATH")

# Instruct models are prompted through their chat template, base models with plain text,
# mirroring the two training scripts.
if [[ "$BASE_MODEL_NAME" == *Instruct* ]]; then
    EVAL_SCRIPT="sft/eval_instruct_model.py"
else
    EVAL_SCRIPT="sft/eval_base_model.py"
fi
echo "EVAL_SCRIPT=${EVAL_SCRIPT}"

ADAPTER_PATH="./output/finetuned_adapters/${BASE_MODEL_NAME}/sft_${TRAIN_DATA_NAME}_${BASE_MODEL_NAME}/final_adapter"
echo "ADAPTER_PATH=${ADAPTER_PATH}"

OUTPUT_DIR="./output/evaluation_results/${BASE_MODEL_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

OUTPUT_FILENAME="eval_results_${TEST_DATA_NAME}_pass${NUM_GENERATED_SAMPLES}_${TEMPERATURE}_sft_${TRAIN_DATA_NAME}_${BASE_MODEL_NAME}.jsonl"
echo "OUTPUT_FILENAME=${OUTPUT_FILENAME}"

BASE_OUTPUT_FILENAME="base_model_results_${TEST_DATA_NAME}_pass${NUM_GENERATED_SAMPLES}_${TEMPERATURE}.jsonl"

EXTRA_ARGS=()
if [[ "$EVAL_BASE_MODEL" == "true" ]]; then
    EXTRA_ARGS+=(--run_base_model)
fi

python "${EVAL_SCRIPT}" \
    --base_model_id "${BASE_MODEL_PATH}" \
    --data_name "${TRAIN_DATA_NAME}" \
    --adapter_path "${ADAPTER_PATH}" \
    --test_data_path "${TEST_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_filename "${OUTPUT_FILENAME}" \
    --base_output_filename "${BASE_OUTPUT_FILENAME}" \
    --num_test_samples "${NUM_TEST_SAMPLES}" \
    --num_generated_responses "${NUM_GENERATED_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --overwrite "${EXTRA_ARGS[@]}"
