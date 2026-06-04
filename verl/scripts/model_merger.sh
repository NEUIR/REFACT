#!/bin/bash
set -x
export PYTHONPATH=.

# Check if PROJECT_ID is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <PROJECT_ID> [STEP]"
    echo "Example: $0 54666"
    echo "Example: $0 54666 1000"
    exit 1
fi

PROJECT_ID=$1
STEP=$2  # Optional second parameter
# HF_MODEL=/user/linbiyuan/models/Qwen3-8B
HF_MODEL=/user/xuxiaoyue/exps/verl/job_49871.iter_100
PROJECT_NAME=823-verl-longcontext
OUT_DIR=/user/xuxiaoyue/exps/verl
CHECKPOINT_DIR=/projects/${PROJECT_NAME}/${PROJECT_ID}/checkpoints
# CHECKPOINT_DIR=/user/xuxiaoyue/ckpt/${PROJECT_ID}/checkpoints/

# Determine the step to use
if [ -n "$STEP" ]; then
    # If STEP is provided as argument, use it
    LAST_STEP=$STEP
    echo "Using step from command line argument: ${LAST_STEP}"
else
    # Otherwise, try to get it from latest_checkpointed_iteration.txt
    LATEST_FILE="${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
    echo "No step provided, reading latest step from: ${LATEST_FILE}"
    
    if [ ! -f "$LATEST_FILE" ]; then
        echo "Error: ${LATEST_FILE} not found and no step provided as argument"
        echo "Usage: $0 <PROJECT_ID> [STEP]"
        exit 1
    fi
    
    LAST_STEP=$(cat ${LATEST_FILE} | tr -d '\n' | tr -d ' ')
    
    if [ -z "$LAST_STEP" ]; then
        echo "Error: Could not read step number from ${LATEST_FILE}"
        exit 1
    fi
    
    echo "Found last step from file: ${LAST_STEP}"
fi

# Check if the directory exists
if [ ! -d "${CHECKPOINT_DIR}/global_step_${LAST_STEP}" ]; then
    echo "Error: Directory ${CHECKPOINT_DIR}/global_step_${LAST_STEP} does not exist"
    exit 1
fi

# Process only the last step
echo "Processing step ${LAST_STEP}..."
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ${CHECKPOINT_DIR}/global_step_${LAST_STEP}/actor \
    --hf_model_path $HF_MODEL \
    --target_dir ${OUT_DIR}/job_${PROJECT_ID}.iter_${LAST_STEP}/

echo "Done! Model saved to: ${OUT_DIR}/job_${PROJECT_ID}.iter_${LAST_STEP}/"