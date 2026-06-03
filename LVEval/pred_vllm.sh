set -euo pipefail

# 脚本所在目录（无论从哪里 bash 都能定位到 LVEval-main）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 2 ]; then
  echo "用法: bash $0 <model_path> <output_dir> [data_path] [prompt_type:cot|coc]" >&2
  exit 1
fi

MODEL_PATH="$1"
OUTPUT_DIR="$2"
# 默认数据路径 = 脚本目录的上一级，即 LVEval-main 的父目录 LV-EVAL
DATA_PATH="${3:-$SCRIPT_DIR/..}"
PROMPT_TYPE="${4:-coc}"

python prediction_vllm.py \
  --prompt-type "${PROMPT_TYPE}" \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-max-length 131072 \
  --max-new-tokens 8192 \
  --batch-size 50
