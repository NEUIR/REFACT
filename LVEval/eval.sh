set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "用法: bash $0 <input_dir> [prompt_type:cot|coc|all]" >&2
  exit 1
fi

INPUT_DIR="$1"
PROMPT_TYPE="${2:-coc}"

python evaluation.py \
  --input-dir "${INPUT_DIR}" \
  --prompt-type "${PROMPT_TYPE}"
