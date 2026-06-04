#!/usr/bin/env bash
set -e

# 简单版一键造数据脚本：
# 1. 下载 HotpotQA CARE 和 MuSiQue
# 2. 调现有脚本处理并合并数据
# 3. 调 OpenAI-compatible API 生成推理轨迹

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-"$ROOT_DIR/pipeline_output"}"

RAW_DIR="$OUT_DIR/raw"
PROCESSED_DIR="$OUT_DIR/processed"
TRACE_FILE="$OUT_DIR/reasoning_trajectories.jsonl"

HOTPOT_RAW="$RAW_DIR/hotpotqa_care.jsonl"
MUSIQUE_RAW="$RAW_DIR/musique_train.jsonl"
COMBINED_FILE="$PROCESSED_DIR/combined.jsonl"

API_BASE="${API_BASE:-https://api.openai.com/v1}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
MODEL="${MODEL:-gpt-4.1-mini}"
API_LIMIT="${API_LIMIT:-20}"
SKIP_API="${SKIP_API:-0}"

mkdir -p "$RAW_DIR" "$PROCESSED_DIR"

echo "==> 1. 下载数据"
python "$ROOT_DIR/download_hf_datasets.py" \
  --only all \
  --hotpot-output "$HOTPOT_RAW" \
  --musique-output "$MUSIQUE_RAW" \
  --format jsonl

echo "==> 2. 处理并合并数据"
python "$ROOT_DIR/build_combined_dataset.py" \
  --hotpotqa-input "$HOTPOT_RAW" \
  --musique-input "$MUSIQUE_RAW" \
  --output-dir "$PROCESSED_DIR" \
  --merged-output-name "$(basename "$COMBINED_FILE")"

if [ "$SKIP_API" = "1" ]; then
  echo "==> 3. 跳过 API 生成推理轨迹"
  echo "完成：$COMBINED_FILE"
  exit 0
fi

if [ -z "$API_KEY" ]; then
  echo "缺少 API_KEY。请设置 OPENAI_API_KEY，或用 SKIP_API=1 跳过 API 步骤。"
  exit 1
fi

echo "==> 3. 调 API 生成推理轨迹"
API_BASE="$API_BASE" \
API_KEY="$API_KEY" \
MODEL="$MODEL" \
API_LIMIT="$API_LIMIT" \
COMBINED_FILE="$COMBINED_FILE" \
TRACE_FILE="$TRACE_FILE" \
python - <<'PY'
import json
import os
import urllib.request

api_base = os.environ["API_BASE"].rstrip("/")
api_key = os.environ["API_KEY"]
model = os.environ["MODEL"]
limit = int(os.environ["API_LIMIT"])
input_file = os.environ["COMBINED_FILE"]
output_file = os.environ["TRACE_FILE"]

def call_api(messages):
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    req = urllib.request.Request(
        api_base + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]

count = 0
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        if limit > 0 and count >= limit:
            break

        item = json.loads(line)
        user_messages = [m for m in item["messages"] if m["role"] == "user"]
        answer = call_api(user_messages)

        item["messages"] = user_messages + [{"role": "assistant", "content": answer}]
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        count += 1
        print("generated", count)

print("推理轨迹已保存到:", output_file)
PY

echo "全部完成"
echo "合并数据: $COMBINED_FILE"
echo "推理轨迹: $TRACE_FILE"
