#!/usr/bin/env python3
"""
一站式构造 HotpotQA + Musique 训练数据。

流程：
  1) 仅对 Musique 执行 `add_support_facts` 逻辑：
     - 从 paragraphs 中抽取 is_supporting=True 的段落，用 PunktSentenceTokenizer 分句，
       合成 `supporting_facts` 字段（与 HotpotQA 对齐）。
     - HotpotQA 已自带 supporting_facts，跳过这一步。
  2) 对两者执行 `convert_data_hp_mq` 的逻辑：
     - 自动识别 hotpotqa（含 'context'）或 musique（含 'paragraphs'）格式；
     - 输出 deepdive 对话格式 {"messages":[...], "_meta":{...}}。
  3) 把两者分别写成 jsonl，再合并成一个 jsonl。

最终产出格式（两者一致）：
  {
    "messages": [
      {"role": "user", "content": "<填好的 PROMPT_TEMPLATE>"},
      {"role": "assistant", "content": "<answer>"}
    ],
    "_meta": {
      "original_id": "...",
      "supporting_facts": ["...", ...],
      "level": "...",        # 仅 HotpotQA 有
      "source": "hotpotqa" 或 "musique"   # 合并时加，便于区分来源
    }
  }
"""

import argparse
import json
import random
import sys
from pathlib import Path

from nltk.tokenize import PunktSentenceTokenizer


# ============================================================
# 与 convert_data_hp_mq.py 完全一致的 Prompt 模板
# ============================================================
PROMPT_TEMPLATE = """\
You are given a question and context.
Instructions:
  - Answer the question by reasoning through the provided context.
  - Do not use prior knowledge and do not fabricate information.
  - During the reasoning process, citing evidence with <evidence N>verbatim_text</evidence> tags at appropriate granularity, where N is a sequential integer (1, 2, 3...). Place citations from context immediately before the claim, inference or conclusion they support.
  - Citations can be entity-level, phrase-level, sentence-level, or multi-sentence.
  - Present your final answer clearly within <answer>...</answer> tags.

Question:
{question}

Context:
{context}"""


# ============================================================
# Step 1: 仅作用于 Musique —— 生成 supporting_facts
# ============================================================
_tokenizer = PunktSentenceTokenizer()


def split_sentences(text: str):
    """与 add_support_facts.py 完全一致的分句 + 合并短碎片逻辑。"""
    raw = _tokenizer.tokenize(text)
    merged = []
    for sent in raw:
        if merged and len(sent.split()) <= 4 and not sent[0].isupper():
            merged[-1] = merged[-1] + " " + sent
        else:
            merged.append(sent)
    return merged


def add_supporting_facts_musique(item: dict) -> dict:
    """对单条 Musique 数据加上 supporting_facts 字段。"""
    supporting_facts = []
    for p in item.get("paragraphs", []):
        if p.get("is_supporting"):
            supporting_facts.extend(split_sentences(p["paragraph_text"]))
    item["supporting_facts"] = supporting_facts
    return item


# ============================================================
# Step 2: 转成 deepdive 对话格式
# ============================================================
def convert_item(item: dict, source: str) -> dict:
    """
    自动识别 hotpotqa / musique 格式并转换。

    Args:
        item: 单条原始（或加过 supporting_facts 后）的数据
        source: "hotpotqa" 或 "musique"，写进 _meta.source
    """
    if "context" in item:
        # HotpotQA: context 已经是字符串
        context_str = item["context"]
    elif "paragraphs" in item:
        # Musique: 拼成 "Title:..\nContent:.." 段落，段间 \n\n
        context_str = "\n\n".join(
            f"Title:{para['title']}\nContent:{para['paragraph_text']}"
            for para in item["paragraphs"]
        )
    else:
        raise ValueError("无法识别数据格式：缺少 'context' 或 'paragraphs' 字段")

    user_content = PROMPT_TEMPLATE.format(
        question=item["question"],
        context=context_str,
    )

    deepdive_item = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": item["answer"]},
        ],
        "_meta": {
            "original_id": item.get("id"),
            "source": source,
        },
    }
    if "level" in item:
        deepdive_item["_meta"]["level"] = item["level"]
    if "supporting_facts" in item:
        deepdive_item["_meta"]["supporting_facts"] = item["supporting_facts"]

    return deepdive_item


# ============================================================
# I/O 工具
# ============================================================
def load_any(path: Path):
    """自动识别 .json (list) 或 .jsonl (一行一条)。"""
    text = path.read_text(encoding="utf-8").lstrip()
    if not text:
        return []
    if text[0] == "[":
        # JSON 数组
        return json.loads(text)
    # JSONL
    data = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            data.append(json.loads(line))
    return data


def write_jsonl(items, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ============================================================
# 主流程
# ============================================================
def process_hotpotqa(input_path: Path, output_path: Path) -> list:
    print(f"[HotpotQA] 读取: {input_path}")
    data = load_any(input_path)
    print(f"[HotpotQA] 共 {len(data)} 条")

    converted = []
    for idx, item in enumerate(data, 1):
        try:
            converted.append(convert_item(item, source="hotpotqa"))
        except Exception as e:
            print(f"[HotpotQA] 警告：第 {idx} 条转换失败: {e}", file=sys.stderr)

    write_jsonl(converted, output_path)
    print(f"[HotpotQA] 转换完成 -> {output_path}  ({len(converted)} 条)")
    return converted


def process_musique(input_path: Path, output_path: Path) -> list:
    print(f"[Musique] 读取: {input_path}")
    data = load_any(input_path)
    print(f"[Musique] 共 {len(data)} 条，开始补 supporting_facts ...")

    for item in data:
        add_supporting_facts_musique(item)

    converted = []
    for idx, item in enumerate(data, 1):
        try:
            converted.append(convert_item(item, source="musique"))
        except Exception as e:
            print(f"[Musique] 警告：第 {idx} 条转换失败: {e}", file=sys.stderr)

    write_jsonl(converted, output_path)
    print(f"[Musique] 转换完成 -> {output_path}  ({len(converted)} 条)")
    return converted


def merge_and_write(hotpot_items, musique_items, merged_path: Path, shuffle: bool, seed: int):
    combined = list(hotpot_items) + list(musique_items)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(combined)
    write_jsonl(combined, merged_path)
    print(
        f"[Merged] 合并完成 -> {merged_path}  "
        f"(共 {len(combined)} 条 = hotpotqa {len(hotpot_items)} + musique {len(musique_items)})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="一键构造 HotpotQA + Musique 合并训练数据"
    )
    parser.add_argument(
        "--hotpotqa-input",
        default="/user/jinzhensheng/hotpotqa/hotpotqa.json",
        help="HotpotQA 原始数据路径（.json 数组或 .jsonl）",
    )
    parser.add_argument(
        "--musique-input",
        default="/user/jinzhensheng/Musique/musique_full_v1.0_train.jsonl",
        help="Musique 原始数据路径（.jsonl）",
    )
    parser.add_argument(
        "--output-dir",
        default="/user/jinzhensheng/construct_coc_data/output",
        help="输出目录（会生成 hotpotqa.jsonl / musique.jsonl / combined.jsonl）",
    )
    parser.add_argument(
        "--hotpotqa-output-name", default="hotpotqa.jsonl",
        help="HotpotQA 转换后输出文件名",
    )
    parser.add_argument(
        "--musique-output-name", default="musique.jsonl",
        help="Musique 转换后输出文件名",
    )
    parser.add_argument(
        "--merged-output-name", default="combined.jsonl",
        help="合并后输出文件名",
    )
    parser.add_argument(
        "--skip-hotpotqa", action="store_true",
        help="跳过 HotpotQA 处理",
    )
    parser.add_argument(
        "--skip-musique", action="store_true",
        help="跳过 Musique 处理",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="合并时不打乱顺序（默认会打乱）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="合并打乱用的随机种子",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hotpot_items, musique_items = [], []

    if not args.skip_hotpotqa:
        hotpot_input = Path(args.hotpotqa_input)
        if not hotpot_input.exists():
            sys.exit(f"错误：HotpotQA 输入不存在: {hotpot_input}")
        hotpot_items = process_hotpotqa(
            hotpot_input, output_dir / args.hotpotqa_output_name
        )

    if not args.skip_musique:
        musique_input = Path(args.musique_input)
        if not musique_input.exists():
            sys.exit(f"错误：Musique 输入不存在: {musique_input}")
        musique_items = process_musique(
            musique_input, output_dir / args.musique_output_name
        )

    if hotpot_items or musique_items:
        merge_and_write(
            hotpot_items,
            musique_items,
            output_dir / args.merged_output_name,
            shuffle=not args.no_shuffle,
            seed=args.seed,
        )

    print("全部完成。")


if __name__ == "__main__":
    main()
