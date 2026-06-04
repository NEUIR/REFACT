'''
处理longbench文本数据，格式为messages格式
原始数据格式：{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ..}]}

支持 evidence_based 奖励训练：
  - 从 user message 中提取 question 和 context (source_text)
  - 从 assistant message 中提取 <answer>...</answer> 作为 ground_truth
  - 将 question 和 source_text 放入 extra_info 供奖励函数使用
'''

import argparse
import os
import json
import glob
import re

import datasets


def load_jsonl(file_path):
    """加载jsonl文件"""
    all_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            all_data.append(data)
    return all_data


# ============================================================
#  从 user message 中提取 question 和 context
# ============================================================

def extract_question_and_context(user_message: str):
    """
    从 user message 中提取 question 和 context (source_text)。
    
    支持多种常见格式：
      1. 有明确的 "Question:" 和 "Context:" 标记
      2. 有 "Question:" 但 context 在前面的大段文本中
      3. 其他自定义格式
    
    Returns:
        (question, context) 元组
    """
    question = ""
    context = ""
    
    # 格式1: 明确的 Question: 和 Context: 标记 (如样例数据)
    q_match = re.search(
        r'Question:\s*\n(.*?)(?:\n\s*\n|\nContext:)',
        user_message, re.DOTALL
    )
    c_match = re.search(
        r'Context:\s*\n(.*?)$',
        user_message, re.DOTALL
    )
    
    if q_match and c_match:
        question = q_match.group(1).strip()
        context = c_match.group(1).strip()
        return question, context
    
    # 格式2: Question: 在文本中，context 是之前/之后的全部内容
    q_match2 = re.search(r'Question:\s*(.*?)(?:\n|$)', user_message)
    if q_match2:
        question = q_match2.group(1).strip()
        # context 是去掉 question 行和 instruction 部分后的剩余文本
        # 尝试找到 Context: 标记
        c_match2 = re.search(r'Context:\s*(.*)', user_message, re.DOTALL)
        if c_match2:
            context = c_match2.group(1).strip()
        else:
            # 没有明确的 Context 标记，取问题之后的所有内容
            q_end = q_match2.end()
            remaining = user_message[q_end:].strip()
            if remaining:
                context = remaining
            else:
                # 问题在末尾，context 是问题之前的内容(去掉instruction)
                context = user_message[:q_match2.start()].strip()
        return question, context

    # 格式3: 没有明确标记，整个 message 作为 context，question 为空
    # 此时调用方应使用完整 user_message 作为 prompt
    return "", user_message


def extract_answer_from_assistant(assistant_message: str) -> str:
    """
    从 assistant message 中提取 <answer>...</answer> 标签内容作为 ground_truth。
    如果没有 answer 标签，则使用整个 assistant message（去掉 think 部分）。
    """
    # 优先提取 <answer> 标签
    answer_match = re.search(r'<answer>(.*?)</answer>', assistant_message, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # 退回: 去掉 <think>...</think> 后的内容
    if '<think>' in assistant_message and '</think>' in assistant_message:
        after_think = assistant_message.split('</think>')[-1].strip()
        if after_think:
            return after_think
    
    # 最终退回: 使用整个 assistant message
    return assistant_message.strip()


# ============================================================
#  Prompt 模板
# ============================================================

# 原有的额外提示
#additional_prompt_legacy = (
#    "Hint: Please identify all relevant information in the article. "
#    "You should explicitly cite the relevant original text in your reasoning "
#    "process to avoid hallucinations and ensure no information is missed."
#)

# ★ 新增：evidence_based 训练的格式化提示
EVIDENCE_BASED_SYSTEM_PROMPT = (
    "You are given a question and context.\n"
    "Instructions:\n"
    "  - Answer the question by reasoning through the provided context.\n"
    "  - Do not use prior knowledge and do not fabricate information.\n"
    "  - During your reasoning process, cite evidence using <evidence N>verbatim_text</evidence> tags. Indices (N) must start at 1 and increment strictly (1, 2, 3...). Place citations from context immediately before the claim, inference or conclusion they support.\n"
    "  - Citations can be entity-level, phrase-level, sentence-level, or multi-sentence.\n"
    "  - Present your final answer clearly within <answer>...</answer> tags."
)


def format_user_prompt(question: str, context: str, mode: str = "evidence_based") -> str:
    """
    根据模式格式化 user prompt。
    
    Args:
        question: 提取出的问题
        context: 提取出的上下文
        mode: "evidence_based" 使用新格式, "legacy" 保持原格式
    """
    if mode == "evidence_based":
        return (
            f"{EVIDENCE_BASED_SYSTEM_PROMPT}\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}"
        )
    else:
        # legacy 模式: 保持原始 user_message 不变
        return f"Question:\n{question}\n\nContext:\n{context}"


# ============================================================
#  主流程
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="输入文件夹路径，包含多个json/jsonl文件")
    parser.add_argument("--output_path", default="/user/xuxiaoyue/rldata/", help="输出目录")
    parser.add_argument("--test_size", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--source", type=str, default="ruler_cite", help="数据源名称")
    parser.add_argument("--max_limit", type=int, default=None, help="最大数据量")
    parser.add_argument("--add_prompt", action="store_true", help="额外提示 (legacy模式)")
    parser.add_argument("--mode", type=str, default="evidence_based",
                        choices=["evidence_based", "legacy"],
                        help="数据处理模式: evidence_based(新四种奖励) 或 legacy(原有逻辑)")
    # ★ 新增：允许自定义奖励权重
    parser.add_argument("--format_weight", type=float, default=0.1, help="格式奖励权重")
    parser.add_argument("--correctness_weight", type=float, default=0.5, help="正确性奖励权重")
    parser.add_argument("--consistency_weight", type=float, default=0.2, help="一致性奖励权重")
    parser.add_argument("--citation_validity_weight", type=float, default=0.2, help="引用有效性奖励权重")
    parser.add_argument("--similarity_threshold", type=float, default=0.8, help="一致性匹配阈值")

    args = parser.parse_args()

    # 获取文件夹中所有的json和jsonl文件
    json_files = []
    for pattern in ["*.json", "*.jsonl"]:
        json_files.extend(glob.glob(os.path.join(args.input_path, pattern)))

    if not json_files:
        raise ValueError(f"在目录 {args.input_path} 中没有找到json或jsonl文件")

    print(f"找到 {len(json_files)} 个文件: {json_files}")
    print(f"处理模式: {args.mode}")
    print(f"数据源: {args.source}")

    # 使用datasets库加载多个json文件
    def gen(files):
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if "messages" in data:
                            yield {"messages": data["messages"]}
                    except Exception:
                        pass

    ds = datasets.Dataset.from_generator(gen, gen_kwargs={"files": json_files})
    dataset = datasets.DatasetDict({"train": ds})
    print(f"Loaded {len(dataset['train'])} examples")

    # 分割训练测试集
    all_dataset = dataset["train"]
    if args.max_limit:
        all_dataset = all_dataset.select(range(min(args.max_limit, len(all_dataset))))

    train_test_split = all_dataset.train_test_split(
        test_size=args.test_size, shuffle=True, seed=42
    )
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    # 统计信息
    stats = {"total": 0, "with_question": 0, "with_context": 0, "with_answer_tag": 0}

    # 处理数据的映射函数
    def make_map_fn(split):
        def process_fn(example, idx):
            messages = example["messages"]

            # 提取用户消息和助手回复
            user_message = None
            assistant_message = None

            for msg in messages:
                if msg["role"] == "user":
                    user_message = msg["content"]
                elif msg["role"] == "assistant":
                    assistant_message = msg["content"]

            stats["total"] += 1

            if args.mode == "evidence_based":
                # ★ evidence_based 模式：提取 question, context, answer
                question, context = extract_question_and_context(user_message)
                ground_truth = extract_answer_from_assistant(assistant_message)

                if question:
                    stats["with_question"] += 1
                if context:
                    stats["with_context"] += 1
                if '<answer>' in (assistant_message or ''):
                    stats["with_answer_tag"] += 1

                # 如果提取到了 question，重新格式化 prompt
                if question:
                    formatted_prompt = format_user_prompt(question, context, mode="evidence_based")
                else:
                    # 没有提取到结构化的 question，保留原 user_message 并追加格式提示
                    formatted_prompt = (
                        f"{EVIDENCE_BASED_SYSTEM_PROMPT}\n\n{user_message}"
                    )

                data = {
                    "data_source": args.source,
                    "prompt": [
                        {
                            "role": "user",
                            "content": formatted_prompt,
                        }
                    ],
                    "ability": "ruler_cite",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": ground_truth,
                    },
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "question": question,
                        "source_text": context,
                        "input_length": len(user_message) if user_message else 0,
                        "reasoning_hop": 0,
                        # 奖励权重配置
                        "format_weight": args.format_weight,
                        "correctness_weight": args.correctness_weight,
                        "consistency_weight": args.consistency_weight,
                        "citation_validity_weight": args.citation_validity_weight,
                        "similarity_threshold": args.similarity_threshold,
                    },
                }

            else:
                # legacy 模式：保持原有逻辑
                if args.add_prompt:
                    user_message = user_message + "\n\n" + additional_prompt_legacy

                data = {
                    "data_source": args.source,
                    "prompt": [
                        {
                            "role": "user",
                            "content": user_message,
                        }
                    ],
                    "ability": args.source,
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": assistant_message,
                    },
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "input_length": 0,
                        "reasoning_hop": 0,
                    },
                }

            # 删除原始的messages字段
            if "messages" in example:
                del example["messages"]

            # 调试输出
            if idx < 3:
                print("=" * 80)
                print(f"[Sample {idx}] mode={args.mode}")
                print(f"  prompt (last 200 chars): ...{data['prompt'][0]['content'][-200:]}")
                print(f"  ground_truth: {data['reward_model']['ground_truth'][:100]}")
                if args.mode == "evidence_based":
                    print(f"  question: {data['extra_info']['question'][:100]}")
                    print(f"  source_text length: {len(data['extra_info']['source_text'])}")
                print("-" * 80)

            return data

        return process_fn

    # 应用处理函数
    all_dataset = all_dataset.map(function=make_map_fn("all"), with_indices=True)
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # 创建输出目录
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    # 保存处理后的数据
    all_dataset.to_parquet(os.path.join(output_path, "all.parquet"))
    train_dataset.to_parquet(os.path.join(output_path, "train.parquet"))
    test_dataset.to_parquet(os.path.join(output_path, "test.parquet"))

    print(f"\n数据处理完成！")
    print(f"模式: {args.mode}")
    print(f"数据源: {args.source}")
    print(f"总数据量: {len(all_dataset)}")
    print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")
    if args.mode == "evidence_based":
        print(f"\n统计信息:")
        print(f"  提取到 question: {stats['with_question']}/{stats['total']}")
        print(f"  提取到 context:  {stats['with_context']}/{stats['total']}")
        print(f"  包含 <answer> 标签: {stats['with_answer_tag']}/{stats['total']}")
        print(f"\n奖励权重配置:")
        print(f"  format_weight:            {args.format_weight}")
        print(f"  correctness_weight:       {args.correctness_weight}")
        print(f"  consistency_weight:       {args.consistency_weight}")
        print(f"  citation_validity_weight: {args.citation_validity_weight}")
        print(f"  similarity_threshold:     {args.similarity_threshold}")