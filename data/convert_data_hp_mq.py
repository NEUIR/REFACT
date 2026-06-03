#!/usr/bin/env python3
"""
将 sampled_hotpotqa / sampled_musique 格式转换为 deepdive 对话格式 (jsonl)
支持两种输入格式自动识别
"""

import json
import sys
from pathlib import Path


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


def convert_item(item):
    """
    自动识别 hotpotqa 和 musique 两种格式
    
    Args:
        item: 单条数据（hotpotqa 或 musique 格式）
        
    Returns:
        dict: deepdive 对话格式的数据
    """
    # 判断数据格式并构建上下文字符串
    if 'context' in item:
        # HotPotQA 格式: context 是字符串
        context_str = item['context']
    elif 'paragraphs' in item:
        # Musique 格式: paragraphs 是列表
        context_str = "\n\n".join(
            f"Title:{para['title']}\nContent:{para['paragraph_text']}"
            for para in item['paragraphs']
        )
    else:   
        raise ValueError("无法识别数据格式：缺少 'context' 或 'paragraphs' 字段")

    # 使用模板生成用户消息
    user_content = PROMPT_TEMPLATE.format(
        question=item['question'],
        context=context_str,
    )

    # 构建 deepdive 格式
    deepdive_item = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": item['answer']},
        ]
    }

    # 保留原始元数据
    if 'id' in item:
        deepdive_item['_meta'] = {"original_id": item['id']}
        # 可选：保留更多元数据
        if 'level' in item:
            deepdive_item['_meta']['level'] = item['level']
        if 'supporting_facts' in item:
            deepdive_item['_meta']['supporting_facts'] = item['supporting_facts']

    return deepdive_item


def main():
    """主函数"""
    # ========== 配置文件路径 ==========
    input_file = "/user/jinzhensheng/hotpotqa/sampled_3000_new.json"
    output_file = "/user/jinzhensheng/construct_data/hotpotqa_3000_new.jsonl"
    # ==================================

    # 验证输入文件
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_file}", file=sys.stderr)
        sys.exit(1)

    # 创建输出目录（如果不存在）
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 读取输入文件
        print(f"正在读取: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        #with open(input_file, 'r', encoding='utf-8') as f:
        #    data = [json.loads(line) for line in f if line.strip()]
        if not isinstance(data, list):
            print(f"错误: 输入文件应包含列表格式的数据", file=sys.stderr)
            sys.exit(1)

        print(f"找到 {len(data)} 条数据待转换")

        # 转换并写入输出文件
        success_count = 0
        print(f"正在写入: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as fout:
            for idx, item in enumerate(data, 1):
                try:
                    converted = convert_item(item)
                    fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    success_count += 1
                    
                    # 进度提示
                    if idx % 100 == 0:
                        print(f"  已处理: {idx}/{len(data)}")
                        
                except Exception as e:
                    print(f"警告: 处理第 {idx} 条数据时出错: {e}", file=sys.stderr)
                    continue

        print(f"✓ 转换完成! 成功转换 {success_count}/{len(data)} 条数据")
        print(f"✓ 输出文件: {output_file}")

    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"错误: 文件读写失败: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: 未预期的错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()