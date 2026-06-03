"""
将 MuSiQue 格式的 JSON 文件转换为带有 supporting_facts 字段的格式。
从 paragraphs 中提取 is_supporting 为 true 的 paragraph_text，
使用 PunktSentenceTokenizer 将段落拆分为句子，组成 supporting_facts 列表，
与 HotpotQA 格式对齐。
"""

import json
from nltk.tokenize import PunktSentenceTokenizer

# ========== 在这里修改输入输出文件路径 ==========
input_file = "/user/jinzhensheng/Musique/sampled_4000_new.jsonl"
output_file = "/user/jinzhensheng/construct_data/musique_with_supporting_facts_4000_new_.jsonl"
# ===============================================

tokenizer = PunktSentenceTokenizer()


def split_sentences(text):
    """使用 PunktSentenceTokenizer 分句，并合并被错误拆分的碎片。"""
    raw = tokenizer.tokenize(text)

    # 后处理：将过短的碎片句（如 "632)"）合并到前一句
    merged = []
    for sent in raw:
        if merged and len(sent.split()) <= 4 and not sent[0].isupper():
            merged[-1] = merged[-1] + " " + sent
        else:
            merged.append(sent)
    return merged


def convert(input_path, output_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    for item in data:
        supporting_facts = []
        for p in item.get('paragraphs', []):
            if p.get('is_supporting'):
                supporting_facts.extend(split_sentences(p['paragraph_text']))
        item['supporting_facts'] = supporting_facts

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"处理完成！共处理 {len(data)} 条数据。")


convert(input_file, output_file)