from metrics import (
    qa_f1_score,
    qa_f1_score_with_gold_ans,
    qa_f1_zh_score,
    qa_f1_zh_score_with_gold_ans,
    rouge_zh_score_blacklist,
)

REASONING_INSTRUCTION_EN = (
    "\n\nInstructions:\n"
    "  - During the reasoning process, citing evidence with "
    "<evidence N>verbatim_text</evidence> tags at appropriate granularity, "
    "where N is a sequential integer (1, 2, 3...)."
    "Place citations from article immediately before the claim, "
    "inference or conclusion they support.\n"
    "  - Only give me the answer. Present your final answer clearly within <answer>...</answer> tags.\n\n"
)

REASONING_INSTRUCTION_ZH = (
    "\n\n要求：\n"
    "  - 在推理过程中，使用 <evidence N>原文引用</evidence> 标签以合适的粒度引用证据，"
    "其中 N 是顺序整数（1, 2, 3...）。序号N从1严格递增。"
    "将文章中的引用紧接在其支撑的论断、推理或结论之前。\n"
    "  - 只需要直接给出问题的答案。将最终答案清晰地放在 <answer>...</answer> 标签内。\n\n"
)

ANSWER_TAG_EN = " Present your final answer within <answer>...</answer> tags."
ANSWER_TAG_ZH = " 请将最终答案放在 <answer>...</answer> 标签内。"

COC_DATASET_PROMPT = {
    "hotpotwikiqa_mixup": (
        "Answer the question based on the given passages. "
        "Questions and answers are only relevant to some passages."
        + REASONING_INSTRUCTION_EN
        + "Article: {context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "loogle_SD_mixup": (
        "Please answer the following question based on the given passages. "
        "Questions and answers are only relevant to one passage."
        + REASONING_INSTRUCTION_EN
        + "Article: {context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "loogle_CR_mixup": (
        "Please answer the following question based on the given passages. "
        "Questions and answers are only relevant to one passage."
        + REASONING_INSTRUCTION_EN
        + "Article: {context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "loogle_MIR_mixup": (
        "Please answer the following question based on the given passages. "
        "Questions and answers are only relevant to one passage."
        + REASONING_INSTRUCTION_EN
        + "Article: {context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "multifieldqa_en_mixup": (
        "Please answer the following question based on the given passages. "
        "Questions and answers are only relevant to one passage."
        + REASONING_INSTRUCTION_EN
        + "Article: {context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "multifieldqa_zh_mixup": (
        "请阅读以下文章并用中文回答问题，问题和答案只与其中一篇文章有关。"
        + REASONING_INSTRUCTION_ZH
        + "文章：{context}\n\n"
        "问题：{input}\n"
        "回答："
    ),
    "factrecall_en": (
        "Please answer the following questions based on the given article."
        + REASONING_INSTRUCTION_EN
        + "Article: {context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "factrecall_zh": (
        "请基于给定的文章回答下述问题。"
        + REASONING_INSTRUCTION_ZH
        + "文章：{context}\n\n"
        "问题：{input}\n"
        "回答："
    ),
    "cmrc_mixup": (
        "请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。"
        + REASONING_INSTRUCTION_ZH
        + "文章：{context}\n\n"
        "问题：{input}\n"
        "回答："
    ),
    "lic_mixup": (
        "请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。"
        + REASONING_INSTRUCTION_ZH
        + "文章：{context}\n\n"
        "问题：{input}\n"
        "回答："
    ),
    "dureader_mixup": (
        "请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。"
        + REASONING_INSTRUCTION_ZH
        + "文章：{context}\n\n"
        "问题：{input}\n"
        "回答："
    ),
}

COT_DATASET_PROMPT = {
    "hotpotwikiqa_mixup": "Answer the question based on the given passages. Questions and answers are only relevant to some passages. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\n" + ANSWER_TAG_EN + "\n\nQuestion: {input}\nAnswer:",
    "loogle_SD_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\n" + ANSWER_TAG_EN + "\n\nQuestion: {input}\nAnswer:",
    "loogle_CR_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\n" + ANSWER_TAG_EN + "\n\nQuestion: {input}\nAnswer:",
    "loogle_MIR_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\n" + ANSWER_TAG_EN + "\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_en_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\n" + ANSWER_TAG_EN + "\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh_mixup": "请阅读以下文章并用中文回答问题，问题和答案只与其中一篇文章有关。只需要直接给出问题的答案，不要输出其他任何解释和证据。\n\n文章：{context}\n\n只需要直接给出问题的答案，不要输出其他任何解释和证据。" + ANSWER_TAG_ZH + "\n\n问题：{input}\n回答：",
    "factrecall_en": "Please answer the following questions based on the given article.\n\nArticle: {context}\n\nPlease answer the following questions based on the above article." + ANSWER_TAG_EN + "\n\nQuestion: {input}\nAnswer:",
    "factrecall_zh": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n现在请基于上述文章回答下面的问题。" + ANSWER_TAG_ZH + "\n\n问题：{input}\n回答：",
    "cmrc_mixup": "请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。\n\n文章：{context}\n\n" + ANSWER_TAG_ZH + "\n\n问题：{input}\n回答：",
    "lic_mixup": "请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。\n\n文章：{context}\n\n" + ANSWER_TAG_ZH + "\n\n问题：{input}\n回答：",
    "dureader_mixup": "请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。\n\n文章：{context}\n\n" + ANSWER_TAG_ZH + "\n\n问题：{input}\n回答：",
}

PROMPT_TYPES = {
    "cot": COT_DATASET_PROMPT,
    "coc": COC_DATASET_PROMPT,
}


def get_dataset_prompt(prompt_type: str, dataset_name: str) -> str:
    prompt_type = prompt_type.lower()
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(
            f"未知的 prompt_type: {prompt_type}，可选值: {list(PROMPT_TYPES.keys())}"
        )
    prompt_dict = PROMPT_TYPES[prompt_type]
    if dataset_name not in prompt_dict:
        raise KeyError(
            f"prompt_type={prompt_type} 下找不到数据集 {dataset_name} 的 prompt 模板"
        )
    return prompt_dict[dataset_name]


DATASET_METRIC = {
    "hotpotwikiqa_mixup": qa_f1_score_with_gold_ans,
    "loogle_SD_mixup": qa_f1_score_with_gold_ans,
    "loogle_CR_mixup": qa_f1_score_with_gold_ans,
    "loogle_MIR_mixup": qa_f1_score_with_gold_ans,
    "multifieldqa_en_mixup": qa_f1_score_with_gold_ans,
    "multifieldqa_zh_mixup": qa_f1_zh_score_with_gold_ans,
    "factrecall_en": qa_f1_score,
    "factrecall_zh": qa_f1_zh_score,
    "cmrc_mixup": qa_f1_zh_score_with_gold_ans,
    "lic_mixup": qa_f1_zh_score_with_gold_ans,
    "dureader_mixup": rouge_zh_score_blacklist,
}

DATASET_SELECTED_FULL = [
    "hotpotwikiqa_mixup",
    "loogle_SD_mixup",
    "loogle_CR_mixup",
    "loogle_MIR_mixup",
    "multifieldqa_en_mixup",
    "multifieldqa_zh_mixup",
    "factrecall_en",
    "factrecall_zh",
    "cmrc_mixup",
    "lic_mixup",
    "dureader_mixup",
]

DATASET_SELECTED = [
    "hotpotwikiqa_mixup",
    "loogle_SD_mixup",
    "multifieldqa_en_mixup",
    "multifieldqa_zh_mixup",
    "factrecall_en",
    "factrecall_zh",
]

DATASET_LENGTH_LEVEL = [
    "16k",
    "32k",
    "64k",
    "128k",
]

DATASET_LENGTH_LEVEL_DEBUG = [
    "16k",
]
