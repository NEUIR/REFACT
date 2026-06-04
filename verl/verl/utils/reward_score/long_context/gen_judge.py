import re
import time
from .api import call_api
from .utils.gen_judge_config import JUDGE_SETTINGS


def score_with_llm_judge(pred, answer, judge_config_key):
    """原有的 LLM Judge 评分函数，完全不变。"""
    config = JUDGE_SETTINGS[judge_config_key]
    judge_model = config["model"]
    judge_system_prompt = config["system_prompt"]

    # 和原代码完全一致：system_prompt + 内容拼成一个字符串
    messages = f"{judge_system_prompt}\n\nStandard Answer: {answer}\n\nModel Prediction: {pred}"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = call_api(judge_model, messages, temperature=0.0)
            score = parse_score(response)
            if score is not None:
                return score
        except Exception as e:
            print(f"Error calling judge {judge_model}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return 0.0


def score_citation_validity(question: str, evidence_text: str,
                            judge_config_key: str = "citation_validity") -> str:
    """
    ★ 新增：引用有效性评分函数。
    调用方式与 score_with_llm_judge 完全一致：
    把 system_prompt + 内容拼成一个字符串，传给 call_api。
    """
    config = JUDGE_SETTINGS.get(judge_config_key)
    if config is None:
        print(f"⚠️ [CitationValidity] Config '{judge_config_key}' not found")
        return ""

    judge_model = config["model"]
    system_prompt = config["system_prompt"]

    # ★ 和 score_with_llm_judge 完全一样的拼接方式
    messages = (
        f"{system_prompt}\n\n"
        f"=== Referenced Evidence ===\n"
        f"{evidence_text}\n\n"
        f"=== Question ===\n"
        f"{question}\n\n"
        f"Based ONLY on the evidence above, answer the question. "
        f"Wrap your answer in <answer>...</answer> tags."
    )

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # ★ 和原代码完全一致的调用方式，不加任何新参数
            response = call_api(judge_model, messages, temperature=0.0)

            if not response or response == "No response":
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue

            # 提取 <answer> 标签内容
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                return answer_match.group(1).strip()

            # 没有 answer 标签，直接返回完整回复
            return response.strip()

        except Exception as e:
            print(f"⚠️ [CitationValidity] Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return ""


def parse_score(response):
    """原有的分数解析函数，完全不变。"""
    if not response:
        return None

    match = re.search(r'\[\[(\d+)\]\]', response)
    if match:
        val = int(match.group(1))
        if val in [0, 1]:
            return float(val)

    match = re.search(r'Score:\s*(\d+)', response, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if val in [0, 1]:
            return float(val)

    text = response.strip()
    if text in ['0', '1']:
        return float(text)

    return None