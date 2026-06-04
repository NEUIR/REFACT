# Evidence-based Reward Functions for RL Training
# 针对 Qwen3-8B thinking 模式 + evidence 引用的强化学习奖励函数
#
# 四个奖励维度：
# 1. 格式奖励 (format_reward): <think></think> 结构 + <evidence N></evidence> 成对匹配
# 2. 答案奖励 (answer_reward): </think> 之后的答案与 ground truth 的 F1 分数
# 3. 引用忠实度奖励 (faithfulness_reward): evidence 内容是否能在原文 context 中找到
# 4. 引用质量奖励 (evidence_reward): 提取 evidence 内容，调用 API 评估引用是否充分

import re
import string
import os
import time
from typing import Dict, Optional, List, Tuple


# ============================================================================
# 第一部分：基础工具函数
# ============================================================================

def detect_language(text: str) -> str:
    """检测文本是中文还是英文"""
    if not text:
        return "en"
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(re.sub(r'\s', '', text))
    if total_chars == 0:
        return "en"
    return "zh" if chinese_count / total_chars > 0.3 else "en"


def normalize_answer(s: str, language: str = "auto") -> str:
    """归一化答案：小写化、去标点、去冠词、规范空白"""
    if not s:
        return ""
    if language == "auto":
        language = detect_language(s)

    def remove_articles(text):
        if language == "zh":
            return re.sub(r'[的了吗呢啊嘛]', ' ', text)
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punc(text):
        if language == "zh":
            all_punc = set('，。！？；：""''（）【】《》、·…—' + string.punctuation)
        else:
            all_punc = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in all_punc)

    return ' '.join(remove_articles(remove_punc(s.lower())).split())


def tokenize(text: str) -> List[str]:
    """混合语言分词：英文按单词，中文按字"""
    if not text:
        return []
    return re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', text.lower())


def get_tokens(s: str, language: str = "auto") -> List[str]:
    """归一化后分词"""
    return tokenize(normalize_answer(s, language))


def compute_f1(gold: str, pred: str, language: str = "auto") -> float:
    """计算 F1 分数"""
    gold_toks = get_tokens(gold, language)
    pred_toks = get_tokens(pred, language)
    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        return 0.0
    common = set(gold_toks) & set(pred_toks)
    num_same = sum(min(gold_toks.count(t), pred_toks.count(t)) for t in common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


# ============================================================================
# 第二部分：解析工具
# ============================================================================

def extract_think_content(solution_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    提取 <think>...</think> 内的思考过程和之后的答案。
    
    Returns:
        (think_content, answer_content)
    """
    think_match = re.search(r'<think>(.*?)</think>', solution_str, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        answer_content = solution_str[think_match.end():].strip()
        return think_content, answer_content
    
    # 兼容：只有 </think> 的情况（模型可能在生成开头省略了 <think>）
    if '</think>' in solution_str:
        parts = solution_str.split('</think>', 1)
        think_content = re.sub(r'^<think>\s*', '', parts[0]).strip()
        answer_content = parts[1].strip() if len(parts) > 1 else ""
        return think_content, answer_content
    
    return None, None


def extract_evidence_pairs(think_content: str) -> List[Dict]:
    """
    提取所有成对匹配的 <evidence N>...</evidence> 标签。
    兼容 evidence / evdience 两种拼写，以及有无空格。
    
    Returns:
        List[Dict]: [{index, content, raw_tag}, ...]
    """
    evidence_list = []
    patterns = [
        r'<evidence\s+(\d+)>(.*?)</evidence>',
        r'<evdience\s+(\d+)>(.*?)</evdience>',
        r'<evidence(\d+)>(.*?)</evidence>',
        r'<evdience(\d+)>(.*?)</evdience>',
    ]
    
    seen_indices = set()
    for pattern in patterns:
        for match in re.finditer(pattern, think_content, re.DOTALL):
            idx = int(match.group(1))
            content = match.group(2).strip()
            if idx not in seen_indices and content:
                evidence_list.append({
                    'index': idx,
                    'content': content,
                    'raw_tag': match.group(0)
                })
                seen_indices.add(idx)
    
    evidence_list.sort(key=lambda x: x['index'])
    return evidence_list


def extract_context_from_prompt(prompt_str: str) -> str:
    """
    从 user prompt 中提取 Context 部分（原文内容）。
    
    支持的格式：
    - "Context:\n..."
    - "Context:\n...\n\nQuestion:"（Context 在 Question 之前）
    
    [修复] 增加更多格式兼容，并处理 prompt_str 本身就是纯文本的情况。
    """
    if not prompt_str:
        return ""
    
    # 格式1：Context: 之后的所有内容（最常见）
    match = re.search(r'Context:\s*\n(.*)', prompt_str, re.DOTALL)
    if match:
        context = match.group(1).strip()
        # 如果 context 后面还有 Question:，截断
        q_match = re.search(r'\n\s*Question:', context)
        if q_match:
            context = context[:q_match.start()].strip()
        return context
    
    # 格式2：Context 在 Question 之前（有些数据集是这个顺序）
    match = re.search(r'Question:.*?\n\n(.*)', prompt_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # [修复] 如果没有找到 Context/Question 标记，说明传入的不是完整 prompt
    # 返回空字符串而不是原文（避免把问题当成 context）
    return ""


def extract_question_from_prompt(prompt_str: str) -> str:
    """从 user prompt 中提取 Question 部分"""
    if not prompt_str:
        return ""
    match = re.search(r'Question:\s*\n?(.*?)(?:\n\nContext:|$)', prompt_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# ============================================================================
# 第三部分：格式奖励 (Format Reward)
# ============================================================================

def compute_format_reward(solution_str: str, debug: bool = False) -> Dict:
    """
    评分标准：
    - 有 <think></think> 结构：+0.3
    - 内部有 ≥1 对 <evidence N></evidence>：+0.4
    - evidence 序号从1开始连续：+0.15
    - </think> 之后有非空答案：+0.15
    """
    score = 0.0
    details = {
        'has_think_tags': False,
        'num_evidence_pairs': 0,
        'evidence_sequential': False,
        'has_answer_after_think': False,
    }
    
    think_content, answer_content = extract_think_content(solution_str)
    
    if think_content is not None:
        details['has_think_tags'] = True
        score += 0.3
        
        evidence_pairs = extract_evidence_pairs(think_content)
        details['num_evidence_pairs'] = len(evidence_pairs)
        
        if len(evidence_pairs) >= 1:
            score += 0.4
            indices = [e['index'] for e in evidence_pairs]
            if indices == list(range(1, len(indices) + 1)):
                details['evidence_sequential'] = True
                score += 0.15
        
        if answer_content and len(answer_content.strip()) > 0:
            details['has_answer_after_think'] = True
            score += 0.15
    
    if debug:
        print(f"[Format Reward] Score: {score:.2f}, Details: {details}")
    
    return {'score': score, 'details': details}


# ============================================================================
# 第四部分：答案奖励 (Answer Reward) — 只使用 F1
# ============================================================================

def compute_answer_reward(
    solution_str: str,
    ground_truth: str,
    language: str = "auto",
    debug: bool = False
) -> Dict:
    """
    答案奖励：直接用 F1 分数。
    """
    _, answer_content = extract_think_content(solution_str)
    
    if answer_content is not None:
        pred_answer = answer_content.strip()
    else:
        lines = [l.strip() for l in solution_str.strip().split('\n') if l.strip()]
        pred_answer = lines[-1] if lines else ""
    
    gt_str = str(ground_truth).strip()
    
    if not pred_answer or not gt_str:
        return {'score': 0.0, 'pred_answer': pred_answer, 'details': {}}
    
    f1 = compute_f1(gt_str, pred_answer, language)
    
    if debug:
        print(f"[Answer Reward] Pred: '{pred_answer[:100]}', GT: '{gt_str[:100]}', F1: {f1:.4f}")
    
    return {'score': f1, 'pred_answer': pred_answer, 'details': {'f1': f1}}


# ============================================================================
# 第五部分：引用忠实度奖励 (Faithfulness Reward)
# ============================================================================

def normalize_for_matching(text: str) -> str:
    """轻量归一化，用于引用匹配：只做小写 + 压缩空白，保留标点"""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.lower()).strip()


def check_evidence_in_context(evidence_content: str, context: str, threshold: float = 0.8) -> Tuple[bool, float]:
    """
    检查单条 evidence 是否能在原文 context 中找到。
    
    三级匹配策略：
    1. 精确子串匹配（归一化后）→ 1.0
    2. 省略号分段匹配 → 各片段命中率
    3. 滑动窗口 F1 匹配 → 最大 F1 ≥ threshold 视为匹配
    """
    if not evidence_content or not context:
        return False, 0.0
    
    norm_evidence = normalize_for_matching(evidence_content)
    norm_context = normalize_for_matching(context)
    
    # 策略1：精确子串匹配
    if norm_evidence in norm_context:
        return True, 1.0
    
    # 策略2：省略号 "..." 分段匹配
    if '...' in evidence_content:
        fragments = [f.strip() for f in evidence_content.split('...') if f.strip()]
        if fragments:
            found_count = sum(1 for f in fragments if normalize_for_matching(f) in norm_context)
            frag_ratio = found_count / len(fragments)
            if frag_ratio >= 0.8:
                return True, frag_ratio
    
    # 策略3：滑动窗口 F1
    ev_tokens = tokenize(norm_evidence)
    ctx_tokens = tokenize(norm_context)
    
    if not ev_tokens or not ctx_tokens:
        return False, 0.0
    
    window_size = len(ev_tokens)
    best_f1 = 0.0
    step = max(1, window_size // 4)
    
    for i in range(0, max(1, len(ctx_tokens) - window_size + 1), step):
        window = ctx_tokens[i:i + window_size + window_size // 2]
        common = set(ev_tokens) & set(window)
        num_same = sum(min(ev_tokens.count(t), window.count(t)) for t in common)
        if num_same == 0:
            continue
        precision = num_same / len(ev_tokens)
        recall = num_same / len(window)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        best_f1 = max(best_f1, f1)
        if best_f1 >= 0.95:
            break
    
    return best_f1 >= threshold, best_f1


def compute_faithfulness_reward(
    solution_str: str,
    prompt_str: str,
    debug: bool = False
) -> Dict:
    """
    引用忠实度奖励 = 忠实 evidence 数 / 总 evidence 数
    
    检查每条 <evidence> 中的内容是否真的来自输入原文。
    
    [修复] 当无法提取 context 时，返回 0.0 而不是 0.5，
    避免没有原文验证时给出虚假的中等分数。
    """
    think_content, _ = extract_think_content(solution_str)
    
    if think_content is None:
        return {'score': 0.0, 'details': {'reason': 'no_think_content'}}
    
    evidence_list = extract_evidence_pairs(think_content)
    
    if not evidence_list:
        return {'score': 0.0, 'details': {'reason': 'no_evidence_found'}}
    
    context = extract_context_from_prompt(prompt_str)
    
    if not context:
        if debug:
            print("[Faithfulness] WARNING: No context found in prompt, returning 0.0")
        return {'score': 0.0, 'details': {'reason': 'no_context_in_prompt'}}
    
    faithful_count = 0
    per_evidence_results = []
    
    for ev in evidence_list:
        is_faithful, match_score = check_evidence_in_context(ev['content'], context)
        per_evidence_results.append({
            'index': ev['index'],
            'is_faithful': is_faithful,
            'match_score': round(match_score, 4),
            'content_preview': ev['content'][:60] + '...' if len(ev['content']) > 60 else ev['content']
        })
        if is_faithful:
            faithful_count += 1
    
    score = faithful_count / len(evidence_list)
    
    if debug:
        print(f"[Faithfulness] {faithful_count}/{len(evidence_list)} evidence are faithful")
        for r in per_evidence_results:
            status = "✓" if r['is_faithful'] else "✗"
            print(f"  Evidence {r['index']}: {status} (score={r['match_score']}) | {r['content_preview']}")
    
    return {
        'score': score,
        'details': {
            'faithful_count': faithful_count,
            'total_count': len(evidence_list),
            'per_evidence': per_evidence_results,
        }
    }


# ============================================================================
# 第六部分：引用质量奖励 (Evidence Quality Reward) — API 评估
# ============================================================================

def build_evidence_prompt(evidence_list: List[Dict], question: str, language: str = "auto") -> str:
    """
    构建发送给 API 的 prompt。
    
    [修复] 
    1. 根据语言自动选择中/英文 prompt
    2. 更强的约束：明确禁止使用引用之外的任何知识
    3. 增加 system-level 约束提示
    """
    evidence_text = ""
    for e in evidence_list:
        evidence_text += f"[Citation {e['index']}]: {e['content']}\n"
    
    if language == "auto":
        language = detect_language(question)
    
    if language == "zh":
        return f"""你是一个严格的证据评估器。你必须且只能基于下面提供的引用内容来回答问题。

绝对禁止：
- 禁止使用你自己的知识
- 禁止使用训练数据中的信息
- 禁止进行任何推测或推理超出引用内容的范围
- 如果引用内容不足以直接回答问题，你必须回答"无法回答"

引用内容：
{evidence_text}

问题：{question}

请仅基于上述引用内容，给出简洁答案（一个词或一个短语即可）。如果引用不足以回答，回答"无法回答"。

答案："""
    else:
        return f"""You are a strict evidence evaluator. You must answer the question using ONLY the citations provided below.

ABSOLUTE RULES:
- You MUST NOT use any internal knowledge or training data
- You MUST NOT make any inference beyond what is explicitly stated in the citations
- If the citations are insufficient to answer the question, you MUST respond with "Cannot answer"

Citations:
{evidence_text}

Question: {question}

Answer using ONLY the citations above. Give a concise answer (a word or short phrase). If citations are insufficient, respond "Cannot answer".

Answer:"""


def call_llm_api(
    prompt: str,
    api_url: str = None,
    api_key: str = None,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    timeout: int = 30
) -> Optional[str]:
    """调用 LLM API，兼容 OpenAI / vLLM 格式"""
    import requests
    
    if api_url is None:
        api_url = os.environ.get("REWARD_API_URL", "http://localhost:8000/v1/chat/completions")
    if api_key is None:
        api_key = os.environ.get("REWARD_API_KEY", "EMPTY")
    
    # [修复] 使用 system message 来更强地约束模型行为
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict evidence evaluator. You can ONLY use the provided citations to answer. You have NO other knowledge. If citations are insufficient, say 'Cannot answer'."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"[Evidence API] Failed after {max_retries} retries: {e}")
                return None


def compute_evidence_reward(
    solution_str: str,
    ground_truth: str,
    question: str = "",
    api_url: str = None,
    api_key: str = None,
    api_model: str = "gpt-4o-mini",
    language: str = "auto",
    debug: bool = False
) -> Dict:
    """
    引用质量奖励：evidence 内容能否支撑出正确答案。
    
    API 可用 → API 基于 evidence 回答 → F1 比较
    API 不可用 → evidence 与 ground_truth 的 F1 覆盖率 × 0.6
    """
    think_content, _ = extract_think_content(solution_str)
    
    if think_content is None:
        return {'score': 0.0, 'details': {'reason': 'no_think_content'}}
    
    evidence_list = extract_evidence_pairs(think_content)
    if not evidence_list:
        return {'score': 0.0, 'details': {'reason': 'no_evidence_found'}}
    
    gt_str = str(ground_truth).strip()
    
    # 尝试 API
    if question:
        prompt = build_evidence_prompt(evidence_list, question, language=language)
        api_response = call_llm_api(prompt, api_url=api_url, api_key=api_key, model=api_model)
        
        if api_response is not None:
            # [修复] 如果 API 回答"无法回答"/"Cannot answer"，给 0 分
            if api_response.lower() in ["无法回答", "cannot answer", "n/a", "unable to answer"]:
                score = 0.0
            else:
                score = compute_f1(gt_str, api_response, language)
            if debug:
                print(f"[Evidence API] Response: '{api_response}', GT: '{gt_str}', F1: {score:.4f}")
            return {
                'score': score,
                'details': {'method': 'api', 'api_response': api_response, 'num_evidence': len(evidence_list)}
            }
    
    # 兜底：本地启发式
    all_evidence_text = ' '.join([e['content'] for e in evidence_list])
    coverage = compute_f1(gt_str, all_evidence_text, language)
    score = coverage * 0.6
    
    if debug:
        print(f"[Evidence Heuristic] Coverage F1: {coverage:.4f}, Final: {score:.4f}")
    
    return {
        'score': score,
        'details': {'method': 'heuristic', 'num_evidence': len(evidence_list), 'coverage_f1': coverage}
    }


# ============================================================================
# 第七部分：统一入口
# ============================================================================

def compute_score(
    solution_str: str,
    ground_truth: str,
    prompt_str: Optional[str] = None,
    question: Optional[str] = None,
    # 权重
    format_weight: float = 0.15,
    answer_weight: float = 0.40,
    faithfulness_weight: float = 0.25,
    evidence_weight: float = 0.20,
    # 配置
    language: str = "auto",
    api_url: str = None,
    api_key: str = None,
    api_model: str = "gpt-4o-mini",
    debug: bool = False,
    **kwargs
) -> Dict:
    """
    统一入口，四个维度加权求和。
    
    默认权重：format=0.15, answer=0.40, faithfulness=0.25, evidence=0.20
    
    [修复] 
    - question 现在可以直接传入，不再从 prompt_str 二次提取
    - prompt_str 用于 faithfulness（需要完整 user prompt 含 context）
    - question 用于 evidence quality API 调用
    """
    # [修复] question 提取逻辑：
    # 优先使用直接传入的 question 参数
    # 其次从 prompt_str 中提取
    if not question and prompt_str:
        question = extract_question_from_prompt(prompt_str)
    
    # 1. 格式奖励
    format_result = compute_format_reward(solution_str, debug=debug)
    
    # 2. 答案奖励（F1）
    answer_result = compute_answer_reward(solution_str, ground_truth, language=language, debug=debug)
    
    # 3. 引用忠实度奖励（需要完整 prompt 含 context）
    faithfulness_result = compute_faithfulness_reward(solution_str, prompt_str or "", debug=debug)
    
    # 4. 引用质量奖励（API，需要 question）
    evidence_result = compute_evidence_reward(
        solution_str, ground_truth, question=question or "",
        api_url=api_url, api_key=api_key, api_model=api_model,
        language=language, debug=debug
    )
    
    total_score = (
        format_weight * format_result['score'] +
        answer_weight * answer_result['score'] +
        faithfulness_weight * faithfulness_result['score'] +
        evidence_weight * evidence_result['score']
    )
    
    result = {
        'score': total_score,
        'format_score': format_result['score'],
        'answer_score': answer_result['score'],
        'faithfulness_score': faithfulness_result['score'],
        'evidence_score': evidence_result['score'],
        'acc': answer_result['score'] >= 0.8,
        'pred': answer_result.get('pred_answer', ''),
        'details': {
            'format': format_result['details'],
            'answer': answer_result['details'],
            'faithfulness': faithfulness_result['details'],
            'evidence': evidence_result['details'],
        }
    }
    
    if debug:
        print(f"\n{'='*60}")
        print(f"[Total] {total_score:.4f}")
        print(f"  Format:       {format_result['score']:.4f} x {format_weight} = {format_weight * format_result['score']:.4f}")
        print(f"  Answer(F1):   {answer_result['score']:.4f} x {answer_weight} = {answer_weight * answer_result['score']:.4f}")
        print(f"  Faithfulness: {faithfulness_result['score']:.4f} x {faithfulness_weight} = {faithfulness_weight * faithfulness_result['score']:.4f}")
        print(f"  Evidence API: {evidence_result['score']:.4f} x {evidence_weight} = {evidence_weight * evidence_result['score']:.4f}")
        print(f"{'='*60}\n")
    
    return result


# ============================================================================
# 第八部分：verl 框架集成接口
# ============================================================================

def default_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict = None,
    **kwargs
) -> Dict:
    """
    与 verl 框架兼容的入口。
    
    [修复] 关键修改：
    1. prompt_str 从 extra_info['prompt_str'] 获取（应为完整 user message，含 Context）
    2. question 从 extra_info['question'] 获取（纯问题文本）
    3. 二者分开传递，避免二次提取的 bug
    """
    prompt_str = None
    question = None
    api_url = None
    api_key = None
    api_model = "gpt-4o-mini"
    
    if extra_info:
        prompt_str = extra_info.get('prompt_str', None)
        question = extra_info.get('question', None)
        api_url = extra_info.get('api_url', None)
        api_key = extra_info.get('api_key', None)
        api_model = extra_info.get('api_model', 'gpt-4o-mini')
    
    # [修复] 处理 ground_truth 为 list 的情况
    if isinstance(ground_truth, list):
        # 取第一个元素作为主答案
        gt_str = str(ground_truth[0]).strip() if ground_truth else ""
    else:
        gt_str = str(ground_truth).strip()
    
    weight_configs = {
        "hotpotqa_evidence":   {"format_weight": 0.15, "answer_weight": 0.40, "faithfulness_weight": 0.25, "evidence_weight": 0.20},
        "evidence_qa_strict":  {"format_weight": 0.10, "answer_weight": 0.30, "faithfulness_weight": 0.30, "evidence_weight": 0.30},
        "evidence_qa_format":  {"format_weight": 0.35, "answer_weight": 0.30, "faithfulness_weight": 0.20, "evidence_weight": 0.15},
        "default":             {"format_weight": 0.15, "answer_weight": 0.40, "faithfulness_weight": 0.25, "evidence_weight": 0.20},
    }
    
    weights = weight_configs.get(data_source, weight_configs["default"])
    
    return compute_score(
        solution_str=solution_str,
        ground_truth=gt_str,
        prompt_str=prompt_str,
        question=question,
        **weights,
        language="auto",
        api_url=api_url,
        api_key=api_key,
        api_model=api_model,
        debug=False,
    )


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    
    # 模拟用户输入（含 Context 原文）
    test_prompt = """Instruction:
- Answer based ONLY on provided pages. No prior knowledge or fabrication.

Question:
Which "Slings and Arrows" actor hosted the 9th Gemini Awards?

Context:
Paul Gross
Paul Michael Gross, OC (born April 30, 1959) is a Canadian actor, producer, director, singer, and writer born in Calgary, Alberta.  He is known for his lead role as Constable Benton Fraser in the television series "Due South" as well as his 2008 war film "Passchendaele", which he wrote, produced, directed, and starred in.  He later found success with another Canadian TV series, "Slings and Arrows".

9th Gemini Awards
The 9th Gemini Awards was held in 1995 to honour achievements in Canadian television.  It was hosted by Paul Gross and Tina Keeper and was broadcast on CBC.

Susan Coyne
Susan Coyne is a Canadian writer and actress, best known as one of the co-creators and co-stars of the award-winning "Slings and Arrows", a TV series which ran 2003-06 about a Canadian Shakespearean theatre company."""

    ground_truth = "Paul Gross"

    # ---- 测试1：好的输出 ----
    test_good = """<think>
I need to find which Slings and Arrows actor hosted the 9th Gemini Awards.

First, let me check the 9th Gemini Awards. <evidence 1>The 9th Gemini Awards was held in 1995 to honour achievements in Canadian television</evidence>. And <evidence 2>It was hosted by Paul Gross and Tina Keeper and was broadcast on CBC</evidence>.

Now I need to check who is connected to Slings and Arrows. <evidence 3>He later found success with another Canadian TV series, "Slings and Arrows"</evidence>. This confirms Paul Gross was in Slings and Arrows.

Also <evidence 4>Susan Coyne is a Canadian writer and actress, best known as one of the co-creators and co-stars of the award-winning "Slings and Arrows"</evidence>, but she was not a host.
</think>
Paul Gross"""

    # ---- 测试2：坏的输出 ----
    test_bad = """<think>
Let me think about this. <evidence 1>Paul Gross won the Academy Award for Best Actor in 2010</evidence>. Also <evidence 2>The 9th Gemini Awards was a grand ceremony held at the Royal Opera House in London</evidence>.
</think>
Paul Gross"""

    # ---- 测试3：没有 evidence ----
    test_no_evidence = """<think>
I think the answer is Paul Gross because he was in Slings and Arrows.
</think>
Paul Gross"""

    # ---- 测试4：没有 think 标签 ----
    test_no_think = "Paul Gross"

    print("=" * 80)
    print(" 测试1：好的输出（evidence 来自原文）".center(70, '='))
    print("=" * 80)
    compute_score(test_good, ground_truth, prompt_str=test_prompt, debug=True)

    print("=" * 80)
    print(" 测试2：坏的输出（evidence 是捏造的）".center(70, '='))
    print("=" * 80)
    compute_score(test_bad, ground_truth, prompt_str=test_prompt, debug=True)

    print("=" * 80)
    print(" 测试3：无 evidence".center(70, '='))
    print("=" * 80)
    compute_score(test_no_evidence, ground_truth, prompt_str=test_prompt, debug=True)

    print("=" * 80)
    print(" 测试4：无 think 标签".center(70, '='))
    print("=" * 80)
    compute_score(test_no_think, ground_truth, prompt_str=test_prompt, debug=True)
    
    # ---- 测试5：模拟 verl 框架调用（修复后的流程）----
    print("=" * 80)
    print(" 测试5：模拟 verl 框架调用".center(70, '='))
    print("=" * 80)
    
    # 模拟 longcontext_qa_evidence.py 处理后的数据
    extra_info = {
        'question': 'Which "Slings and Arrows" actor hosted the 9th Gemini Awards?',
        'prompt_str': test_prompt,  # 完整 user message
        'full_answer': test_good,
    }
    
    result = default_compute_score(
        data_source="hotpotqa_evidence",
        solution_str=test_good,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    print(f"verl 调用结果: score={result['score']:.4f}")
    print(f"  format={result['format_score']:.4f}, answer={result['answer_score']:.4f}")
    print(f"  faithfulness={result['faithfulness_score']:.4f}, evidence={result['evidence_score']:.4f}")