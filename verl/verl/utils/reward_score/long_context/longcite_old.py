# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
from typing import Dict, Optional, List, Tuple
from .gen_judge import score_with_llm_judge, score_citation_validity


# ============================================================
#  语言检测 & 归一化 & 分词
# ============================================================

def detect_language(text: str) -> str:
    if not text:
        return "en"
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(re.sub(r'\s', '', text))
    if total_chars == 0:
        return "en"
    return "zh" if chinese_count / total_chars > 0.3 else "en"


def normalize_answer(s, language="auto"):
    if not s:
        return ""
    if language == "auto":
        language = detect_language(s)
    s_clean = s.strip()
    if re.match(r'^[-+]?[\d,.]+$', s_clean) and any(c.isdigit() for c in s_clean):
        if re.match(r'^[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?$', s_clean):
            s_clean = s_clean.replace(',', '')
            return s_clean.rstrip('.')
        s = s.replace(',', ' ')

    def remove_articles(text):
        if language == "zh":
            return re.sub(r'[的了吗呢啊嘛]', ' ', text)
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        if language == "zh":
            all_punc = set('，。！？；：""''（）【】《》、·…—' + string.punctuation)
        else:
            all_punc = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in all_punc)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def tokenize(text: str, language="auto") -> List[str]:
    if not text:
        return []
    return re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', text.lower())


def get_tokens(s, language="auto"):
    if not s:
        return []
    return tokenize(normalize_answer(s, language))


# ============================================================
#  基础指标计算
# ============================================================

def compute_em(a_gold, a_pred, language="auto"):
    return int(normalize_answer(a_gold, language) == normalize_answer(a_pred, language))


def compute_sub_em(a_gold, a_pred, language="auto"):
    return int(normalize_answer(a_gold, language) in normalize_answer(a_pred, language))


def compute_exact_match(pred_text: str, ref_text: str, language="auto") -> bool:
    if not pred_text or not ref_text:
        return False
    return normalize_answer(pred_text, language) == normalize_answer(ref_text, language)


def compute_contains_match(pred_text: str, ref_text: str, language="auto") -> bool:
    if not pred_text or not ref_text:
        return False
    return normalize_answer(ref_text, language) in normalize_answer(pred_text, language)


def compute_f1(a_gold, a_pred, language="auto"):
    gold_toks = get_tokens(a_gold, language)
    pred_toks = get_tokens(a_pred, language)
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


def compute_recall(a_gold, a_pred, language="auto"):
    gold_toks = get_tokens(a_gold, language)
    pred_toks = get_tokens(a_pred, language)
    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        return 0.0
    common = set(gold_toks) & set(pred_toks)
    num_same = sum(min(gold_toks.count(t), pred_toks.count(t)) for t in common)
    return num_same / len(gold_toks)


def lcs_length(seq1: List[str], seq2: List[str]) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(pred_text: str, ref_text: str, language="auto") -> float:
    if not pred_text or not ref_text:
        return 0.0
    if language == "auto":
        language = detect_language(pred_text + " " + ref_text)
    pred_tokens = tokenize(pred_text, language)
    ref_tokens = tokenize(ref_text, language)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs_len = lcs_length(pred_tokens, ref_tokens)
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def compute_math_match(pred_text: str, ref_text: str) -> bool:
    def clean(s):
        if not s:
            return ""
        s = s.strip().replace(',', '')
        return s[:-1] if s.endswith('.') else s

    p, r = clean(pred_text), clean(ref_text)
    try:
        float(p); float(r)
    except ValueError:
        return False

    def split_num(s):
        return (s.split('.')[0], s.split('.')[1]) if '.' in s else (s, "")

    p_int, p_dec = split_num(p)
    r_int, r_dec = split_num(r)
    try:
        if int(float(p_int)) != int(float(r_int)):
            return False
        int_val = int(float(p_int))
    except (ValueError, OverflowError):
        return False
    k = len(str(abs(int_val))) if int_val != 0 else 0
    needed = 3 - k
    if needed <= 0:
        return True

    def get_prefix(s, n):
        return s[:n].ljust(n, '0')

    if k > 0:
        return get_prefix(p_dec, needed) == get_prefix(r_dec, needed)
    if not r_dec.strip('0'):
        return not p_dec.strip('0')
    start_idx = next(i for i, c in enumerate(r_dec) if c != '0')
    limit = start_idx + 3
    return get_prefix(p_dec, limit) == get_prefix(r_dec, limit)


# ============================================================
#  辅助函数
# ============================================================

def parse_ground_truth(ground_truth, method=None) -> List[str]:
    text_gt = str(ground_truth).strip()
    if method == 'orderedlist' and "\n" in text_gt:
        parsed_dict = parse_ordered_list(text_gt)
        if parsed_dict:
            return [parsed_dict[k] for k in sorted(parsed_dict.keys())]
    return [text_gt]


def parse_ordered_list(text: str) -> Dict[int, str]:
    if not text:
        return {}
    items = {}
    current_idx = -1
    current_content = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^(\d+)\.\s*(.*)', line)
        if match:
            if current_idx != -1:
                items[current_idx] = " ".join(current_content).strip()
            current_idx = int(match.group(1))
            current_content = [match.group(2)]
        elif current_idx != -1:
            current_content.append(line)
    if current_idx != -1:
        items[current_idx] = " ".join(current_content).strip()
    return items


def similar_in(thinking_process: str, meta: str) -> bool:
    if meta in thinking_process:
        return True
    return compute_recall(meta, thinking_process) > 0.8


def calculate_meta_score(solution_str: str, language: str, meta_info: List) -> float:
    thinking_process = solution_str.split('</think>')[0] if '</think>' in solution_str else solution_str
    thinking_process = normalize_answer(thinking_process, language)
    meta_info_normalized = []
    for meta in meta_info:
        if isinstance(meta, list):
            meta_info_normalized.append([normalize_answer(m, language) for m in meta])
        else:
            meta_info_normalized.append([normalize_answer(meta, language)])
    hit_num = sum(
        1 for meta_list in meta_info_normalized
        if any(similar_in(thinking_process, m) for m in meta_list)
    )
    return hit_num / len(meta_info_normalized)


# ============================================================
#  候选答案提取 (★ <answer> 标签为最高优先级)
# ============================================================

def extract_answer_tag_content(solution_str: str) -> Optional[str]:
    """提取 <answer>...</answer> 中的内容（最高优先级）"""
    match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_all_candidate_answers(solution_str: str, debug: bool = False) -> List[str]:
    """
    提取所有候选答案。
    ★ 最高优先级: <answer>...</answer> 标签内容，放在候选列表最前面。
    """
    if not solution_str:
        return []

    candidates = []

    # ★ 最高优先级: <answer> 标签
    answer_tag = extract_answer_tag_content(solution_str)
    if answer_tag:
        candidates.append(answer_tag)
        if debug:
            print(f"[Priority] <answer> tag found: '{answer_tag[:100]}'")

    original_solution_str = solution_str
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[-1].strip()

    patterns = [
        r'(?:the\s+)?answer\s+is[:\s]+([^\n.]+)',
        r'answer\s*:\s*([^\n]+)',
        r'ans\s*:\s*([^\n]+)',
        r'final\s+answer\s*:\s*([^\n]+)',
        r'final\s+answer\s+is[:\s]+([^\n.]+)',
        r'solution\s*:\s*([^\n]+)',
        r'result\s*:\s*([^\n]+)',
        r'therefore[,\s]+([^\n.]+)',
        r'so[,\s]+the\s+answer\s+is[:\s]+([^\n.]+)',
        r'in\s+conclusion[,\s]+([^\n.]+)',
        r'thus[,\s]+([^\n.]+)',
        r'Answer:\s*([^\n]+)',
        r'answer\s*(?:is|:)\s*([^\n]+)',
        r'Final answer:\s*([^\n]+)',
        r'ANSWER:\s*([^\n]+)',
        r'\*\*Final Answer\*\*:?\s*([^\n]+)',
        r'\*\*Answer\*\*:?\s*([^\n]+)',
        r'答案[:：]\s*([^\n]+)',
        r'回答[:：]\s*([^\n]+)',
        r'最终答案[:：]\s*([^\n]+)',
        r'\*\*答案\*\*[:：]?\s*([^\n]+)',
        r'\*\*最终答案\*\*[:：]?\s*([^\n]+)',
        r'(?:Therefore|Thus|So),?\s*(?:the\s+)?answer\s+is:?\s*([^\n]+)',
        r'Final\s+Answer:\s*([^\n]+)',
        r'\\boxed\{([^}]+)\}',
        r'\\boxed\{\\text\{([^}]+)\}\}',
    ]

    for pattern in patterns:
        for match in re.findall(pattern, solution_str, flags=re.IGNORECASE | re.DOTALL):
            if isinstance(match, tuple):
                match = match[0] if match else ""
            answer = re.sub(r'\s+', ' ', re.sub(r'[.。]+$', '', str(match).strip()))
            if answer and answer not in candidates:
                candidates.append(answer)

    # 兜底: </think> 后首句 和 全文末句
    if "</think>" in original_solution_str:
        after_think = original_solution_str.split("</think>")[-1].strip()
        if after_think:
            first_sentence = re.split(r'[\n]+', after_think)[0]
            if first_sentence and first_sentence not in candidates:
                candidates.append(first_sentence)
    if original_solution_str:
        last_sentence = re.split(r'[\n]+', original_solution_str)[-1]
        if last_sentence and last_sentence not in candidates:
            candidates.append(last_sentence)

    return candidates


def calculate_candidate_score(pred_answer, gt_answer, scoring_method, language="auto"):
    if scoring_method == "exact_match":
        return 1.0 if compute_exact_match(pred_answer, gt_answer, language) else 0.0
    elif scoring_method == "math":
        return 1.0 if compute_math_match(pred_answer, gt_answer) else 0.0
    elif scoring_method == "contains":
        return 1.0 if compute_contains_match(pred_answer, gt_answer, language) else 0.0
    elif scoring_method == "rouge_l":
        return compute_rouge_l(pred_answer, gt_answer, language)
    elif scoring_method == "f1":
        return compute_f1(gt_answer, pred_answer, language)
    elif scoring_method in ("recall", "fulltext_recall"):
        return compute_recall(gt_answer, pred_answer, language)
    elif scoring_method == "hybrid":
        if compute_exact_match(pred_answer, gt_answer, language):
            return 1.0
        elif compute_contains_match(pred_answer, gt_answer, language):
            return 0.8
        return compute_rouge_l(pred_answer, gt_answer, language)
    else:  # "balanced"
        if compute_em(gt_answer, pred_answer, language) == 1.0:
            return 1.0
        return compute_f1(gt_answer, pred_answer, language)


# ============================================================
#  ★ 标签提取工具函数
# ============================================================

def extract_think_content(solution_str: str) -> Optional[str]:
    """提取 <think>...</think> 中的内容"""
    match = re.search(r'<think>(.*?)</think>', solution_str, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_evidence_blocks(text: str) -> List[Tuple[int, str]]:
    """
    提取所有成对的 <evidence N>...</evidence> 块。
    开标签带序号，闭标签不带序号: <evidence 1>内容</evidence>
    返回 [(序号, 内容), ...]
    """
    pattern = r'<evidence\s+(\d+)>(.*?)</evidence>'
    matches = re.findall(pattern, text, re.DOTALL)
    results = []
    for idx_str, content in matches:
        content = content.strip()
        if content:
            results.append((int(idx_str), content))
    return results


# ============================================================
#  ★ 奖励 1: 格式奖励 (严格版)
# ============================================================

def compute_format_reward(solution_str: str, debug: bool = False) -> float:
    """
    严格格式检查：
      1. <think>...</think>  恰好出现 1 次
      2. <answer>...</answer> 恰好出现 1 次
      3. <think> 内部至少有 1 对序号匹配的 <evidence N>...</evidence N>
    全部满足 → 1.0，否则 → 0.0
    """
    # 检查 <think> 恰好出现 1 次
    think_count = len(re.findall(r'<think>', solution_str))
    think_close_count = len(re.findall(r'</think>', solution_str))
    has_single_think = (think_count == 1 and think_close_count == 1)

    # 检查 <answer> 恰好出现 1 次
    answer_count = len(re.findall(r'<answer>', solution_str))
    answer_close_count = len(re.findall(r'</answer>', solution_str))
    has_single_answer = (answer_count == 1 and answer_close_count == 1)

    # 检查 evidence 成对出现且在 <think> 内
    has_valid_evidence = False
    think_content = extract_think_content(solution_str)
    if think_content:
        evidence_blocks = extract_evidence_blocks(think_content)
        # 开标签: <evidence N>  闭标签: </evidence> (无序号)
        open_tags = re.findall(r'<evidence\s+\d+>', think_content)
        close_tags = re.findall(r'</evidence>', think_content)
        # 开闭标签数量必须一致，且至少有 1 对
        tags_paired = (len(open_tags) == len(close_tags) and len(open_tags) > 0)
        has_valid_evidence = len(evidence_blocks) > 0 and tags_paired

    result = 1.0 if (has_single_think and has_single_answer and has_valid_evidence) else 0.0

    if debug:
        print(f"  [Format] think={think_count}/{think_close_count} "
              f"answer={answer_count}/{answer_close_count} "
              f"evidence_valid={has_valid_evidence} → {result}")

    return result


# ============================================================
#  ★ 奖励 2: 正确性奖励 (<answer> 最高优先级)
# ============================================================

def compute_correctness_reward(solution_str: str, ground_truth: str,
                                language: str = "auto", debug: bool = False) -> float:
    """
    ★ 最高优先级从 <answer> 提取，若无则退回候选提取。
    与标准答案计算 F1。
    """
    # 最高优先级: <answer> 标签
    answer_content = extract_answer_tag_content(solution_str)
    if answer_content:
        f1 = compute_f1(ground_truth, answer_content, language)
        if debug:
            print(f"  [Correctness] <answer> tag: '{answer_content[:80]}' → F1={f1:.4f}")
        return f1

    # 退回: 候选答案提取
    if debug:
        print("  [Correctness] No <answer> tag, fallback to candidate extraction")
    candidates = extract_all_candidate_answers(solution_str, debug=debug)
    gt_answers = parse_ground_truth(ground_truth)
    best = 0.0
    for cand in candidates:
        for gt in gt_answers:
            best = max(best, compute_f1(gt, cand, language))
    return best


# ============================================================
#  ★ 奖励 3: 一致性奖励 (滑动窗口序列匹配)
# ============================================================

def _sliding_window_rouge_l(evidence_tokens: List[str],
                             source_tokens: List[str],
                             window_ratio: float = 2.0) -> float:
    """
    用滑动窗口在 source_tokens 上找与 evidence_tokens 最匹配的片段。
    返回最高的 precision = LCS长度 / evidence长度。
    即: evidence中有多少内容（按顺序）能在原文某个片段中找到。
 
    为什么用 precision 而不是 F1:
      - evidence = "The Power Bank has a 10000mAh battery capacity"  (8 tokens)
      - window_ratio=2.0 → 窗口=16 tokens
      - 完美匹配时: LCS=8, precision=8/8=1.0, recall=8/16=0.5, F1=0.667
      - F1的recall部分被窗口大小拖累, 但我们不关心窗口中没被覆盖的部分
      - 我们只关心: evidence这句话是不是来自原文 → precision
    """
    ev_len = len(evidence_tokens)
    if ev_len == 0 or not source_tokens:
        return 0.0
 
    window_size = max(ev_len, int(ev_len * window_ratio))
    step = max(1, window_size // 4)
    best_score = 0.0
 
    for start in range(0, max(1, len(source_tokens) - window_size + 1), step):
        end = min(start + window_size, len(source_tokens))
        window_tokens = source_tokens[start:end]
        lcs_len = lcs_length(evidence_tokens, window_tokens)
        if lcs_len == 0:
            continue
        # ★ 改动: 用 precision 替代 F1
        precision = lcs_len / ev_len
        best_score = max(best_score, precision)
        # 已经找到完美匹配，提前退出
        if best_score >= 1.0:
            break
 
    return best_score


def compute_consistency_reward(solution_str: str, source_text: str,
                                language: str = "auto",
                                # ★ 移除了 similarity_threshold 参数, 不再需要
                                debug: bool = False) -> float:
    """
    检查 <think> 中每个 <evidence N>...</evidence> 的内容
    能否在原文中找到对应片段。
 
    对每个evidence计算: precision = LCS(evidence, 最佳窗口) / len(evidence)
    返回所有evidence的precision平均值作为一致性奖励。
 
    - 完全来自原文 → 每个接近1.0 → 平均接近1.0
    - 混入编造引用 → 编造的接近0 → 拉低平均分
    - 全部编造     → 每个接近0   → 平均接近0
    """
    think_content = extract_think_content(solution_str)
    if not think_content:
        if debug:
            print("  [Consistency] No <think> content")
        return 0.0
 
    evidence_blocks = extract_evidence_blocks(think_content)
    if not evidence_blocks:
        if debug:
            print("  [Consistency] No valid <evidence> blocks")
        return 0.0
 
    if not source_text:
        if debug:
            print("  [Consistency] No source_text provided")
        return 0.0
 
    source_tokens = tokenize(source_text.lower(), language)
    if not source_tokens:
        return 0.0
 
    # ★ 改动: 收集每个evidence的分数, 最后算平均
    scores = []
    for idx, ev_content in evidence_blocks:
        ev_tokens = tokenize(ev_content.lower(), language)
        if not ev_tokens:
            scores.append(0.0)
            continue
 
        score = _sliding_window_rouge_l(ev_tokens, source_tokens)
        scores.append(score)
 
        if debug:
            print(f"  [Consistency] evidence {idx}: "
                  f"len={len(ev_tokens)}, similarity={score:.4f}")
 
    # ★ 改动: 返回平均分, 而非命中比例
    avg_score = sum(scores) / len(scores) if scores else 0.0
 
    if debug:
        print(f"  [Consistency] 平均相似度: {avg_score:.4f}")
 
    return avg_score


# ============================================================
#  ★ 奖励 4: 引用有效性奖励
# ============================================================

def compute_citation_validity_reward(solution_str: str, ground_truth: str,
                                      question: str,
                                      language: str = "auto",
                                      judge_config_key: str = "citation_validity",
                                      debug: bool = False) -> float:
    """
    将所有 evidence 拼接后 + question 发给 API，
    让模型仅根据引用回答，返回回答与标准答案的 F1。
    """
    think_content = extract_think_content(solution_str)
    if not think_content:
        if debug:
            print("  [CitationValidity] No <think> content")
        return 0.0

    evidence_blocks = extract_evidence_blocks(think_content)
    if not evidence_blocks:
        if debug:
            print("  [CitationValidity] No valid <evidence> blocks")
        return 0.0

    evidence_texts = [f"[Evidence {idx}]: {content}"
                      for idx, content in sorted(evidence_blocks, key=lambda x: x[0])]
    all_evidence = "\n\n".join(evidence_texts)

    if debug:
        print(f"  [CitationValidity] {len(evidence_blocks)} evidence blocks, "
              f"total {len(all_evidence)} chars")

    try:
        api_answer = score_citation_validity(
            question=question, evidence_text=all_evidence,
            judge_config_key=judge_config_key
        )
    except Exception as e:
        print(f"⚠️ [CitationValidity] API call failed: {type(e).__name__}: {e}")
        return 0.0

    if not api_answer:
        if debug:
            print("  [CitationValidity] API returned empty")
        return 0.0

    f1 = compute_f1(ground_truth, api_answer, language)
    if debug:
        print(f"  [CitationValidity] API answer='{api_answer[:80]}' → F1={f1:.4f}")
    return f1


# ============================================================
#  ★ 综合四种奖励
# ============================================================

def compute_score_with_evidence(
    solution_str: str,
    ground_truth,
    question: str = "",
    source_text: str = "",
    format_weight: float = 0.1,
    correctness_weight: float = 0.4,
    consistency_weight: float = 0.2,
    citation_validity_weight: float = 0.3,
    # ★ similarity_threshold 可以保留参数但不再传给 consistency
    similarity_threshold: float = 0.6,
    language: str = "auto",
    debug: bool = False,
) -> Dict:
    if language == "auto":
        language = detect_language(solution_str + str(ground_truth))

    gt_answer = str(ground_truth.get('answer', ground_truth)).strip() \
        if isinstance(ground_truth, dict) else str(ground_truth).strip()

    # 提取信息
    answer_from_tag = extract_answer_tag_content(solution_str)
    candidates = []
    if not answer_from_tag:
        candidates = extract_all_candidate_answers(solution_str)

    think_content = extract_think_content(solution_str)
    evidence_blocks = []
    if think_content:
        evidence_blocks = extract_evidence_blocks(think_content)

    # 计算奖励
    r_format = compute_format_reward(solution_str, debug=False)
    r_correct = compute_correctness_reward(solution_str, gt_answer, language, debug=False)

    r_consist = 0.0
    if source_text:
        # ★ 改动: 不再传 similarity_threshold
        r_consist = compute_consistency_reward(
            solution_str, source_text, language, debug=False)

    r_cite = 0.0
    if question:
        r_cite = compute_citation_validity_reward(
            solution_str, gt_answer, question, language, debug=False)

    total = (format_weight * r_format
             + correctness_weight * r_correct
             + consistency_weight * r_consist
             + citation_validity_weight * r_cite)

    pred_answer = answer_from_tag
    if pred_answer is None:
        pred_answer = candidates[0] if candidates else None

    # 一次性打印
    ev_str = "(none)"
    if evidence_blocks:
        parts = []
        for idx, content in evidence_blocks:
            t = content[:300]
            s = f"...({len(content)})" if len(content) > 300 else ""
            parts.append(f"[{idx}]{t}{s}")
        ev_str = " | ".join(parts)

    log = (
        f"\n{'='*80}\n"
        f"[Q]: {question[:300] if question else 'N/A'}\n"
        f"[GT]: {gt_answer}\n"
        f"[OUTPUT]: {solution_str}\n"
        f"[PRED]: {pred_answer if pred_answer else 'NONE'}\n"
        f"[EVIDENCE]: {ev_str}\n"
        f"[REWARD] fmt={r_format:.2f} cor={r_correct:.2f} con={r_consist:.2f} cite={r_cite:.2f} => {total:.4f}\n"
        f"{'='*80}"
    )
    print(log, flush=True)

    return {
        "score": total,
        "acc": r_correct == 1.0,
        "pred": pred_answer,
        "format_reward": r_format,
        "correctness_reward": r_correct,
        "consistency_reward": r_consist,
        "citation_validity_reward": r_cite,
    }

# ============================================================
#  统一入口 compute_score (兼容原有 + 新增 evidence_based)
# ============================================================

def compute_score(solution_str: str,
                 ground_truth,
                 prompt_str: Optional[str] = None,
                 format_reward: float = 0.0,
                 answer_reward: float = 1.0,
                 meta_reward: float = 0.0,
                 scoring_method: str = "balanced",
                 language: str = "auto",
                 debug: bool = False,
                 # evidence_based 专用参数
                 question: str = "",
                 source_text: str = "",
                 format_weight: float = 0.1,
                 correctness_weight: float = 0.4,
                 consistency_weight: float = 0.2,
                 citation_validity_weight: float = 0.3,
                 similarity_threshold: float = 0.6,
                 ):

    if language == "auto":
        language = detect_language(solution_str + str(ground_truth))

    # ★ evidence_based 评分
    if scoring_method == "evidence_based":
        return compute_score_with_evidence(
            solution_str=solution_str, ground_truth=ground_truth,
            question=question, source_text=source_text,
            format_weight=format_weight,
            correctness_weight=correctness_weight,
            consistency_weight=consistency_weight,
            citation_validity_weight=citation_validity_weight,
            similarity_threshold=similarity_threshold,
            language=language, debug=debug,
        )

    # ---- 以下为原有逻辑 ----

    if scoring_method == "llm_judge":
        pred_answer = solution_str.split("</think>")[-1].strip() \
            if "</think>" in solution_str else solution_str
        gt_answers = parse_ground_truth(ground_truth)
        score = score_with_llm_judge(pred_answer, gt_answers[0], "default")
        return {"score": score, "acc": score == 1.0, "pred": pred_answer}

    if scoring_method == "orderedlist":
        return compute_score_ordered_list(solution_str, ground_truth, language, debug)

    candidate_answers = extract_all_candidate_answers(solution_str, debug=debug)

    if scoring_method == "fulltext_recall":
        full_text = solution_str.split("</think>")[-1].strip() \
            if "</think>" in solution_str else solution_str.strip()
        if full_text and full_text not in candidate_answers:
            candidate_answers.append(full_text)

    if 'meta' in scoring_method:
        gt_answers = parse_ground_truth(ground_truth['answer'])
    else:
        gt_answers = parse_ground_truth(ground_truth)

    pred_answer = None
    best_score = -1.0

    if 'meta' in scoring_method:
        meta_score = calculate_meta_score(solution_str, language, ground_truth['meta'])

    if candidate_answers and gt_answers:
        for candidate in candidate_answers:
            scores = []
            for gt_answer in gt_answers:
                if scoring_method == "metarecall":
                    a_score = calculate_candidate_score(candidate, gt_answer, 'recall', language)
                    s = meta_reward * meta_score + (1 - meta_reward) * a_score
                else:
                    s = calculate_candidate_score(candidate, gt_answer, scoring_method, language)
                scores.append(s)
            avg = sum(scores) / len(scores)
            if avg > best_score:
                best_score = avg
                pred_answer = candidate
        print(f"\n  Best Answer Selected: {pred_answer} (Score: {best_score:.4f})")
    else:
        best_score = 0.0

    if pred_answer is None:
        best_score = 0.0

    return {"score": best_score, "acc": best_score == 1.0, "pred": pred_answer}


def compute_score_ordered_list(solution_str, ground_truth, language="auto", debug=False):
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[-1].strip()
    preds_dict = parse_ordered_list(solution_str)
    gt_answers = parse_ground_truth(ground_truth, method='orderedlist')
    if not gt_answers:
        return {"score": 0.0, "acc": False, "pred": solution_str}
    total = sum(
        compute_recall(gt, preds_dict[i + 1], language="auto")
        for i, gt in enumerate(gt_answers) if (i + 1) in preds_dict
    )
    final = total / len(gt_answers)
    extracted = "\n".join([preds_dict[k] for k in sorted(preds_dict.keys())])
    return {"score": final, "acc": final == 1.0, "pred": extracted}


# ============================================================
#  测试
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" Test: Evidence-Based Reward ".center(80, '='))

    test_solution = """<think>
根据文档内容，我需要找到关于电池容量的信息。

<evidence 1>The Power Bank has a 10000mAh battery capacity.</evidence 1>

<evidence 2>It supports fast charging with 18W output.</evidence 2>

综合以上信息，电池容量是 10000mAh。
</think>

<answer>10000mAh</answer>"""

    test_source = ("The Power Bank has a 10000mAh battery capacity. "
                   "It supports fast charging with 18W output power. "
                   "The device weighs 200g.")

    result = compute_score(
        solution_str=test_solution,
        ground_truth="10000mAh",
        scoring_method="evidence_based",
        question="What is the battery capacity of the Power Bank?",
        source_text=test_source,
        debug=True,
    )
    print(f"Result: {result}")

    # 测试格式不合格的情况
    print("\n--- Test: Bad Format (double think) ---")
    bad_format = "<think>aaa</think><think>bbb</think><answer>10000mAh</answer>"
    fmt = compute_format_reward(bad_format, debug=True)
    print(f"Format reward: {fmt}")

    print("\n--- Test: Bad Format (mismatched evidence) ---")
    bad_ev = "<think><evidence 1>xxx</think><answer>ok</answer>"
    fmt2 = compute_format_reward(bad_ev, debug=True)
    print(f"Format reward (no closing </evidence>): {fmt2}")

    print("\n--- Test: Good Format ---")
    good = "<think><evidence 1>xxx</evidence><evidence 2>yyy</evidence></think><answer>ok</answer>"
    fmt3 = compute_format_reward(good, debug=True)
    print(f"Format reward (correct): {fmt3}")