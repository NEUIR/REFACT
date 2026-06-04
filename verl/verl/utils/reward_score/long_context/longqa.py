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
#  ★ 奖励 3: 一致性奖励 (SequenceMatcher 子串匹配)
# ============================================================

from difflib import SequenceMatcher


def _find_best_match_ratio(evidence: str, source: str) -> float:
    """
    使用 SequenceMatcher 在长原文中找到与 evidence 最匹配的片段。
    
    核心思路：
      1. 先对原文做粗粒度分块（按句子/段落），快速定位候选区域
      2. 对候选区域用 SequenceMatcher 精确计算相似度
      3. 返回最高相似度
    
    SequenceMatcher.ratio() 返回的是：
      2.0 * matching_chars / total_chars
    即两个序列中匹配字符数占总字符数的比例，完全匹配=1.0。
    
    对于子串场景（evidence 是原文的一部分），即使原文很长，
    只要在对应窗口内匹配到，ratio 就会很高。
    """
    evidence = evidence.strip().lower()
    source = source.lower()
    
    if not evidence or not source:
        return 0.0
    
    ev_len = len(evidence)
    
    # 如果 evidence 很短（<50字符），直接做子串查找
    if ev_len < 50:
        if evidence in source:
            return 1.0
        # 短文本用 SequenceMatcher 在全文找最长匹配
        sm = SequenceMatcher(None, evidence, source, autojunk=False)
        # find_longest_match 找到最长连续匹配块
        match = sm.find_longest_match(0, len(evidence), 0, len(source))
        if match.size == 0:
            return 0.0
        return match.size / ev_len
    
    # 对于较长的 evidence，用滑动窗口 + SequenceMatcher
    # 窗口大小略大于 evidence，确保能完整包含
    window_size = max(ev_len, int(ev_len * 1.5))
    step = max(1, ev_len // 3)  # 步长为 evidence 长度的 1/3
    
    best_ratio = 0.0
    
    for start in range(0, max(1, len(source) - ev_len + 1), step):
        end = min(start + window_size, len(source))
        window = source[start:end]
        
        sm = SequenceMatcher(None, evidence, window, autojunk=False)
        ratio = sm.ratio()
        
        if ratio > best_ratio:
            best_ratio = ratio
        
        # 提前退出
        if best_ratio > 0.95:
            break
    
    return best_ratio


def _find_best_match_ratio_fast(evidence: str, source: str) -> float:
    """
    快速版本：先尝试直接子串匹配，再退回 SequenceMatcher。
    适用于模型直接从原文复制引用的场景（最常见）。
    """
    evidence_clean = evidence.strip().lower()
    source_lower = source.lower()
    
    if not evidence_clean or not source_lower:
        return 0.0
    
    # 快速路径1: 完全子串匹配 → 1.0
    if evidence_clean in source_lower:
        return 1.0
    
    # 快速路径2: 去掉首尾空白和标点后再试
    import string
    ev_stripped = evidence_clean.strip(string.punctuation + string.whitespace)
    if ev_stripped and ev_stripped in source_lower:
        return 0.95
    
    # 快速路径3: 按行/句拆分 evidence，检查每句是否在原文中
    # 适用于多句拼接引用的场景
    ev_sentences = re.split(r'[。.!!\?\?；;\n]+', evidence_clean)
    ev_sentences = [s.strip() for s in ev_sentences if len(s.strip()) > 5]
    if ev_sentences:
        matched_sentences = sum(1 for s in ev_sentences if s in source_lower)
        sentence_ratio = matched_sentences / len(ev_sentences)
        if sentence_ratio >= 0.8:
            return sentence_ratio
    
    # 慢速路径: SequenceMatcher 滑动窗口
    return _find_best_match_ratio(evidence_clean, source_lower)


def compute_consistency_reward(solution_str: str, source_text: str,
                                language: str = "auto",
                                similarity_threshold: float = 0.7,
                                debug: bool = False) -> float:
    """
    检查 <think> 中每个 <evidence N>...</evidence> 能否在原文中找到对应片段。
    
    匹配策略（由快到慢）:
      1. 直接子串匹配 → 1.0 (最快，覆盖逐字引用)
      2. 去标点后子串匹配 → 0.95
      3. 按句拆分匹配 → 句子匹配比例 (适用于多句引用)
      4. SequenceMatcher 滑动窗口 → ratio (适用于有轻微改写的引用)
    
    Args:
        solution_str: 模型完整输出
        source_text: 原始文档全文
        language: 语言
        similarity_threshold: 相似度阈值，默认 0.7
        debug: 调试模式
    
    Returns:
        匹配的 evidence 比例 (0.0 ~ 1.0)
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

    hit_count = 0
    total_score = 0.0

    for idx, ev_content in evidence_blocks:
        if not ev_content.strip():
            continue

        score = _find_best_match_ratio_fast(ev_content, source_text)
        is_match = score >= similarity_threshold

        if is_match:
            hit_count += 1
        total_score += score

        if debug:
            preview = ev_content[:60].replace('\n', ' ')
            print(f"  [Consistency] evidence {idx}: "
                  f"'{preview}...' → score={score:.4f}, match={is_match}")

    n = len(evidence_blocks)
    # 返回匹配比例
    match_ratio = hit_count / n if n > 0 else 0.0

    if debug:
        avg_score = total_score / n if n > 0 else 0.0
        print(f"  [Consistency] {hit_count}/{n} matched, "
              f"avg_score={avg_score:.4f} → ratio={match_ratio:.4f}")

    return match_ratio


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

    api_answer = score_citation_validity(
        question=question, evidence_text=all_evidence,
        judge_config_key=judge_config_key
    )

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
    similarity_threshold: float = 0.6,
    language: str = "auto",
    debug: bool = False,
) -> Dict:
    if language == "auto":
        language = detect_language(solution_str + str(ground_truth))

    gt_answer = str(ground_truth.get('answer', ground_truth)).strip() \
        if isinstance(ground_truth, dict) else str(ground_truth).strip()

    print("\n" + "=" * 80)
    print(" Evidence-Based Reward Evaluation ".center(80, '='))

    # 1. 格式奖励
    r_format = compute_format_reward(solution_str, debug=debug)
    print(f"  [1] Format Reward:            {r_format:.4f}")

    # 2. 正确性奖励
    r_correct = compute_correctness_reward(solution_str, gt_answer, language, debug=debug)
    print(f"  [2] Correctness Reward (F1):  {r_correct:.4f}")

    # 3. 一致性奖励
    r_consist = 0.0
    if source_text:
        r_consist = compute_consistency_reward(
            solution_str, source_text, language,
            similarity_threshold=similarity_threshold, debug=debug)
    elif debug:
        print("  [3] Consistency Reward: SKIPPED (no source_text)")
    print(f"  [3] Consistency Reward:       {r_consist:.4f}")

    # 4. 引用有效性
    r_cite = 0.0
    if question:
        r_cite = compute_citation_validity_reward(
            solution_str, gt_answer, question, language, debug=debug)
    elif debug:
        print("  [4] Citation Validity: SKIPPED (no question)")
    print(f"  [4] Citation Validity (F1):   {r_cite:.4f}")

    # 加权总分
    total = (format_weight * r_format
             + correctness_weight * r_correct
             + consistency_weight * r_consist
             + citation_validity_weight * r_cite)

    print(f"\n  Total = {format_weight}×{r_format:.3f} + {correctness_weight}×{r_correct:.3f}"
          f" + {consistency_weight}×{r_consist:.3f} + {citation_validity_weight}×{r_cite:.3f}"
          f" = {total:.4f}")
    print("=" * 80 + "\n")

    pred_answer = extract_answer_tag_content(solution_str)
    if pred_answer is None:
        cands = extract_all_candidate_answers(solution_str)
        pred_answer = cands[0] if cands else None

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