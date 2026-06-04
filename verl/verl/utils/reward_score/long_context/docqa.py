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
from typing import Dict, Tuple, Optional, List


def normalize_answer(s):
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extract the final answer from the model's response string."""
    # Extract final answer using XML-style tags
    if "</think>" not in solution_str:
        print("[DocQA] No valid answer tags found")
        return None, solution_str 
    
    final_answer = solution_str.split("</think>")[-1].strip()
    return final_answer, solution_str


def extract_answer(text: str) -> str:
    """Extract answer from text by removing common prefixes."""
    if not text:
        return ""
    
    # Look for the word "answer" followed by "is" or ":" and extract everything after
    # This will match patterns like:
    # - "the answer is..."
    # - "Therefore, the answer is..."
    # - "I think the answer is..."
    # - "Answer: ..."
    match = re.search(r'\banswer\s*(?:is|:)\s*(.+)', text, flags=re.IGNORECASE)
    
    if match:
        # Extract everything after "answer is/:"
        answer = match.group(1).strip()
        # Remove trailing punctuation
        answer = answer.rstrip(".")
        return answer
    
    # If no "answer" pattern found, return empty string
    return ""


def parse_model_answer(answer_text: str) -> Optional[str]:
    """Parse the model's answer to extract the actual value."""
    if not answer_text:
        return None
    
    if isinstance(answer_text, dict) and "target" in answer_text:
        answer_text = answer_text["target"]
    
    # Extract the actual answer
    answer = extract_answer(str(answer_text))
    
    return answer


def get_tokens(s):
    """Tokenize a string into words."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_em(a_gold, a_pred):
    """Compute exact match score."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_sub_em(a_gold, a_pred):
    """Compute substring exact match score."""
    normalized_gold = normalize_answer(a_gold)
    normalized_pred = normalize_answer(a_pred)
    return int(normalized_gold in normalized_pred)


def compute_f1(a_gold, a_pred):
    """Compute F1 score between gold and predicted answers."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    
    if len(gold_toks) == 0 and len(pred_toks) == 0:
        return 1.0
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return 0.0
    
    common = set(gold_toks) & set(pred_toks)
    num_same = sum(min(gold_toks.count(t), pred_toks.count(t)) for t in common)
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_toks) if len(pred_toks) > 0 else 0.0
    recall = num_same / len(gold_toks) if len(gold_toks) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1


def calc_metrics(predictions: List[str], answers: List[str]) -> Dict[str, float]:
    """Calculate multiple metrics for a set of predictions and answers."""
    assert len(predictions) == len(answers)
    
    em_scores = []
    sub_em_scores = []
    f1_scores = []
    
    for pred, ans in zip(predictions, answers):
        if isinstance(ans, list):
            # Handle multiple valid answers
            em = max(compute_em(a, pred) for a in ans)
            sub_em = max(compute_sub_em(a, pred) for a in ans)
            f1 = max(compute_f1(a, pred) for a in ans)
        else:
            em = compute_em(ans, pred)
            sub_em = compute_sub_em(ans, pred)
            f1 = compute_f1(ans, pred)
        
        em_scores.append(em)
        sub_em_scores.append(sub_em)
        f1_scores.append(f1)
    
    return {
        'em': sum(em_scores) / len(em_scores) if len(em_scores) > 0 else 0.0,
        'sub_em': sum(sub_em_scores) / len(sub_em_scores) if len(sub_em_scores) > 0 else 0.0,
        'f1': sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0.0,
    }


def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 prompt_str: Optional[str] = None,
                 format_reward: float = 0.0,
                 answer_reward: float = 1.0):
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        prompt_str: Original prompt (not used in rule-based version)
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Dictionary containing score, accuracy, and predicted answer
    """
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    
    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")
    
    # Validate answer content
    answer_score = 0
    if answer_text:
        pred_status = parse_model_answer(answer_text)
        gt_status = parse_model_answer(ground_truth)
        
        if pred_status:
            print(f"\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")
            
            # Calculate metrics
            metrics = calc_metrics([pred_status], [gt_status])
            
            # Use sub_em as the primary metric (as per QwenLong-L1)
            metric = metrics['sub_em']
            answer_score = metric
            
            print(f"  EM Score: {metrics['em']}")
            print(f"  Sub-EM Score: {metrics['sub_em']}")
            print(f"  F1 Score: {metrics['f1']}")
            print(f"  Answer Score: {answer_score}")
        else:
            answer_score = 0.0
            print("Fail to parse answer")
    else:
        print("\n[Content Validation] Skipped due to format errors or missing answer")
    
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Answer: {answer_score}")
    print("="*80 + "\n")
    
    return {
        "score": answer_score,
        "acc": answer_score == 1.0,
        "pred": answer_text,
    } 