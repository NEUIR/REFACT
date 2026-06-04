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
from typing import Optional


def extract_non_reasoning_content(
    text: str,
    think_start_token: str = '<think>',
    think_end_token: str = '</think>',
) -> str:
    if think_start_token not in text and think_end_token in text:
        return text.split(think_end_token)[-1].strip()

    # Original behavior for complete tag pairs
    reasoning_regex = re.compile(rf'{think_start_token}(.*?){think_end_token}',
                                 re.DOTALL)
    non_reasoning_content = reasoning_regex.sub('', text).strip()
    return non_reasoning_content


def ruler_cwe_score(predictions, gold):
    """Simplified scoring similar to RulerCweEvaluator.
    
    Args:
        predictions: List of prediction strings
        gold: List of reference answers (each can be a list or single string)
    
    Returns:
        Dict with score
    """
    # score = sum([
    #     sum([1.0 if r.lower() in pred.lower() else 0.0
    #          for r in (ref if isinstance(ref, list) else [ref])])
    #     / len(ref if isinstance(ref, list) else [ref])
    #     for pred, ref in zip(predictions, gold)
    # ]) / len(predictions)

    score = sum([1.0 if ref.lower() in predictions.lower() else 0.0 for ref in gold]) / len(gold)
    return score

def ruler_cwe_score_any(predictions, gold):
    """Simplified scoring similar to RulerCweEvaluator.
    
    Args:
        predictions: List of prediction strings
        gold: List of reference answers (each can be a list or single string)
    
    Returns:
        Dict with score
    """
    # score = sum([
    #     sum([1.0 if r.lower() in pred.lower() else 0.0
    #          for r in (ref if isinstance(ref, list) else [ref])])
    #     / len(ref if isinstance(ref, list) else [ref])
    #     for pred, ref in zip(predictions, gold)
    # ]) / len(predictions)

    score = 1.0 if any(ref.lower() in predictions.lower() for ref in gold) else 0.0
    return score

def compute_score(solution_str: str, 
                 ground_truth,
                 prompt_str: Optional[str] = None,
                 format_reward: float = 0.0,
                 answer_reward: float = 1.0,
                 scoring_method: str = "balanced",
                 language: str = "auto",
                 debug: bool = False):
    """Simplified compute score function using extract_non_reasoning_content and contains matching.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary or string containing ground truth answer
        prompt_str: Original prompt (not used)
        format_reward: Points awarded/deducted for format correctness (not used)
        answer_reward: Points awarded/deducted for answer correctness (not used)
        scoring_method: Evaluation method (not used in simplified version)
        language: Language for processing (not used in simplified version)
        debug: Enable debug output
        
    Returns:
        Dictionary containing score, accuracy, and predicted answer
    """
    if debug:
        print("\n" + "="*80)
        print(" Simplified Ruler Scoring ".center(80, '='))
        print(f"\nOriginal Response:\n{solution_str}")
    
    # Step 1: Extract non-reasoning content
    processed_text = extract_non_reasoning_content(solution_str)
    
    if debug:
        print(f"\nProcessed Text (after removing thinking tags):\n{processed_text}")

    gt_answer = ground_truth
    print(f"gt_answer: {gt_answer}")
    
    if scoring_method == "any":
        score = ruler_cwe_score_any(processed_text, gt_answer)
    else:
        score = ruler_cwe_score(processed_text, gt_answer)
    
    if debug:
        print(f"\nFinal Score: {score}")
        print("="*80 + "\n")

    # processed_text = "\n\n--------------------------------\n\n".join([processed_text, gt_answer])
    
    return {
        "score": score,
        "acc": score == 1.0,
        "pred": processed_text,
    }
