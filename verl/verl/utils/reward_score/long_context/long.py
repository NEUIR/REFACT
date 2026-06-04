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
from typing import Dict, Tuple, Optional


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extract the final answer from the model's response string."""
    # Extract final answer using XML-style tags
    if "</think>" not in solution_str:
        print("[Error] No valid answer tags found")
        return None, solution_str 
    
    final_answer = solution_str.split("</think>")[-1].strip()
    return final_answer, solution_str


def parse_model_answer(answer_text: str) -> Optional[str]:
    """Parse the model's answer to extract the actual value."""
    if not answer_text:
        return None
    
    if isinstance(answer_text, dict) and "target" in answer_text:
        answer_text = answer_text["target"]
    
    answer_text = str(answer_text).strip()
    
    # Extract choice from various formats
    # Look for patterns like (a), (b), (c), (d) or A, B, C, D
    patterns = [
        r'\(([a-d])\)',  # Matches (a), (b), (c), (d)
        r'^([a-d])\)',   # Matches a), b), c), d) at start
        r'^([a-d])$',    # Matches single letter a, b, c, d
        r'^([A-D])$',    # Matches single letter A, B, C, D
        r'\(([A-D])\)',  # Matches (A), (B), (C), (D)
        r'^([A-D])\)',   # Matches A), B), C), D) at start
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    # Try to extract from common answer formats
    if "answer is" in answer_text.lower():
        # Extract text after "answer is"
        parts = re.split(r'answer is\s*', answer_text, flags=re.IGNORECASE)
        if len(parts) > 1:
            remaining = parts[1].strip()
            # Check if it starts with a choice
            for pattern in patterns:
                match = re.match(pattern, remaining, re.IGNORECASE)
                if match:
                    return match.group(1).lower()
    
    # If no pattern matches, try to find any occurrence of choice letters
    choice_match = re.search(r'[^a-z]([a-d])[^a-z]', answer_text.lower())
    if choice_match:
        return choice_match.group(1)
    
    # Last resort: check if the entire answer is just a letter
    if answer_text and len(answer_text) == 1 and answer_text.lower() in 'abcd':
        return answer_text.lower()
    
    return None


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
        try:
            pred_status = parse_model_answer(answer_text)
            gt_status = parse_model_answer(ground_truth)
            
            if pred_status:
                print(f"\n[Content Validation]")
                print(f"  Expected: {gt_status}")
                print(f"  Predicted: {pred_status}")

                # Convert both to lowercase for comparison
                if pred_status and gt_status:
                    pred_status = pred_status.lower()
                    gt_status = gt_status.lower()

                if pred_status == gt_status:
                    answer_score = 1
                    print("  Content validation: FULL MATCH")
                else:
                    answer_score = 0
                    print("  Content validation: MISMATCH")
            else:
                answer_score = 0
                print("Fail to parse answer")
        except Exception as e:
            print(f"Error during answer validation: {e}")
            answer_score = 0
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