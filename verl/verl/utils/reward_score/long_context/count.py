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
from typing import Dict, Optional


def extract_answer_number(solution_str: str) -> Optional[int]:
    """Extract number from <answer></answer> tags."""
    if not solution_str:
        return None
    
    # Extract content between <answer> and </answer> tags
    match = re.search(r'<answer>(.*?)</answer>', solution_str, flags=re.DOTALL | re.IGNORECASE)
    
    if not match:
        print("[Error] No valid <answer></answer> tags found")
        return None
    
    answer_content = match.group(1).strip()
    
    # Extract the first number from the answer content
    number_match = re.search(r'(\d+)', answer_content)
    
    if number_match:
        return int(number_match.group(1))
    
    print(f"[Error] No number found in answer content: {answer_content}")
    return None


def parse_ground_truth(ground_truth) -> Optional[int]:
    """Parse ground truth to extract the expected number."""
    if isinstance(ground_truth, dict):
        # If ground_truth is a dict, try to get the target value
        if "target" in ground_truth:
            target = ground_truth["target"]
        elif "answer" in ground_truth:
            target = ground_truth["answer"]
        else:
            # Use the first value if no standard key
            target = next(iter(ground_truth.values()))
    else:
        target = ground_truth
    
    # Convert to int if it's a string number
    if isinstance(target, str):
        number_match = re.search(r'(\d+)', target)
        if number_match:
            return int(number_match.group(1))
        else:
            print(f"[Error] No number found in ground truth string: {target}")
            return None
    elif isinstance(target, (int, float)):
        return int(target)
    
    print(f"[Error] Unable to parse ground truth: {target} (type: {type(target)})")
    return None


def extract_thinking_steps(solution_str: str) -> dict:
    """Extract and analyze the thinking process with better language support."""
    thinking_analysis = {
        "has_thinking": False,
        "mentions_passage": False,
        "shows_counting": False,
        "step_by_step": False,
        "mentions_different": False,
        "uses_numbering": False,
    }
    
    # Extract thinking content
    if "<think>" in solution_str and "</think>" in solution_str:
        thinking_analysis["has_thinking"] = True
        thinking_content = solution_str.split("<think>")[1].split("</think>")[0].lower()
        
        # Check if mentions "passage" or paragraph patterns
        passage_patterns = [
            "passage", "paragraph", "section", "part",
            "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
        ]
        if any(pattern in thinking_content for pattern in passage_patterns):
            thinking_analysis["mentions_passage"] = True
            
        # Check for numbering/listing patterns (both English and Chinese)
        numbering_patterns = [
            # English patterns
            r"first\s+(?:passage|paragraph|section|part)",
            r"second\s+(?:passage|paragraph|section|part)", 
            r"third\s+(?:passage|paragraph|section|part)",
            r"fourth\s+(?:passage|paragraph|section|part)",
            r"p1", r"p2", r"p3", r"p4", r"p5",
            r"\b1\.", r"\b2\.", r"\b3\.", r"\b4\.", r"\b5\.",
            r"\b1\)", r"\b2\)", r"\b3\)", r"\b4\)", r"\b5\)",
        ]
        
        import re
        numbering_found = any(re.search(pattern, thinking_content) for pattern in numbering_patterns)
        if numbering_found:
            thinking_analysis["uses_numbering"] = True
            
        # Check if shows counting behavior (multilingual)
        counting_words = [
            # English counting words
            "count", "counting", "total", "altogether", "sum", "add up",
            "how many", "number of", "identify", "list", "enumerate",
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        ]
        if any(word in thinking_content for word in counting_words):
            thinking_analysis["shows_counting"] = True
            
        # Check for step-by-step reasoning (multilingual)
        step_indicators = [
            # English step indicators
            "first", "second", "third", "fourth", "fifth",
            "then", "next", "finally", "lastly", "step", "let me",
            "1.", "2.", "3.", "4.", "5.", "step 1", "step 2",
        ]
        if any(indicator in thinking_content for indicator in step_indicators):
            thinking_analysis["step_by_step"] = True
            
        # Check if mentions "different" or "distinct" (multilingual)
        different_words = [
            # English
            "different", "distinct", "unique", "separate",
            "not the same", "same", "repeat", "repeated"
        ]
        if any(word in thinking_content for word in different_words):
            thinking_analysis["mentions_different"] = True
    
    return thinking_analysis


def compute_thinking_reward(thinking_analysis: dict) -> float:
    """Compute intermediate reward for thinking process quality with refined scoring."""
    thinking_reward = 0.0
    
    if thinking_analysis["has_thinking"]:
        thinking_reward += 0.01  # Basic thinking format
        
        if thinking_analysis["mentions_passage"]:
            thinking_reward += 0.02  # Understands the task is about passages
            
        if thinking_analysis["uses_numbering"]:
            thinking_reward += 0.02  # Uses systematic numbering (P1, P2, etc.)
            
        if thinking_analysis["shows_counting"]:
            thinking_reward += 0.05  # Shows counting behavior
            
        if thinking_analysis["step_by_step"]:
            thinking_reward += 0.02  # Structured reasoning
            
        if thinking_analysis["mentions_different"]:
            thinking_reward += 0.05  # Understands need to distinguish
            
    return thinking_reward


def compute_score(solution_str: str, 
                 ground_truth,
                 prompt_str: Optional[str] = None,
                 format_reward: float = 0.0,
                 answer_reward: float = 1.0):
    """Computes score for counting task with progressive rewards and thinking analysis.
    
    Args:
        solution_str: Raw model response string containing <think></think> and <answer></answer> tags
        ground_truth: Dictionary containing ground truth data
        prompt_str: Original prompt (not used)
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Dictionary containing score, accuracy, predicted answer, and thinking analysis
    """
    print("\n" + "="*80)
    print(" Processing Counting Task with Thinking Analysis ".center(80, '='))
    
    # Analyze thinking process
    thinking_analysis = extract_thinking_steps(solution_str)
    thinking_reward = compute_thinking_reward(thinking_analysis)
    
    print(f"\n[Thinking Analysis]")
    print(f"  Has structured thinking: {thinking_analysis['has_thinking']}")
    print(f"  Mentions passages: {thinking_analysis['mentions_passage']}")
    print(f"  Shows counting behavior: {thinking_analysis['shows_counting']}")
    print(f"  Step-by-step reasoning: {thinking_analysis['step_by_step']}")
    print(f"  Mentions 'different': {thinking_analysis['mentions_different']}")
    print(f"  Thinking reward: +{thinking_reward:.2f}")
    
    # Extract predicted number
    pred_number = extract_answer_number(solution_str)
    print(f"\n[Model Response]\n{solution_str}")
    
    # Parse ground truth
    gt_number = parse_ground_truth(ground_truth)
    
    # Calculate final answer score with progressive rewards
    answer_score = 0.0
    if pred_number is not None and gt_number is not None:
        print(f"\n[Number Validation]\n Expected: {gt_number}, Predicted: {pred_number}")
        
        # Progressive reward system for final answer
        if pred_number == gt_number:
            answer_score = 1.0
            print(f"  Result: ✓ Perfect answer! (+1.0)")
        else:
            diff = abs(pred_number - gt_number)
            if diff == 1:
                answer_score = 0.5  # Very close
                print(f"  Result: ~ Close by 1 (+0.7)")
            elif diff == 2:
                answer_score = 0.3  # Somewhat close
                print(f"  Result: ~ Close by 2 (+0.4)")
            elif diff <= 5:
                answer_score = 0.1  # At least in the right ballpark
                print(f"  Result: ~ In ballpark (+0.2)")
            elif diff <= 10:
                answer_score = 0.05  # Show some understanding
                print(f"  Result: ~ Some understanding (+0.1)")
            else:
                answer_score = 0.0  # Too far off
                print(f"  Result: ✗ Too far off (+0.0)")
    else:
        if pred_number is None:
            print("\n[Error] Failed to extract number from model response")
        if gt_number is None:
            print("\n[Error] Failed to parse ground truth number")
        answer_score = 0.0
    
    # Combine thinking reward and answer score
    total_score = thinking_reward + answer_score
    
    print("\n" + "-"*80)
    print(f" Final Score Breakdown ".center(80, '-') + f"\n Thinking Reward: {thinking_reward:.2f}, Answer Reward: {answer_score:.2f}, Total Score: {total_score:.2f}")
    print("="*80 + "\n")
    
    return {
        "score": total_score,
        "acc": answer_score == 1.0,
        "pred": pred_number,
    } 