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
import math
from typing import Dict, Tuple, Optional
import numpy as np
from sympy import Rational


def round_up_to_decimal(number, decimals):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        pass
    return False


def normalize(prediction: str):
    """Normalize the prediction string to a standard format."""
    # Preprocessing the string [Stage 1]
    if '<|startofstep|>' in prediction:
        prediction = prediction.split('<|startofstep|>')[0]
    
    if 'boxed' in prediction:
        ans_match = re.compile(r'\\boxed\{(.+?)\}')
        match = ans_match.findall(prediction)
        if match:
            if isinstance(match[-1], tuple):
                match = match[-1][-1]
            else:
                match = match[-1]
            prediction = match
    
    if 'herefore' in prediction:
        prediction = prediction.split('herefore')[-1].strip()
    
    if 'final answer' in prediction or 'Final answer' in prediction:
        if 'final answer' in prediction:
            prediction = prediction.split('final answer')[-1].strip()
        else:
            prediction = prediction.split('Final answer')[-1].strip()
    
    if re.match(r'^[^a-zA-Z0-9+-]+', prediction):
        prediction = re.sub(r'^[^a-zA-Z0-9+-]+', '', prediction)
    
    if re.match(r'.*[^a-zA-Z0-9+-]+$', prediction):
        prediction = re.sub(r'[^a-zA-Z0-9+-]+$', '', prediction)
    
    if 'The answer is' in prediction:
        if 'is $' in prediction:
            ans_match = re.compile(r'is \$(.+?)\$')
            match = ans_match.findall(prediction)
            if match:
                prediction = match[-1]
        else:
            prediction = prediction.split('The answer is')[-1].strip()
    
    # Remove dollar signs and other formatting
    if '$' in prediction:
        prediction = prediction.replace('$', '')
    
    # Handle special number formats
    if re.match(r'[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$', prediction):
        prediction = re.search(r'([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$', prediction).group(1)
    if re.match(r'[^-+\d]{1,2}(?:[\d,]*\.*\d+)$', prediction):
        prediction = re.search(r'[^-+\d]{1,2}((?:[\d,]*\.*\d+))$', prediction).group(1)

    # Preprocessing the number [Stage 1]
    if '10^' in prediction:
        prediction = re.sub(r'10\^(-?\d+)', r'math.pow(10, \1)', prediction)
    if ' x ' in prediction:
        prediction = prediction.replace(' x ', '*')
    if ' × ' in prediction:
        prediction = prediction.replace(' × ', '*')
    if is_number(prediction):
        prediction = prediction.replace(',', '')

    # Preprocessing the option [Stage 3]
    if '(a)' in prediction or '(b)' in prediction or '(c)' in prediction or '(d)' in prediction:
        prediction = '"' + re.search(r'\([a-d]\)', prediction).group(0) + '"'

    # If the prediction is empty, use dummy '0'
    if not prediction:
        prediction = '0'

    # Converting the string answer to a number/list/bool/option
    try:
        prediction = eval(prediction)
    except Exception:
        # Fallback: try extracting a number from free-form text (less strict).
        m = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?", str(prediction))
        if m:
            try:
                prediction = float(m[-1].replace(",", ""))
            except Exception:
                prediction = 0
        else:
            prediction = 0 

    # Performing common type conversion
    if isinstance(prediction, (set, tuple)):
        prediction = list(prediction)
        if isinstance(prediction[0], complex):
            prediction = [tmp.real for tmp in prediction]
        elif isinstance(prediction[0], Rational):
            prediction = [float(tmp) for tmp in prediction]
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    else:
        if isinstance(prediction, complex):
            prediction = prediction.real
        elif isinstance(prediction, Rational):
            prediction = float(prediction)

    return prediction


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extract the final answer from the model's response string."""
    # Prefer content after </think>. If missing, fall back to using the whole output
    # (rollouts can omit tags; reward should still be computable).
    if "</think>" in solution_str:
        final_answer = solution_str.split("</think>")[-1].strip()
        return final_answer, solution_str

    # Fallback: use the tail (more likely contains the final answer)
    print("[Warn] No </think> tag found; fallback to tail extraction")
    tail = solution_str[-2000:].strip()
    return tail, solution_str


def parse_model_answer(answer_text: str) -> Optional[str]:
    """Parse the model's answer to extract the actual value."""
    if not answer_text:
        return None
    
    if isinstance(answer_text, dict) and "target" in answer_text:
        answer_text = answer_text["target"]
    
    # Remove "Therefore, the answer is" prefix if present
    if "Therefore, the answer is" in answer_text:
        answer_text = answer_text.replace("Therefore, the answer is", "").strip().rstrip(".").strip()

    # Common variants
    for marker in [
        "The correct answer is",
        "The answer is",
        "Answer:",
        "Final answer:",
        "final answer:",
    ]:
        if marker in answer_text:
            answer_text = answer_text.split(marker, 1)[-1].strip().rstrip(".").strip()

    # If multiple-choice, extract (a)-(d)
    m_opt = re.search(r"\(([a-dA-D])\)", answer_text)
    if m_opt:
        return f"({m_opt.group(1).lower()})"

    # Otherwise, robustly extract the last numeric token from the text.
    # This makes reward less strict about verbose explanations, currency symbols, commas, etc.
    num_pat = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
    nums = num_pat.findall(answer_text)
    if nums:
        return nums[-1].replace(",", "")

    return answer_text.strip() if answer_text.strip() else None


def compare_two_numbers(p, gt, include_percentage=True, tolerance=1e-5):
    """Compare two numbers for near-equality."""
    # Early return for exact matches
    if p == gt:
        return True
    
    # Handle float comparisons
    if not isinstance(p, float) or not isinstance(gt, float):
        return False
    
    # Handle case when gt is zero
    if abs(gt) < tolerance:
        # If gt is essentially zero, check if p is also essentially zero
        return abs(p) < tolerance
    
    # Relative tolerance check
    if 0.99 < p / gt < 1.01:
        return True
    
    # Absolute tolerance check
    if abs(p - gt) < tolerance:
        return True
    
    # Percentage representation check
    if include_percentage and 0.99 < p / (gt * 100) < 1.01:
        return True
    
    return False


def get_acc(prediction, gt, cot=True):
    """Calculate accuracy by comparing prediction with ground truth."""
    print(f"get_acc({prediction}, {gt})")
    
    if cot:
        prediction = normalize(prediction)
        gt = normalize(gt)
    else:
        try:
            prediction = float(prediction)
        except:
            return 0
    
    print(f"after normalize pre = {prediction}")
    print(f"after normalize gt = {gt}")
    
    answer_type = type(gt).__name__
    print(f"answer_type::{answer_type}")
    
    if answer_type not in ["int", "float", "float64", "bool"]:
        return 0
    
    if isinstance(prediction, (str, int, float, bool)) or isinstance(prediction, list):
        # Comparing prediction against the reference
        if answer_type in ['bool']:
            acc = int(prediction == gt)
        elif answer_type == 'int':
            acc = int(compare_two_numbers(prediction, gt))
        elif answer_type == 'float' or answer_type == 'float64':
            acc = int(compare_two_numbers(prediction, gt))
        else:
            acc = 0
    else:
        acc = 0
        print("Error: ", prediction, type(prediction))
    
    return acc


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
            acc = get_acc(pred_status, gt_status)
            answer_score = acc
            print(f"  Answer Score: {answer_score}")
        else:
            answer_score = 0
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