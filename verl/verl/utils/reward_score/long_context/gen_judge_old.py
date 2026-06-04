import re
import time
from .api import call_api
from .utils.gen_judge_config import JUDGE_SETTINGS

def score_with_llm_judge(pred, answer, judge_config_key):
    config = JUDGE_SETTINGS[judge_config_key]
    judge_model = config["model"]
    judge_system_prompt = config["system_prompt"]

    # call api and parse score
    # Since api.py wraps content in a single user message, we construct a single string
    messages = f"{judge_system_prompt}\n\nStandard Answer: {answer}\n\nModel Prediction: {pred}"

    # if any error, recall api
    # if too many errors, return 0
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Set temperature to 0 for deterministic scoring
            response = call_api(judge_model, messages, temperature=0.0)
            score = parse_score(response)
            if score is not None:
                return score
        except Exception as e:
            print(f"Error calling judge {judge_model}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    # return score (typically 0 or 1)
    return 0.0

def parse_score(response):
    if not response:
        return None
        
    # Look for [[score]] pattern first
    match = re.search(r'\[\[(\d+)\]\]', response)
    if match:
        val = int(match.group(1))
        if val in [0, 1]:
            return float(val)

    # Look for "Score: X"
    match = re.search(r'Score:\s*(\d+)', response, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if val in [0, 1]:
            return float(val)
            
    # Check if the response is just a number
    text = response.strip()
    if text in ['0', '1']:
        return float(text)
        
    return None


if __name__ == "__main__":
    pred = "Among the boxers whose professional debut is explicitly linked to the age of 21 (英文名证实或推断为出发年龄), and considering the question for the youngest in terms of current age (as of 2024), the three relevant boxers are Hugh Roddin, Charlie Flynn, and David Brophy. Hugh Roddin, born in 1887, competed at 21 in 1908 but died in 1954 at age 66, placing him deceased and not relevant for the youngest living fighter. Charlie Flynn, born 6 November 1993, turned professional in 2014 at age 21 and is currently alive (as of 2024). David Brophy, born 9 June 1990, turned professional in February 2011 at age 21 and is also alive (as of 2024). Calculate their ages: Charlie Flynn is 2024 - 1993 = 31 years old, and David Brophy is 2024 - 1990 = 34 years old. Therefore, Charlie Flynn is younger than David Brophy (31 vs. 34 years old in 2024), making him the youngest of the trio. Other boxers like Kevin Anderson, John Cheshire, and Jackie Brown made their debut in their 20s but not specifically at age 21. The remaining boxers (e.g., Jim Brady, Doug Young, Murray Sutherland, Ian McLeod, John Simpson, Walter Ross, and Alex Arthur) do not confirm a debut age of 21, or relevant context for its timing to BMI 2024 comparisons. Thus, Charlie Flynn is the youngest professional boxer who famously debuted at age 21 as of 2024."

    answer = "John Docherty (boxer)"

    print(score_with_llm_judge(pred, answer, "default"))