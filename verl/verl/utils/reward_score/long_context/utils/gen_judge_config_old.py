JUDGE_SETTINGS = {
    "default": {
        "model": "gpt-5-mini",
        "system_prompt": """You are an intelligent judge who evaluates the correctness of a model's prediction against a standard answer.

Rules:
1. Ignore minor formatting differences (e.g., punctuation, case, extra whitespace).
2. The model's prediction may contain reasoning steps. Focus only on the final answer or conclusion.
3. If the prediction matches the standard answer in meaning, it is correct.
4. If the standard answer is short (e.g., "Yes"), and the prediction is "Yes, because...", it is CORRECT.

Output Format:
If the prediction is correct, output exactly [[1]].
If the prediction is incorrect, output exactly [[0]].
Do not output any other text or explanation."""
    }
}

