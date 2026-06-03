import os
import re
import json
import argparse
import pandas as pd

from config_vllm import DATASET_METRIC, PROMPT_TYPES
from utils_vllm import ensure_dir


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="LVEval 评测脚本（支持 CoT / CoC 子目录布局）"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="预测结果根目录；默认会拼接 prompt 类型子目录",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=list(PROMPT_TYPES.keys()) + ["all"],
        default="coc",
        help="评测哪种 prompt 类型；'all' 表示同时评测所有类型",
    )
    parser.add_argument(
        "--no-prompt-type-subdir",
        action="store_true",
        help="不在 input-dir 下追加 prompt 类型子目录（input-dir 已直接指向 jsonl）",
    )
    return parser.parse_args(args)


def custom_sort(s):
    letters = re.findall(r"[a-zA-Z]+", s)
    numbers = re.findall(r"\d+", s)
    return (letters, int(numbers[0])) if numbers else (letters, 0)


def scorer(dataset, predictions, answers, gold_anss):
    dataset_name = re.split(r"_.{1,3}k", dataset)[0]
    if dataset_name not in DATASET_METRIC:
        raise KeyError(f"未在 DATASET_METRIC 中找到指标定义: {dataset_name}")
    metric_fn = DATASET_METRIC[dataset_name]

    total_score = 0.0
    total_sample = 0
    scores = {metric_fn.__name__: []}
    for prediction, ground_truths, gold_ans in zip(predictions, answers, gold_anss):
        total_sample += 1
        score = 0.0
        for ground_truth in ground_truths:
            score = max(score, metric_fn(prediction, ground_truth, gold_ans))
            break
        total_score += score
        scores[metric_fn.__name__].append(score)

    if total_sample == 0:
        return 0.0, scores
    return round(100 * total_score / total_sample, 2), scores


def evaluate_one_dir(eval_dir: str, prompt_type_label: str = ""):
    """
    对单个目录下的所有 *.jsonl 进行评测，结果写入 <eval_dir>/eval_result/。
    prompt_type_label 仅用于打印与日志区分。
    """
    if not os.path.isdir(eval_dir):
        print(f"[跳过] 目录不存在: {eval_dir}")
        return

    save_dir = os.path.join(eval_dir, "eval_result")
    ensure_dir(save_dir)

    all_files = [
        f for f in os.listdir(eval_dir) if os.path.isfile(os.path.join(eval_dir, f))
    ]
    all_files.sort(key=custom_sort)

    all_results = dict()
    all_scores = dict()

    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, gold_anss = [], [], []
        dataset = filename.split(".")[0]
        dataset_name = re.split(r"_.{1,3}k", dataset)[0]
        length = dataset.split("_")[-1]

        with open(os.path.join(eval_dir, filename), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                gold_anss.append(data.get("gold_ans", None))

        if not predictions:
            print(f"[跳过] {filename} 无有效样本")
            continue

        score_mean, _metric_scores = scorer(dataset, predictions, answers, gold_anss)
        all_scores[dataset] = score_mean
        all_results.setdefault(dataset_name, []).append({length: score_mean})

    out_json = os.path.join(save_dir, "result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=4)

    panda_list = []
    for dataset_name, length_score_list in all_results.items():
        lengths_scores = dict()
        for item in length_score_list:
            length, score = list(item.items())[0]
            lengths_scores[length] = score
        panda_dict = {"dataset_name": dataset_name}
        panda_dict.update(**lengths_scores)
        panda_list.append(panda_dict)

    dataframe = pd.DataFrame(panda_list)
    if prompt_type_label:
        print(f"\n=== prompt_type = {prompt_type_label} ===")
    print(f"评测目录: {eval_dir}")
    print(dataframe, "\n")

    out_csv = os.path.join(save_dir, "result.csv")
    dataframe.to_csv(out_csv, index=False)
    print(f"已写入: {out_json}\n          {out_csv}")


def main():
    args = parse_args()
    root_dir = args.input_dir.rstrip("/")

    if args.prompt_type == "all":
        targets = list(PROMPT_TYPES.keys())
    else:
        targets = [args.prompt_type]

    for pt in targets:
        if args.no_prompt_type_subdir:
            eval_dir = root_dir
            label = pt if args.prompt_type == "all" else ""
        else:
            eval_dir = os.path.join(root_dir, pt)
            label = pt

        evaluate_one_dir(eval_dir, prompt_type_label=label)


if __name__ == "__main__":
    main()
