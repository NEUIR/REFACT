import os
import re
import argparse
from tqdm import tqdm

from config_vllm import (
    PROMPT_TYPES,
    get_dataset_prompt,
    DATASET_SELECTED,
    DATASET_LENGTH_LEVEL,
)
from utils_vllm import (
    ensure_dir,
    seed_everything,
    get_dataset_names,
    build_chat_qwen3,
    truncate_prompt,
    extract_final_answer,
    vllm_generate,
    dump_preds_results,
    load_LVEval_dataset,
    load_vllm_model,
)


def get_pred_vllm(
    llm,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    enable_thinking=True,
    batch_size=32,
):
    print("正在构建 prompt ...")
    all_prompts = []
    for json_obj in tqdm(data, desc="构建 prompt"):
        prompt = prompt_format.format(**json_obj)
        prompt = truncate_prompt(tokenizer, prompt, max_length)
        chat_prompt = build_chat_qwen3(
            tokenizer, prompt, enable_thinking=enable_thinking
        )
        all_prompts.append(chat_prompt)

    print(f"开始 vLLM 推理，共 {len(all_prompts)} 条，batch_size={batch_size} ...")
    all_raw_preds = []
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="vLLM 推理"):
        batch_prompts = all_prompts[i : i + batch_size]
        batch_results = vllm_generate(
            llm, batch_prompts, max_gen, enable_thinking=enable_thinking
        )
        all_raw_preds.extend(batch_results)

    print("正在提取答案 ...")
    preds = []
    for json_obj, raw_pred in zip(data, all_raw_preds):
        final_answer = extract_final_answer(raw_pred)
        preds.append(
            {
                "pred": final_answer,
                "raw_pred": format_raw_pred_for_save(raw_pred),
                "answers": json_obj["answers"],
                "gold_ans": json_obj.get("answer_keywords", None),
                "input": json_obj["input"],
                "all_classes": json_obj.get("all_classes", None),
                "length": json_obj["length"],
            }
        )

    return preds


def format_raw_pred_for_save(raw_pred):
    if "\n\n<answer>" not in raw_pred:
        return raw_pred
    raw_pred = raw_pred.replace("</think>", "")
    return raw_pred.replace("\n\n<answer>", "</think>\n\n<answer>", 1)


def process_datasets(datasets, args):
    """加载 vLLM 模型，遍历所有数据集进行推理。"""
    llm, tokenizer = load_vllm_model(
        model_path=args.model_path,
        max_model_len=args.model_max_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    prompt_max_length = args.model_max_length - args.max_new_tokens
    print(
        f"Prompt 截断长度: {prompt_max_length} "
        f"(模型最大长度 {args.model_max_length} - 最大输出 {args.max_new_tokens})"
    )

    output_dir = args.output_dir
    ensure_dir(output_dir)

    for dataset in tqdm(datasets, desc="数据集进度"):
        output_path = os.path.join(output_dir, dataset + ".jsonl")

        if os.path.exists(output_path) and not args.overwrite:
            print(f"跳过（已存在）: {output_path}")
            continue

        datas = load_LVEval_dataset(dataset, args.data_path)
        if not datas:
            print(f"警告：数据集 {dataset} 为空，跳过")
            continue

        dataset_name = re.split(r"_\d{1,3}k", dataset)[0]
        prompt_format = get_dataset_prompt(args.prompt_type, dataset_name)

        preds = get_pred_vllm(
            llm=llm,
            tokenizer=tokenizer,
            data=datas,
            max_length=prompt_max_length,
            max_gen=args.max_new_tokens,
            prompt_format=prompt_format,
            enable_thinking=args.enable_thinking,
            batch_size=args.batch_size,
        )

        dump_preds_results(preds, output_path)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="LVEval 推理脚本 (vLLM + Qwen3 thinking + CoT/CoC prompt 切换)"
    )

    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=list(PROMPT_TYPES.keys()),
        required=True,
        help="prompt 类型：cot=直接作答+<answer> 标签；coc=推理链+evidence 引用+<answer> 标签",
    )

    parser.add_argument(
        "--model-path", type=str, required=True, help="模型路径（如 Qwen3-8B）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="模型名称（默认从 model-path 提取）",
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=131072,
        help="模型最大上下文长度（默认 131072）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="模型最大输出长度（默认 8192，含 thinking 过程）",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="本地数据集路径（不指定则从 HuggingFace 加载）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="输出根目录；最终落盘到 <output-dir>/<prompt-type>/<dataset>.jsonl",
    )
    parser.add_argument(
        "--no-prompt-type-subdir",
        action="store_true",
        help="不在输出目录下追加 prompt 类型子目录（直接写入 --output-dir）",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="覆盖已有结果文件"
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="张量并行 GPU 数量（默认使用全部 GPU）",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU 显存利用率（默认 0.95）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="vLLM 推理 batch 大小（默认 32）",
    )

    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=True,
        help="开启 Qwen3 thinking 模式（默认开启）",
    )
    parser.add_argument(
        "--no-thinking", action="store_true", help="关闭 thinking 模式"
    )

    args = parser.parse_args(args)

    if args.no_thinking:
        args.enable_thinking = False

    args.prompt_type = args.prompt_type.lower()

    model_path = args.model_path.rstrip("/")
    if not args.model_name:
        args.model_name = os.path.basename(model_path)

    if not args.no_prompt_type_subdir:
        args.output_dir = os.path.join(args.output_dir, args.prompt_type)

    assert args.max_new_tokens < args.model_max_length, (
        f"--max-new-tokens ({args.max_new_tokens}) 必须小于 "
        f"--model-max-length ({args.model_max_length})"
    )

    return args


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    ensure_dir(args.output_dir)

    datasets = get_dataset_names(DATASET_SELECTED, DATASET_LENGTH_LEVEL)

    print("=" * 60)
    print(f"Prompt 类型: {args.prompt_type}")
    print(f"模型: {args.model_name}")
    print(f"模型路径: {args.model_path}")
    print(f"最大上下文长度: {args.model_max_length}")
    print(f"最大输出长度: {args.max_new_tokens}")
    print(f"Thinking 模式: {'开启' if args.enable_thinking else '关闭'}")
    print(f"Batch Size: {args.batch_size}")
    print(f"数据集数量: {len(datasets)}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    process_datasets(datasets, args)

    print("\n全部推理完成！")
