"""
utils.py - 工具函数
适配 vLLM 推理 + Qwen3-8B enable_thinking 模式
核心改动：
  1. 用 vLLM 的 LLM 类替代 HuggingFace model.generate()
  2. 使用 tokenizer.apply_chat_template 构建 Qwen3 对话格式
  3. 新增 extract_final_answer() 提取 <answer> 标签或 </think> 后的内容
  4. truncate_prompt 适配 vLLM tokenizer
"""

import os
import re
import json
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_dataset_names(dataset_names, length_levels):
    datasets = []
    for name in dataset_names:
        for length in length_levels:
            datasets.append(f"{name}_{length}")
    return datasets


# ============================================================
# 答案提取：从模型的完整输出中提取最终答案
# 优先级：
#   1. 提取 <answer>...</answer> 标签中的内容
#   2. 如果没有 <answer> 标签，提取 </think> 之后的全部内容
#   3. 如果也没有 </think>，返回原始输出
# ============================================================
def extract_final_answer(response: str) -> str:
    """
    从 Qwen3 thinking 模式的输出中提取最终答案。

    模型输出格式通常为：
        <think>思考过程...</think>
        正式回答内容，其中可能包含 <answer>最终答案</answer>

    提取策略：
        1. 优先提取 <answer>...</answer> 中的内容
        2. 若无 <answer> 标签，提取 </think> 之后的内容
        3. 若也无 </think>，返回完整输出（去除首尾空白）
    """
    # 策略1：提取 <answer>...</answer> 中的内容（取最后一个匹配，防止 think 中误匹配）
    answer_matches = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_matches:
        return answer_matches[-1].strip()

    # 策略2：提取 </think> 之后的内容
    think_split = response.split('</think>')
    if len(think_split) > 1:
        after_think = think_split[-1].strip()
        if after_think:
            return after_think

    # 策略3：返回原始输出
    return response.strip()


# ============================================================
# vLLM 模型加载
# ============================================================
def load_vllm_model(model_path, max_model_len, tensor_parallel_size=None, gpu_memory_utilization=0.95):
    """
    加载 vLLM 模型和 tokenizer。

    Args:
        model_path: 模型路径
        max_model_len: 模型最大上下文长度
        tensor_parallel_size: 张量并行 GPU 数量，默认使用全部可用 GPU
        gpu_memory_utilization: GPU 显存利用率
    """
    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count()

    print(f"正在加载模型: {model_path}")
    print(f"  张量并行 GPU 数: {tensor_parallel_size}")
    print(f"  最大上下文长度: {max_model_len}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        # Qwen3 enable_thinking 模式不需要额外参数，
        # 只需要在 chat template 中设置 enable_thinking=True
    )

    print("模型加载完成！")
    return llm, tokenizer


# ============================================================
# 构建 Qwen3 对话格式 prompt（支持 enable_thinking）
# ============================================================
def build_chat_qwen3(tokenizer, prompt, enable_thinking=True):
    """
    使用 Qwen3 的 chat template 构建对话格式。

    enable_thinking=True 时，模型会在回答前先输出 <think>...</think> 思考过程。
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    # Qwen3 的 tokenizer 支持 enable_thinking 参数
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return text


# ============================================================
# prompt 截断（中间截断策略，保留首尾）
# ============================================================
def truncate_prompt(tokenizer, prompt, max_length):
    """
    遵循 LongBench 的中间截断策略：当 prompt 超过 max_length 时，
    保留前一半和后一半的 token，截断中间部分。
    """
    tokenized_prompt = tokenizer(
        prompt, truncation=False, return_tensors="pt"
    ).input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = (
            tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
            + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        )
    return prompt


# ============================================================
# vLLM 批量推理
# ============================================================
def vllm_generate(llm, prompts, max_gen, enable_thinking=True):
    """
    使用 vLLM 进行批量推理。

    Args:
        llm: vLLM 的 LLM 实例
        prompts: 已经格式化好的 prompt 列表（已包含 chat template）
        max_gen: 最大生成 token 数
        enable_thinking: 是否开启 thinking 模式
    """
    sampling_params = SamplingParams(
        max_tokens=max_gen,
        temperature=0.6,       # Qwen3 thinking 模式推荐 temperature=0.6
        top_p=0.95,            # Qwen3 thinking 模式推荐 top_p=0.95
        top_k=20,              # Qwen3 thinking 模式推荐 top_k=20
    )

    outputs = llm.generate(prompts, sampling_params)

    # 按 request_id 排序以保持与输入顺序一致
    outputs = sorted(outputs, key=lambda x: int(x.request_id))

    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append(generated_text)

    return results


# ============================================================
# 数据集加载
# ============================================================
def load_LVEval_dataset(dataset_name, data_path=None):
    """
    加载 LVEval 数据集。

    dataset_name 示例: "dureader_mixup_16k"
    数据集文件夹名: "dureader_mixup"（去掉长度后缀 _16k/_32k 等）

    实际文件路径: {data_path}/{folder_name}/{dataset_name}.jsonl
    例如: /user/.../LV-EVAL/dureader_mixup/dureader_mixup_16k.jsonl
    """
    print(f"正在加载数据集 >>>>>>>>> {dataset_name}")
    if data_path:  # 从本地路径加载
        # 从 dataset_name 中提取文件夹名（去掉长度后缀 _16k/_32k/_64k/_128k/_256k）
        folder_name = re.split(r'_\d{1,3}k$', dataset_name)[0]

        # 优先查找: {data_path}/{folder_name}/{dataset_name}.jsonl
        data_file = os.path.join(data_path, folder_name, dataset_name + ".jsonl")

        # 兼容回退: 如果文件夹结构不存在，尝试直接在 data_path 下查找
        if not os.path.exists(data_file):
            data_file_flat = os.path.join(data_path, dataset_name + ".jsonl")
            if os.path.exists(data_file_flat):
                data_file = data_file_flat
            else:
                print(f"文件不存在: {data_file}")
                print(f"也不存在: {data_file_flat}")
                return []

        datas = load_jsonl(data_file)
        print(f"数据集路径 >>>>>>>>> {data_file} (共 {len(datas)} 条)")
    else:  # 从 HuggingFace 加载
        datas = load_dataset("infini-ai/LVEval", dataset_name, split='test', token=True)
    return list(datas)


def load_jsonl(data_path):
    datas = []
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    datas.append(json.loads(line))
    else:
        print(f"文件不存在: {data_path}")
    return datas


# ============================================================
# 结果保存
# ============================================================
def dump_preds_results(preds, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")
    print(f"结果已保存 >>>>>>>>> {save_path}")