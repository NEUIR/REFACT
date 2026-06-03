import os
import json
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=[
        "llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k",
        "internlm-7b-8k", "chatglm2-6b", "Qwen3-4B-LongFaith", "Qwen3-8B-LongFaith",
        "Qwen3-8B-Cite", "Qwen3-4B-Cite", "Qwen3-4B-ACC", "Qwen3-8B-SFT-ruler",
        "Qwen3-8B-SFT", "Qwen3-8B-Base", "Qwen3-8B-RL", "Qwen3-4B", "Qwen3-8B-RL-megtron","Qwen3-8B-RL-LLama"
    ])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--data_path', type=str, default="/user/jinzhensheng/data/",
                        help="Path to local LongBench data folder")
    return parser.parse_args(args)


def load_local_data(data_path, dataset, use_e=False):
    """从本地jsonl文件加载数据"""
    file_name = f"{dataset}_e.jsonl" if use_e else f"{dataset}.jsonl"
    file_path = os.path.join(data_path, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {file_path}")
    return data


def add_cot_instruction(prompt, dataset):
    """在prompt末尾添加思考链指令（预留扩展接口）"""
    return prompt


# ============== build_chat 函数 ==============
def build_chat(tokenizer, prompt, model_name, enable_think=True):
    """构建不同模型的chat格式"""
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "Qwen" in model_name or "qwen" in model_name:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
    return prompt


# ============== 后处理函数 ==============

THINK_TAG_PATTERN = re.compile(r'</?think>')
ANSWER_TAG_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)


def extract_answer_tag(response):
    """
    从 <answer>...</answer> 标签中提取答案。

    优先提取最后一个 <answer> 标签的内容（防止思考过程中出现的干扰）。

    Returns:
        (str, bool): (提取的内容, 是否成功提取)
    """
    matches = ANSWER_TAG_PATTERN.findall(response)
    if matches:
        # 取最后一个 <answer> 标签的内容
        answer = matches[-1].strip()
        if answer:
            return answer, True
    # 检查是否有未闭合的 <answer> 标签
    if "<answer>" in response and "</answer>" not in response:
        pos = response.rfind("<answer>")
        content = response[pos + len("<answer>"):].strip()
        if content:
            print("[WARNING] <answer> 未闭合，提取其后内容作为答案")
            return content, True
    return response, False


def split_by_last_think_close(response):
    """
    按最后一个 </think> 将文本切成前后两段。

    Returns:
        (str, str, bool):
        - think_part: </think> 之前(含可能的思考内容)
        - tail_part: </think> 之后
        - has_close_think: 是否存在 </think>
    """
    last_pos = response.rfind("</think>")
    if last_pos == -1:
        return response, response, False
    think_part = response[:last_pos + len("</think>")]
    tail_part = response[last_pos + len("</think>"):]
    return think_part, tail_part, True


def extract_after_think(response):
    """
    从模型输出中提取最终答案。

    提取优先级：
    1. </think> 之后的 <answer>...</answer>（最高优先级）
    2. </think> 之后的纯文本
    3. 全文 <answer>...</answer>（仅在无 </think> 时启用）
    4. 兜底策略（取最后一行等）

    处理情况：
    1. 正常: <think>思考过程</think><answer>最终答案</answer>
    2. 正常: <think>思考过程</think>最终答案
    3. 未闭合: <think>思考过程...（无</think>，取最后一行）
    4. 无标签: 直接返回原文
    """
    think_part, tail_part, has_close_think = split_by_last_think_close(response)
    tail_part = tail_part.strip()

    # === 优先级1: 只从 </think> 后提取 <answer> ===
    if has_close_think:
        answer, found = extract_answer_tag(tail_part)
        if found:
            return answer

        # === 优先级2: </think> 后纯文本 ===
        if tail_part:
            return tail_part

        # </think> 之后为空，取思考内容最后一行兜底
        print("[WARNING] </think> 之后没有内容，尝试从思考内容中提取")
        think_match = re.search(r'<think>(.*?)</think>', think_part, re.DOTALL)
        if think_match:
            lines = [l.strip() for l in think_match.group(1).split('\n') if l.strip()]
            if lines:
                return lines[-1]
        return response.strip()

    # === 无 </think>：兼容旧输出，允许从全文 <answer> 提取 ===
    answer, found = extract_answer_tag(response)
    if found:
        return answer

    # <think> 未闭合，取最后一行兜底
    if "<think>" in response:
        print("[WARNING] <think> 未闭合，思考过程可能被截断")
        pos = response.rfind("<think>")
        content = response[pos + len("<think>"):].strip()
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if lines:
            return lines[-1]

    return response.strip()


ANSWER_CLEAN_PATTERN = re.compile(r'</?answer>')


def post_process(response, model_name):
    """
    后处理主函数：
    1. 模型特定的清理
    2. 从 <answer> 或 </think> 后提取答案（Qwen系列）
    3. 清理答案格式
    """
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]

    if "Qwen" in model_name or "qwen" in model_name:
        answer = extract_after_think(response)
    else:
        # 非Qwen模型也支持 <answer> 标签提取
        extracted, found = extract_answer_tag(response)
        answer = extracted if found else response.strip()

    # 最终清理：移除残留的 think 和 answer 标签
    answer = THINK_TAG_PATTERN.sub('', answer)
    answer = ANSWER_CLEAN_PATTERN.sub('', answer)
    answer = re.sub(r'^(Answer|答案|回答)\s*[:：]\s*', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\s+', ' ', answer).strip()

    return answer if answer else None


def load_model_and_tokenizer(path, model_name, max_length):
    """加载模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    llm = LLM(
        model=path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_length,
        tensor_parallel_size=8
    )
    return llm, tokenizer


def get_pred(data, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path, raw_out_path):
    """批量推理并保存结果"""
    llm, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, max_length)

    is_qwen = "Qwen" in model_name or "qwen" in model_name

    # 准备所有prompts
    prompts = []
    for json_obj in tqdm(data, desc="Preparing prompts"):
        prompt = prompt_format.format(**json_obj)
        prompt = add_cot_instruction(prompt, dataset)

        add_special = "chatglm3" not in model_name
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt",
                                     add_special_tokens=add_special).input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                     tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name, enable_think=is_qwen)

        prompts.append(prompt)

    # 设置采样参数
    if dataset == "samsum":
        sampling_params = SamplingParams(
            max_tokens=max_gen,
            temperature=0,
            stop_token_ids=[tokenizer.eos_token_id,
                            tokenizer.encode("\n", add_special_tokens=False)[-1]],
        )
    elif is_qwen:
        sampling_params = SamplingParams(
            max_tokens=max_gen,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=max_gen,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

    print(f"\n开始推理 {dataset}，共 {len(prompts)} 条数据...")
    print(f"  max_tokens={max_gen}")
    outputs = llm.generate(prompts, sampling_params)

    # 保存结果 & 统计
    stats = {
        "total": 0,
        "think_extracted": 0,
        "truncated": 0,
        "no_think": 0,
        "empty_answer": 0,
        "answer_tag_used": 0,  # 新增：通过 <answer> 标签提取的数量
    }

    with open(out_path, "w", encoding="utf-8") as f, \
         open(raw_out_path, "w", encoding="utf-8") as f_raw:
        for i, output in enumerate(outputs):
            raw_pred = output.outputs[0].text
            stats["total"] += 1

            has_open = "<think>" in raw_pred
            has_close = "</think>" in raw_pred
            has_answer_tag = "<answer>" in raw_pred

            if has_answer_tag:
                stats["answer_tag_used"] += 1

            if has_open and has_close:
                stats["think_extracted"] += 1
            elif has_open and not has_close:
                stats["truncated"] += 1
            else:
                stats["no_think"] += 1

            pred = post_process(raw_pred, model_name)

            if pred is None:
                stats["empty_answer"] += 1
                pred = ""

            json.dump({
                "pred": pred,
                "answers": data[i]["answers"],
                "all_classes": data[i]["all_classes"],
                "length": data[i]["length"],
            }, f, ensure_ascii=False)
            f.write('\n')

            json.dump({
                "raw_pred": raw_pred,
                "pred": pred,
                "answers": data[i]["answers"],
                "all_classes": data[i]["all_classes"],
                "length": data[i]["length"],
                "input": data[i].get("input", ""),
            }, f_raw, ensure_ascii=False)
            f_raw.write('\n')

    total = stats['total']
    print(f"\n=== {dataset} 推理统计 ===")
    print(f"  总数: {total}")
    print(f"  <answer>标签提取: {stats['answer_tag_used']} ({stats['answer_tag_used']/total*100:.1f}%)")
    print(f"  <think>正常提取: {stats['think_extracted']} ({stats['think_extracted']/total*100:.1f}%)")
    print(f"  思考被截断(未闭合): {stats['truncated']} ({stats['truncated']/total*100:.1f}%)")
    print(f"  无思考标签: {stats['no_think']} ({stats['no_think']/total*100:.1f}%)")
    print(f"  答案为空: {stats['empty_answer']} ({stats['empty_answer']/total*100:.1f}%)")
    print(f"  提取后答案保存至: {out_path}")
    print(f"  完整原始输出保存至: {raw_out_path}\n")


if __name__ == '__main__':
    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = args.model
    data_path = args.data_path
    max_length = model2maxlen[model_name]

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
                     "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        #datasets = ["hotpotqa"]
        datasets = ["multifieldqa_en", "qasper", "hotpotqa", "2wikimqa", "musique"]

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    for d in ["pred", "pred_raw"]:
        os.makedirs(d, exist_ok=True)

    for dataset in datasets:
        data_all = load_local_data(data_path, dataset, use_e=args.e)
        os.makedirs(f"pred/{model_name}", exist_ok=True)
        os.makedirs(f"pred_raw/{model_name}", exist_ok=True)
        out_path = f"pred/{model_name}/{dataset}.jsonl"
        raw_out_path = f"pred_raw/{model_name}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        get_pred(data_all, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path, raw_out_path)