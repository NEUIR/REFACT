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

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.loop_detector import detect_loop
from verl.utils.reward_score.prime_math import (
    _normalize,
    match_answer,
    math_normalize,
)

dapo_executor = ProcessPoolExecutor(max_workers=32)


class DAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        print("overlong_buffer_cfg:", overlong_buffer_cfg)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.enable_loop_detection = kwargs.get("enable_loop_detection", False)
        self.loop_detection_threshold = kwargs.get("loop_detection_threshold", 3)
        
        print(f"[DAPORewardManager] Loop detection enabled: {self.enable_loop_detection}, threshold: {self.loop_detection_threshold}")

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def parallel_compute_score(self, data_sources, response_strs, ground_truths, extra_infos):
        ground_truths = [x['ground_truth'] for x in ground_truths]
        results = list(dapo_executor.map(self.compute_score, data_sources, response_strs, ground_truths, extra_infos))
        return results

    def apply_ttrl(self, data:DataProto):
        for start_pos in range(0,len(data)):
            if data[start_pos].non_tensor_batch[self.reward_fn_key] == 'ttrl':
                # 根据uid查找所有正确结果
                uid=data[start_pos].non_tensor_batch['uid']
                same_uid_indexes=[]
                answer_list=[]
                for i in range(start_pos, len(data)):
                    if data[i].non_tensor_batch['uid'] == uid:
                        same_uid_indexes.append(i)

                        prompt_ids = data[i].batch["prompts"]

                        prompt_length = prompt_ids.shape[-1]
                        response_ids = data[i].batch["responses"]
                        valid_response_length = data[i].batch["attention_mask"][prompt_length:].sum()
                        valid_response_ids = response_ids[:valid_response_length]
                        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                        _, extracted_answer = match_answer(response_str)
                        normalized_answer = math_normalize.normalize_answer(extracted_answer)
                        normalized_answer2 = _normalize(normalized_answer)
                        if normalized_answer2 is None or len(normalized_answer2) == 0:
                            normalized_answer2 = "0"
                        answer_list.append(normalized_answer2)
                        same_uid_indexes.append(i)
                counter = Counter(answer_list)
                mode, count = counter.most_common(1)[0]
                print(f'mode count = {count}')
                for i in same_uid_indexes: # 注意必须后index才是可写入的
                    data.non_tensor_batch['reward_model'][i]['ground_truth'] = mode
                    data.non_tensor_batch[self.reward_fn_key][i] = 'ttrl_checked' # 只有ttrl_checked有校验函数，ttrl没有
        return data

    def find_ttrl(self, data:DataProto, reward_tensor: torch.Tensor):
        # 对相同uid的回答统计正确率，正确率为0的变TTRL数据。谨慎起见枚举加入任务
        for start_pos  in range(0, len(data)):
            if data[start_pos].non_tensor_batch[self.reward_fn_key] != 'ttrl' and \
                data[start_pos].non_tensor_batch[self.reward_fn_key] in ['deepscaler','math500','math_dapo','numina_amc_aime','numina_aops_forum','numina_cn_k12','numina_olympiads','numina_synthetic_math','numina_synthetic_amc']:
                uid = data[start_pos].non_tensor_batch['uid']
                same_uid_indexes=[]
                scores=[]

                for i in range(start_pos, len(data)):
                    if data[i].non_tensor_batch['uid'] == uid:
                        same_uid_indexes.append(i)
                        scores.append(reward_tensor[i])

                if torch.cat(scores, dim=0).sum() == 0:

                    for i in same_uid_indexes:

                        data.non_tensor_batch[self.reward_fn_key][i] = 'ttrl'
        return

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        response_ids = data.batch['responses']
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        extra_infos = data.non_tensor_batch.get('extra_info', None)
        if extra_infos is None:
            extra_infos = [None] * len(sequences_str)
        print("dapo computing score, len(sequences_str):", len(sequences_str))
        results = self.parallel_compute_score(
            data_sources=data.non_tensor_batch[self.reward_fn_key],
            response_strs=sequences_str,
            ground_truths=data.non_tensor_batch['reward_model'],
            extra_infos=extra_infos,
        )

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            # extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # result = self.compute_score(
            #     data_source=data_source,
            #     solution_str=response_str,
            #     ground_truth=ground_truth,
            #     extra_info=extra_info,
            # )
            result = results[i]

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["score"].append(score)
                reward_extra_info["acc"].append(float(score))
                reward_extra_info["pred"].append(response_str)

            # Always record the ground-truth answer
            reward_extra_info["answer"].append(ground_truth)
            
            # Detect loop
            if self.enable_loop_detection:
                is_loop = detect_loop(response_str, threshold=self.loop_detection_threshold, debug=False)
                reward_extra_info["loop_detected"].append(is_loop)
                if is_loop and data_source not in already_print_data_sources:
                    print(f"[Loop Detected] Sample {i}: data_source={data_source}")
            else:
                # Always append False if detection is disabled, to maintain list length
                reward_extra_info["loop_detected"].append(False)

            reward = score

            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                print("computing overlong reward")
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
