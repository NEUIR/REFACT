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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import random
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import rich
import torch
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.workers.rollout.async_server import AsyncLLMServerManager
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.reward import compute_reward
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    WorkerType,
    _timer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):
        self.dynamic_sample_buffer = None
        self.offline_sequence_buffer = None
        self.uid2old_log_prob = dict()
        self.filtered_correct_number = 0
        self.filtered_incorrect_number = 0
        self.async_rollout_mode = False
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn)

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role="actor_rollout",
                                                     reward_config=self.config,)
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_index, batch_dict in enumerate(self.train_dataloader):
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    if "multi_modal_inputs" in new_batch.non_tensor_batch.keys():
                        gen_batch = new_batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                        )
                    else:
                        gen_batch = new_batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids"],
                        )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    pprint("finished generation")
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            if self.config.reward_model.get('parallel_reward', False):
                                # pad the batch
                                new_batch, pad_size = pad_dataproto_to_divisor(new_batch, self.actor_rollout_wg.world_size)
                                # shuffle the batch
                                idxs = list(range(len(new_batch)))
                                random.shuffle(idxs)
                                new_batch = new_batch.select_idxs(idxs)
                                reward_result = self.actor_rollout_wg.verify(new_batch)
                                # restore the batch
                                restore_idxs = [0] * len(idxs)
                                for i, idx in enumerate(idxs):
                                    restore_idxs[idx] = i
                                new_batch = new_batch.select_idxs(restore_idxs)
                                reward_result = reward_result.select_idxs(restore_idxs)
                                # unpad the batch
                                new_batch = unpad_dataproto(new_batch, pad_size=pad_size)
                                reward_result = unpad_dataproto(reward_result, pad_size=pad_size)
                                reward_tensor = reward_result.batch['reward_tensor']
                                reward_extra_infos_dict = reward_result.non_tensor_batch
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)
                            new_batch.batch["token_level_scores"] = reward_tensor
                            print(f"{list(reward_extra_infos_dict.keys())=}")
                        except Exception as e:
                            print(f"Error in reward_fn: {e}, {traceback.format_exc()}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        
                        # --- Start Loop Detection Logic Migration ---
                        # If the reward function returns 'loop_detected' info, we can use it to penalize the score
                        loop_detected = None
                        if 'loop_detected' in reward_extra_infos_dict:
                            loop_detected = reward_extra_infos_dict['loop_detected']
                        elif 'loop_detected' in new_batch.non_tensor_batch:
                            loop_detected = new_batch.non_tensor_batch['loop_detected']

                        if loop_detected is not None and self.config.reward_model.get("reward_kwargs", {}).get("enable_loop_detection", False):
                            loop_penalty_strategy = self.config.reward_model.get("loop_penalty_strategy", "zero_reward")
                            loop_penalty_value = self.config.reward_model.get("loop_penalty_value", 0.5)
                            
                            total_samples = len(loop_detected)
                            if total_samples > 0:
                                # Handle both list and numpy array
                                if isinstance(loop_detected, np.ndarray):
                                    loop_count = int(loop_detected.sum())  # Convert bool array to count
                                else:
                                    loop_count = sum(1 for is_loop in loop_detected if is_loop)
                                loop_rate = loop_count / total_samples
                                print(f"[Loop Detection Stats] {loop_count}/{total_samples} loops detected (rate: {loop_rate:.2%})")
                                metrics["training/loop_rate"] = loop_rate
                            
                            for i in range(len(loop_detected)):
                                # Handle both list and numpy array
                                is_loop = bool(loop_detected[i]) if isinstance(loop_detected, np.ndarray) else loop_detected[i]
                                if is_loop:
                                    if loop_penalty_strategy == "mask":
                                        # Strategy 1: Completely mask the response (no gradient)
                                        # Note: DAPO might not support response_masks modification here directly as it reconstructs batch later
                                        # But let's try to set it if possible, though DAPO logic is complex with dynamic buffer.
                                        # For safety in DAPO, zero_reward is preferred.
                                        pass
                                    elif loop_penalty_strategy == "zero_reward":
                                        # Strategy 2: Set reward to 0 (before GRPO advantage calculation)
                                        new_batch.batch["token_level_scores"][i] = 0.0
                                    elif loop_penalty_strategy == "nothing":
                                        pass
                                    elif loop_penalty_strategy == "constant_penalty":
                                        # Strategy 4: Apply constant penalty
                                        current_score = new_batch.batch["token_level_scores"][i].sum()
                                        new_score = torch.clamp(current_score - loop_penalty_value, min=0.0)
                                        if torch.abs(current_score) > 1e-6:
                                            new_batch.batch["token_level_scores"][i] *= (new_score / current_score)
                                        else:
                                            new_batch.batch["token_level_scores"][i] = 0.0
                        # --- End Loop Detection Logic Migration ---

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
                    print("finished reward")

                    # If all the rollouts are incorrect for a prompt, we should try to use tgt_input_ids from the dataset.
                    if self.config.actor_rollout_ref.actor.get('use_off_policy_loss', False):
                        this_bsz = len(new_batch.batch["input_ids"])
                        N = self.config.actor_rollout_ref.rollout.n
                        is_offline = []
                        for i in range(this_bsz // N):
                            assert len(set(new_batch.non_tensor_batch["uid"][i * N:(i + 1) * N])) == 1, "all the rollouts for a prompt should have the same uid"
                            if (new_batch.batch["token_level_scores"][i * N:(i + 1) * N].sum(dim=-1) < 0.025).all():
                                tgt_response_length = (new_batch.batch["tgt_input_ids"][i * N] != self.tokenizer.pad_token_id).sum()
                                if tgt_response_length > 0:
                                    print(f"use tgt_input_ids from the dataset for prompt {new_batch.non_tensor_batch['uid'][i * N]}")
                                    # update response
                                    new_batch.batch["responses"][i * N] = new_batch.batch["tgt_input_ids"][i * N]
                                    prompt_ids = new_batch.batch["prompts"][i * N]
                                    prompt_length = prompt_ids.shape[-1]
                                    # update attention_mask
                                    new_batch.batch["attention_mask"][i * N][prompt_length:] = 0
                                    # 把responses中不等于pad_token_id的token的attention_mask设置为1
                                    new_batch.batch["attention_mask"][i * N][prompt_length:].masked_fill_(new_batch.batch["responses"][i * N] != self.tokenizer.pad_token_id, 1)
                                    # 把token_level_scores[i * N]中最后一个token的reward值设置为1.0
                                    valid_response_length = new_batch.batch["attention_mask"][i * N][prompt_length:].sum()
                                    new_batch.batch["token_level_scores"][i * N][valid_response_length - 1] = 1.0
                                    is_offline.extend([1] + [0] * (N - 1))
                                else:
                                    is_offline.extend([0] * N)
                            else:
                                is_offline.extend([0] * N)
                        is_offline = torch.tensor(is_offline, dtype=torch.int)
                        response_length = new_batch.batch['responses'].size(1)
                        prefix_mask = is_offline.unsqueeze(-1).repeat((1, response_length))
                        new_batch.batch["prefix_mask"] = prefix_mask

                    if not self.config.algorithm.filter_groups.enable:
                        if self.dynamic_sample_buffer is None:
                            batch = new_batch
                        else:
                            assert NotImplementedError, "dynamic buffer should be None when filter_groups is not enabled"
                            # batch = DataProto.concat([self.dynamic_sample_buffer, new_batch])
                            # self.dynamic_sample_buffer = None
                        # add solve_none and solve_all to metrics
                        uids = batch.non_tensor_batch["uid"]  # (bs * n,)
                        unique_uid = np.unique(uids)  # (bs,)
                        solve_none, solve_all = 0, 0
                        for uid in unique_uid:
                            uid_mask = uids == uid  # (bs * n,)
                            # reward_tensor[uid_mask] is a tensor of shape (n, seq_len)
                            uid_rewards = reward_tensor[uid_mask.astype(bool)].sum(dim=-1)
                            if (uid_rewards > 0.975).all():
                                solve_all += 1
                            elif (uid_rewards < 0.025).all():
                                solve_none += 1
                        metrics["batch/solve_none"] = solve_none / len(unique_uid)
                        metrics["batch/solve_all"] = solve_all / len(unique_uid)
                    else:  # NOTE: When prompts after filtering is less than train batch size, we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
                            if all([x > 0.975 for x in metric_vals]):
                                self.filtered_correct_number += 1
                            elif all([x < 0.025 for x in metric_vals]):
                                self.filtered_incorrect_number += 1

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        print(f"previous dynamic_sample_buffer_size={0 if self.dynamic_sample_buffer is None else len(self.dynamic_sample_buffer)}")
                        if batch is None:
                            if self.dynamic_sample_buffer is None:
                                batch = new_batch
                            else:
                                batch = DataProto.concat([self.dynamic_sample_buffer, new_batch])
                                num_prompt_in_batch += len(self.dynamic_sample_buffer) // self.config.actor_rollout_ref.rollout.n
                                self.dynamic_sample_buffer = None
                        else:
                            if self.dynamic_sample_buffer is None:
                                batch = DataProto.concat([batch, new_batch])
                            else:
                                batch = DataProto.concat([self.dynamic_sample_buffer, batch, new_batch])
                                num_prompt_in_batch += len(self.dynamic_sample_buffer) // self.config.actor_rollout_ref.rollout.n
                                self.dynamic_sample_buffer = None

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            print(f"{num_prompt_in_batch=} >= {prompt_bsz=}")
                            print(f"{num_gen_batches=}. Finish generating, start to train...")
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            off_bsz = int(traj_bsz * 0.25)
                            on_bsz = traj_bsz - off_bsz
                            self.dynamic_sample_buffer = batch[off_bsz: -on_bsz]
                            batch = DataProto.concat([batch[:off_bsz], batch[-on_bsz:]])
                            metrics["batch/dynamic_sample_buffer_size"] = len(self.dynamic_sample_buffer)
                            # add solve_none and solve_all to metrics
                            uids = batch.non_tensor_batch["uid"]  # (bs * n,)
                            unique_uid = np.unique(uids)  # (bs,)
                            solve_none, solve_all = 0, 0
                            _reward_tensor = batch.non_tensor_batch[self.config.algorithm.filter_groups.metric]
                            for uid in unique_uid:
                                uid_mask = uids == uid  # (bs * n,)
                                # reward_tensor[uid_mask] is a tensor of shape (n, seq_len)
                                uid_rewards = _reward_tensor[uid_mask.astype(bool)]
                                # print(f"uid_rewards={uid_rewards}")
                                if (uid_rewards > 0.975).all():
                                    solve_all += 1
                                elif (uid_rewards < 0.025).all():
                                    solve_none += 1
                            metrics["batch/solve_none"] = solve_none / len(unique_uid)
                            metrics["batch/solve_all"] = solve_all / len(unique_uid)
                            # ATTENTION: Here may be some little problems, as we save batch[traj_bsz:] for next train, but it"s ok because it"s just a monitor metric
                            metrics["batch/real_solve_none"] = (solve_none + self.filtered_incorrect_number) / (len(unique_uid) + self.filtered_correct_number + self.filtered_incorrect_number)
                            metrics["batch/real_solve_all"] = (solve_all + self.filtered_correct_number) / (len(unique_uid) + self.filtered_correct_number + self.filtered_incorrect_number)
                            metrics["batch/filtered_incorrect_number"] = self.filtered_incorrect_number
                            metrics["batch/filtered_correct_number"] = self.filtered_correct_number
                            metrics["batch/number_unique_uid"] = len(unique_uid)
                            # self.filtered_incorrect_number = 0
                            # self.filtered_correct_number = 0



                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if self.config.data.get("filter_garbled", False):
                        with _timer("is_garbled", timing_raw):
                            garbled_output = self.actor_rollout_wg.is_garbled(batch)
                            batch = batch.union(garbled_output)
                            metrics["batch/is_garbled_rate"] = batch.batch["is_garbled"].mean().item()
                            print("is garbled finished")
                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                data_metrics = compute_data_metrics(
                        batch=batch,
                        use_critic=self.use_critic,
                )
                data_metrics.update({
                    "filtered_correct_number": self.filtered_correct_number * self.config.actor_rollout_ref.rollout.n,
                    "filtered_incorrect_number": self.filtered_incorrect_number * self.config.actor_rollout_ref.rollout.n,
                })
                metrics.update(data_metrics)
                self.filtered_correct_number = 0
                self.filtered_incorrect_number = 0
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
