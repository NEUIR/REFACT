set -x
WORK_DIR=/local/apps/verl
source $WORK_DIR/examples/env.sh
SFT_MODEL_PATH=/user/linbiyuan/models/Qwen3-1.7B
source $WORK_DIR/examples/setup.sh

TRAIN_FILES="/user/xuxiaoyue/rldata/qa_tailor/train.parquet"

VAL_FILES="/user/xuxiaoyue/rldata/qa_tailor/test.parquet"

source $WORK_DIR/examples/proxy.sh
source $WORK_DIR/examples/ray_start.sh

# Environment check for Ulysses sequence parallel
ULYSSES_SP_SIZE=1
if [ ! -z "$WORLD_SIZE" ] && [ "$WORLD_SIZE" -ge 2 ]; then
    ULYSSES_SP_SIZE=2
fi
echo "Using Ulysses sequence parallel size: $ULYSSES_SP_SIZE"

# For async rollout mode, dataset should return raw chat.
rollout_mode="sync"
if [ "$rollout_mode" = "async" ]; then
    return_raw_chat="True"
    chat_scheduler=examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler
fi

if [ $RANK -eq 0 ]; then
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ++algorithm.use_kl_in_reward=False \
    ++data.filter_overlong_prompts=True \
    ++data.save_batch=False \
    ++data.seed=52314 \
    data.train_files="[$TRAIN_FILES]" \
    data.val_files="[$VAL_FILES]" \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=32768 \
    data.max_response_length=8192 \
    ++data.trust_remote_code=True \
    ++data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    ++actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    ++actor_rollout_ref.actor.clip_ratio_low=0.2 \
    ++actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    ++actor_rollout_ref.actor.entropy_coeff=0.0 \
    ++actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.actor.fsdp_config.param_offload=False \
    ++actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    ++actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SP_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    ++actor_rollout_ref.rollout.enforce_eager=False \
    ++actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.chat_scheduler=$chat_scheduler \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    ++actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
    ++actor_rollout_ref.rollout.disable_log_stats=False \
    ++actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.ppo_max_token_len_per_gpu=81920 \
    reward_model.enable=False \
    ++reward_model.parallel_reward=True \
    reward_model.reward_manager=dapo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    ++trainer.balance_batch=True \
    ++trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.default_local_dir=$CKPT_PATH \
    trainer.logger=['console','tensorboard','swanlab'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=50 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=100 $@
fi

source $WORK_DIR/examples/ray_end.sh