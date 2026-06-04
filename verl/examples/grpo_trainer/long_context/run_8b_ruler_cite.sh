set -x
# WORK_DIR=/local/apps/verl
WORK_DIR=./
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
RUN_NAME=$1
echo "RUN_NAME: $RUN_NAME"

SFT_MODEL_PATH=

# 保存用户外部传入的 CKPT_PATH (因为 env.sh 会覆盖 CKPT_PATH 变量)
USER_SPECIFIED_CKPT_PATH=$CKPT_PATH

export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

source ./examples/env.sh
#source ./examples/setup.sh
echo "TENSORBOARD_DIR: $TENSORBOARD_DIR"

# 确定 Checkpoint 路径
# 1. 如果用户外部传入了 CKPT_PATH，优先使用
if [ -n "$USER_SPECIFIED_CKPT_PATH" ]; then
    CKPT_PATH="$USER_SPECIFIED_CKPT_PATH"
    echo "Using user-specified CKPT_PATH: $CKPT_PATH"
else  
    CKPT_PATH="" #修改Checkpoint路径
    echo "Using auto-generated CKPT_PATH: $CKPT_PATH"
fi

echo "Checking permission for CKPT_PATH: $CKPT_PATH"

if mkdir -p "$CKPT_PATH" && touch "$CKPT_PATH/.test_write" && rm "$CKPT_PATH/.test_write"; then
    echo "SUCCESS: Write permission confirmed for $CKPT_PATH"
else
    echo "ERROR: Cannot write to $CKPT_PATH. Please check permissions or disk space."
    exit 1
fi

# 修改为你的数据集路径
TRAIN_FILES=""
VAL_FILES=""

source ./examples/proxy.sh
source ./examples/ray_start_robust.sh

mkdir -p ./rollout/${RUN_NAME}
rollout_mode="sync"
if [ "$rollout_mode" = "async" ]; then
    return_raw_chat="True"
    chat_scheduler=examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler
fi

# Args:
#   $1: RUN_NAME (required)
#   $2: BATCH_SIZE (optional, default 16)
#   $3: MINI_BATCH_SIZE (optional, default 4)
#   $4: SAVE_FREQ (optional, default 20)
BATCH_SIZE=${2:-16}
MINI_BATCH_SIZE=${3:-8}  # 从8降到4以减少OOM风险
SAVE_FREQ=${4:-10}
MAX_PROMPT_LENGTH=126000
MAX_RESPONSE_LENGTH=4096
MAX_NUM_BATCHED_TOKENS=$(($MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH))
ULYSSES_SEQUENCE_PARALLEL_SIZE=8
# 降低单GPU处理的token数量以避免OOM，原值为 MAX_NUM_BATCHED_TOKENS/ULYSSES_SEQUENCE_PARALLEL_SIZE=32768
MAX_NUM_BATCHED_TOKENS_PER_GPU=$(($MAX_NUM_BATCHED_TOKENS / $ULYSSES_SEQUENCE_PARALLEL_SIZE))
echo "MAX_NUM_BATCHED_TOKENS: $MAX_NUM_BATCHED_TOKENS"
echo "MAX_NUM_BATCHED_TOKENS_PER_GPU: $MAX_NUM_BATCHED_TOKENS_PER_GPU"
echo "ULYSSES_SEQUENCE_PARALLEL_SIZE: $ULYSSES_SEQUENCE_PARALLEL_SIZE"

if [ $RANK -eq 0 ]; then
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ++algorithm.norm_adv_by_std_in_grpo=False \
    ++algorithm.use_kl_in_reward=False \
    data.train_files="[$TRAIN_FILES]" \
    data.val_files="[$VAL_FILES]" \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    ++data.filter_overlong_prompts=True \
    ++data.filter_overlong_prompts_workers=8 \
    ++data.save_batch=False \
    ++data.seed=52314 \
    ++data.trust_remote_code=True \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    ++actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=4 \
    ++actor_rollout_ref.rollout.max_num_seqs=4 \
    ++actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_NUM_BATCHED_TOKENS_PER_GPU \
    ++actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_NUM_BATCHED_TOKENS_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.01 \
    ++actor_rollout_ref.actor.optim.use_mup=False \
    ++actor_rollout_ref.actor.clip_ratio_low=0.2 \
    ++actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    ++actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    ++actor_rollout_ref.actor.fsdp_config.param_offload=False \
    ++actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    ++actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    ++actor_rollout_ref.rollout.enforce_eager=True \
    ++actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.chat_scheduler=$chat_scheduler \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    ++actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    ++reward_model.parallel_reward=True \
    reward_model.reward_manager=dapo \
    algorithm.kl_ctrl.kl_coef=0.002 \
    ++trainer.balance_batch=True \
    ++reward_model.reward_kwargs.enable_loop_detection=True \
    ++reward_model.loop_penalty_strategy=zero_reward \
    ++trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.default_local_dir=$CKPT_PATH \
    trainer.logger=['tensorboard'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.rollout_data_dir=/user/jinzhensheng/rollout/${RUN_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=1000000 \
    trainer.total_epochs=3 \
    ${@:5}
fi

source ./examples/ray_end.sh
