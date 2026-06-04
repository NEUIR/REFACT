unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONUNBUFFERED=1
export HF_TRUST_REMOTE_CODE="1"
# export NCCL_SOCKET_NTHREADS=2
# export NCCL_NSOCKS_PERTHREAD=8
# export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH=/local/apps/verl:$PYTHONPATH
export USE_FP8_GEMM=0
export NCCL_DEBUG=INFO
# export CUDA_LAUNCH_BLOCKING=1
export GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
# export VLLM_LOGGING_LEVEL=DEBUG
# export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO # TRACE or INFO
# export VLLM_TRACE_FUNCTION=1
# check IB status
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_CGRAPH_submit_timeout=3000
export RAY_CGRAPH_get_timeout=3000
echo "IB device info:"
ibv_devinfo
echo "PyTorch NCCL version:"
python -c "import torch; print(torch.cuda.nccl.version())"
echo "NCCL related environment variables:"
env | grep NCCL
env | grep -E "NCCL*|LD_LIBRARY_PATH"
#PROJECT_NAME='verl_grpo_minicpm4_longcontext'
PROJECT_NAME='verl_grpo_qwen3_longcontext_cite'
#EXPERIMENT_NAME='minicpm4_8b_scaling_qa_tailor'
EXPERIMENT_NAME='qwen3_8b-rl_cite'
#CKPT_PATH=/data/checkpoints/${JOB_UID}
CKPT_PATH=/user/jinzhensheng/RL_Qwen3_Models
mkdir -p $CKPT_PATH
export TENSORBOARD_DIR=/user/jinzhensheng/verl-long_context_xy_dev_qwen_dapo/examples/tensorboard_8B_Cite/$PROJECT_NAME/$EXPERIMENT_NAME
mkdir -p $TENSORBOARD_DIR

echo $PATH
echo $LD_LIBRARY_PATH
echo $NCCL_IB_HCA

ulimit -n 1048576

hostname -i
nvidia-smi
free -h

export TENSORBOARD_DIR=/user/jinzhensheng/verl-long_context_xy_dev_qwen_dapo/examples/tensorboard_8B_Cite/$PROJECT_NAME/$EXPERIMENT_NAME
mkdir -p $TENSORBOARD_DIR

export SWANLAB_API_KEY=86XgDZ8lOTIZqhEzPa26X
export SWANLAB_LOG_DIR=/data/tensorboard/swanlab
mkdir -p $SWANLAB_LOG_DIR
export SWANLAB_MODE=cloud
