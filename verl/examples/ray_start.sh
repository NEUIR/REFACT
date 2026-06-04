MASTER_IP=$(getent hosts $MASTER_ADDR | awk '{print $1}')
# 兜底：如果解析失败就用 MASTER_ADDR 原值
MASTER_IP=${MASTER_IP:-$MASTER_ADDR}
RAY_PORT=$(($MASTER_PORT + 1))
export RAY_ADDRESS="${MASTER_IP}:${RAY_PORT}"

if [ "${RANK:-0}" -eq 0 ]; then
    ray start --head \
    --port=$RAY_PORT \
    --num-gpus=${GPUS_PER_NODE} \
    --num-cpus=80 \
    --include-dashboard=false \
    --disable-usage-stats
    echo "Head节点已就绪"
    sleep 10
    ray status
else
    ray start --address="$RAY_ADDRESS" \
    --num-gpus=${GPUS_PER_NODE} \
    --num-cpus=80 \
    --block
    echo "Worker节点已连接"
fi