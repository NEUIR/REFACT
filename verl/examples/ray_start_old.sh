MASTER_IP=$(getent hosts $MASTER_ADDR | awk '{print $1}')
RAY_PORT=$(($MASTER_PORT + 1))
export RAY_ADDRESS="${MASTER_IP}:${RAY_PORT}"

if [ $RANK -eq 0 ]; then
    # 启动head节点
    ray start --head \
    --port=$RAY_PORT \
    --num-gpus=${GPUS_PER_NODE} \
    --num-cpus=80 \
    --include-dashboard=false \
    --disable-usage-stats
    # 等待head节点就绪
    echo "Head节点已就绪"
else
    # worker节点等待head节点就绪
    # 启动worker节点
    ray start --address="$RAY_ADDRESS" \
    --num-gpus=${GPUS_PER_NODE} \
    --num-cpus=80 \
    --block
    echo "Worker节点已连接"
fi
SLEEP_TIME=$((20 * WORLD_SIZE))
if [ $SLEEP_TIME -gt 120 ]; then
    SLEEP_TIME=120
fi
sleep $SLEEP_TIME
ray status