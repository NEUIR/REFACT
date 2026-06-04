#!/bin/bash

MASTER_IP=$(getent hosts $MASTER_ADDR | awk '{print $1}')
RAY_PORT=$(($MASTER_PORT + 1))
export RAY_ADDRESS="${MASTER_IP}:${RAY_PORT}"

# 添加重试机制的函数
start_ray_with_retry() {
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo "尝试启动Ray (第 $((retry_count + 1)) 次)"
        
        if [ $RANK -eq 0 ]; then
            # 启动head节点
            ray start --head \
            --port=$RAY_PORT \
            --num-gpus=${GPUS_PER_NODE} \
            --num-cpus=80 \
            --include-dashboard=false \
            --disable-usage-stats \
            --temp-dir=/tmp/ray_temp_${RANDOM}
            
            if [ $? -eq 0 ]; then
                echo "Head节点启动成功"
                break
            fi
        else
            # worker节点启动
            ray start --address="$RAY_ADDRESS" \
            --num-gpus=${GPUS_PER_NODE} \
            --num-cpus=80 \
            --temp-dir=/tmp/ray_temp_${RANDOM}
            
            if [ $? -eq 0 ]; then
                echo "Worker节点连接成功"
                break
            fi
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            echo "启动失败，等待5秒后重试..."
            sleep 5
            # 清理之前的Ray进程
            ray stop --force 2>/dev/null || true
        fi
    done
    
    if [ $retry_count -eq $max_retries ]; then
        echo "Ray启动失败，已尝试$max_retries次"
        exit 1
    fi
}

# 监控Ray状态的函数
monitor_ray() {
    while true; do
        sleep 30
        if ! ray status >/dev/null 2>&1; then
            echo "检测到Ray异常，尝试重启..."
            ray stop --force 2>/dev/null || true
            sleep 5
            start_ray_with_retry
        fi
    done
}

# 启动Ray
start_ray_with_retry

# 如果是worker节点，启动监控
if [ $RANK -ne 0 ]; then
    monitor_ray &
    MONITOR_PID=$!
    
    # 设置退出处理
    trap "kill $MONITOR_PID 2>/dev/null || true; ray stop --force" EXIT
    
    # 保持进程运行
    wait
else
    SLEEP_TIME=$((20 * WORLD_SIZE))
    if [ $SLEEP_TIME -gt 120 ]; then
        SLEEP_TIME=120
    fi
    sleep $SLEEP_TIME
    ray status
fi
