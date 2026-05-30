#!/bin/bash
# 分析内存不足问题

echo "=========================================="
echo "  内存不足问题分析"
echo "=========================================="
echo ""

# 从错误日志分析
echo "从 output1.txt 可以看到："
echo "- 内存使用: 457.98GB / 480GB (95.4%)"
echo "- 有多个 ray::save_as_lerobot_dataset 进程在运行"
echo "- 每个进程占用 9-12GB 内存"
echo ""

# 计算当前配置下的问题
echo "=== 问题分析 ==="
echo ""

TOTAL_CPUS=128
TOTAL_MEM_GB=480
SAFE_MEM_GB=$((TOTAL_MEM_GB * 85 / 100))  # 85% 安全阈值
MEM_PER_TASK=12  # 从日志看，每个任务实际占用 9-12GB

echo "系统资源："
echo "  - 总 CPU: ${TOTAL_CPUS} 核"
echo "  - 总内存: ${TOTAL_MEM_GB}GB"
echo "  - 安全内存阈值（85%）: ${SAFE_MEM_GB}GB"
echo "  - 每个任务实际内存占用: ${MEM_PER_TASK}GB（从日志估算）"
echo ""

echo "=== 不同 cpus_per_task 配置的内存分析 ==="
echo ""

for CPUS_PER_TASK in 3 6 12 18 24; do
    MAX_PARALLEL=$((TOTAL_CPUS / CPUS_PER_TASK))
    TOTAL_MEM_NEEDED=$((MAX_PARALLEL * MEM_PER_TASK))
    
    if [ $TOTAL_MEM_NEEDED -le $SAFE_MEM_GB ]; then
        STATUS="✓ 安全"
    else
        STATUS="✗ 内存不足"
    fi
    
    printf "cpus_per_task=%2d: 最大并行任务数=%2d, 总内存需求=%4dGB %s\n" \
        $CPUS_PER_TASK $MAX_PARALLEL $TOTAL_MEM_NEEDED "$STATUS"
done

echo ""
echo "=== 问题原因 ==="
echo ""
echo "1. cpus_per_task=24 时："
echo "   - 理论上可以并行: 128/24 = 5 个任务"
echo "   - 5 个任务 × 12GB = 60GB（看起来安全）"
echo ""
echo "2. 但实际情况："
echo "   - 可能有其他任务在运行（如之前的 convert.sh）"
echo "   - Ray 集群可能被多个脚本共享"
echo "   - 系统和其他进程也需要内存"
echo "   - 实际内存使用接近 480GB 的 95%"
echo ""
echo "3. 从日志看："
echo "   - 至少有 10 个 ray::save_as_lerobot_dataset 进程"
echo "   - 说明有多个任务在并行运行"
echo "   - 总内存需求: 10 × 12GB = 120GB+"
echo "   - 加上系统内存，很容易超过阈值"
echo ""

echo "=== 解决方案 ==="
echo ""
echo "方案 1: 降低 cpus_per_task（推荐）"
echo "  --cpus-per-task 12 或更小"
echo "  - 优点: 安全，不会内存不足"
echo "  - 缺点: 可能略慢（但影响很小，因为主要瓶颈是 I/O）"
echo ""
echo "方案 2: 确保没有其他任务在运行"
echo "  - 检查 Ray 集群状态: ray status"
echo "  - 停止其他 convert.sh 脚本"
echo "  - 或者使用独立的 Ray 集群"
echo ""
echo "方案 3: 增加内存阈值（不推荐）"
echo "  export RAY_memory_usage_threshold=0.98"
echo "  - 风险: 可能导致系统 OOM"
echo ""

echo "=== 推荐配置 ==="
echo ""
echo "对于 convert1.sh（只处理 task_327）："
echo "  --cpus-per-task 12"
echo ""
echo "原因："
echo "  - 12 个 CPU 足够处理单个任务"
echo "  - 最多并行: 128/12 = 10 个任务"
echo "  - 内存需求: 10 × 12GB = 120GB（安全）"
echo "  - 即使有其他任务，也不会内存不足"
echo ""










