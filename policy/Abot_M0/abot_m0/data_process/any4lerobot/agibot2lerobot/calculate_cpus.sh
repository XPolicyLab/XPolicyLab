#!/bin/bash
# 计算合理的 cpus_per_task 值

echo "=== 系统资源检查 ==="
echo ""

# 获取 CPU 核心数
TOTAL_CPUS=$(nproc)
echo "总 CPU 核心数: $TOTAL_CPUS"

# 获取总内存（GB）
TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "总内存: ${TOTAL_MEM_GB}GB"

# 安全内存使用量（留 20% 余量）
SAFE_MEM_GB=$((TOTAL_MEM_GB * 80 / 100))
echo "安全可用内存（80%阈值）: ${SAFE_MEM_GB}GB"
echo ""

echo "=== 不同 cpus_per_task 配置分析 ==="
echo "（假设每个任务需要 20GB 内存）"
echo ""

MEM_PER_TASK=20  # 每个任务需要的内存（GB）

for CPUS_PER_TASK in 3 6 9 12 15 18 21 24; do
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
echo "=== 推荐配置 ==="
echo "根据 README，推荐使用 3 CPU cores per task"
echo "如果内存充足，可以适当增加以提高单个任务的处理速度"
echo ""
echo "建议："
echo "- 如果只处理少量任务（如 task_327），可以设置较大值（12-18）"
echo "- 如果处理大量任务，建议使用较小值（3-6）以避免内存不足"










