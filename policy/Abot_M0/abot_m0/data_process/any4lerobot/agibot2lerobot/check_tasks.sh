#!/bin/bash
# 检查任务执行情况

echo "=========================================="
echo "  任务执行情况分析"
echo "=========================================="
echo ""

# 1. 总任务数
total_tasks=$(ls /mnt/nas-data-4/bearbee/AgiBotWorld-Beta/task_info/*.json 2>/dev/null | wc -l)
echo "📊 总任务数: $total_tasks"

# 2. 已完成任务数
completed_tasks=$(find /mnt/nas-data-1/yangyandan/lerobot/agibot/agibotworld -maxdepth 1 -type d 2>/dev/null | wc -l)
completed_tasks=$((completed_tasks - 1))  # 减去根目录本身
echo "✅ 已完成任务数: $completed_tasks"

# 3. 剩余任务数
remaining=$((total_tasks - completed_tasks))
echo "⏳ 剩余任务数: $remaining"
echo ""

# 4. Ray 资源使用
echo "=== Ray 资源使用 ==="
ray status 2>/dev/null | grep -A 5 "Total Usage" || echo "Ray 未运行"
echo ""

# 5. 当前运行的 Ray worker 任务数
echo "=== 当前运行的任务数分析 ==="
ray_cpu_used=$(ray status 2>/dev/null | grep "Total Usage" | grep -oE "[0-9]+\.[0-9]+/[0-9]+\.[0-9]+ CPU" | cut -d'/' -f1)
if [ -n "$ray_cpu_used" ]; then
    # 计算运行的任务数（每个任务需要 3 个 CPU）
    running_tasks=$(echo "scale=0; $ray_cpu_used / 3" | bc 2>/dev/null || echo "计算中...")
    echo "Ray 使用的 CPU: ${ray_cpu_used} 核"
    echo "当前运行的任务数: ${running_tasks} 个（每个任务 3 核）"
    echo ""
    echo "💡 说明:"
    echo "   - 理论上最多可以并行: ⌊128/3⌋ = 42 个任务"
    echo "   - 当前只运行了 ${running_tasks} 个任务"
    if [ "$remaining" -gt 0 ]; then
        echo "   - 还有 ${remaining} 个任务需要处理"
        echo "   - 可能原因: 剩余任务较少，或者任务执行很快"
    else
        echo "   - 所有任务已完成！"
    fi
else
    echo "无法获取 Ray 状态"
fi














