#!/bin/bash
# 快速查看 CPU 使用情况

echo "=========================================="
echo "  快速 CPU 监控"
echo "=========================================="

# 方法1: 使用 ps 查看所有相关进程（包括主进程和 worker）
echo "【方法1】主进程 CPU 使用:"
ps aux | grep "[p]ython.*agibot_h5.py" | grep -v "ray::" | awk '{printf "PID: %-8s CPU: %5s%% MEM: %5s%%\n", $2, $3, $4}'

echo ""
echo "【方法2】Ray Worker 进程统计:"
worker_count=$(ps aux | grep "ray::save_as_lerobot_dataset" | grep -v grep | wc -l)
worker_cpu=$(ps aux | grep "ray::save_as_lerobot_dataset" | grep -v grep | awk '{sum+=$3} END {print sum+0}')
worker_cores=$(echo "scale=1; $worker_cpu / 100" | bc 2>/dev/null || echo "$worker_cpu" | awk '{printf "%.1f", $1/100}')
echo "  Worker 数量: $worker_count"
echo "  Worker 总 CPU 使用率: ${worker_cpu}%"
echo "  Worker 使用核心数: ${worker_cores} 核"

echo ""
echo "【方法3】总 CPU 使用统计（主进程 + Worker）:"
total_cpu=$(ps aux | grep -E "[p]ython.*agibot|ray::save_as_lerobot_dataset" | grep -v grep | awk '{sum+=$3} END {print sum+0}')
process_count=$(ps aux | grep -E "[p]ython.*agibot|ray::save_as_lerobot_dataset" | grep -v grep | wc -l)
cores_used=$(echo "scale=1; $total_cpu / 100" | bc 2>/dev/null || echo "$total_cpu" | awk '{printf "%.1f", $1/100}')
echo "  总进程数: $process_count"
echo "  总 CPU 使用率: ${total_cpu}%"
echo "  总使用核心数: ${cores_used} 核"

echo ""
echo "【方法3】Ray 集群状态 (如果可用):"
ray status 2>/dev/null | grep -E "Resources|CPU" | head -5 || echo "  Ray 未运行或不可用"

echo ""
echo "💡 提示: 运行 'htop' 或 'top' 查看实时情况"

