#!/bin/bash
# 检查当前运行的 Ray 任务

echo "=========================================="
echo "  当前 Ray 任务状态检查"
echo "=========================================="
echo ""

# 检查 Ray 是否运行
if ! command -v ray &> /dev/null; then
    echo "Ray 未安装或不在 PATH 中"
    exit 1
fi

# 检查 Ray 集群状态
echo "=== Ray 集群状态 ==="
ray status 2>/dev/null || echo "Ray 集群未运行"
echo ""

# 检查运行的进程
echo "=== 运行的 ray::save_as_lerobot_dataset 进程 ==="
ps aux | grep "ray::save_as_lerobot_dataset" | grep -v grep | wc -l | xargs echo "进程数量:"
echo ""

# 显示详细信息
echo "=== 进程详细信息 ==="
ps aux | grep "ray::save_as_lerobot_dataset" | grep -v grep | head -10
echo ""

# 检查内存使用
echo "=== 内存使用情况 ==="
free -h
echo ""

# 检查是否有其他 Python 脚本在运行
echo "=== 其他相关 Python 进程 ==="
ps aux | grep -E "(agibot_h5.py|convert)" | grep -v grep
echo ""

echo "=== 建议 ==="
echo "如果看到多个 ray::save_as_lerobot_dataset 进程："
echo "1. 检查是否有其他 convert.sh 脚本在运行"
echo "2. 考虑停止其他脚本，或使用独立的 Ray 集群"
echo "3. 降低 cpus_per_task 以确保内存安全"










