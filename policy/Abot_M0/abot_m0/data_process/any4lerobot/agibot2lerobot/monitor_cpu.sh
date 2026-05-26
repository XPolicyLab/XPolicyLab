#!/bin/bash
# CPU 使用情况详细监控脚本

echo "=========================================="
echo "  convert.sh CPU 使用情况监控"
echo "=========================================="
echo ""

# 1. 查找主进程
MAIN_PID=$(pgrep -f "python.*agibot_h5.py" | head -1)
if [ -z "$MAIN_PID" ]; then
    echo "❌ 未找到运行中的 agibot_h5.py 进程"
    echo "请确保 convert.sh 正在运行"
    exit 1
fi

echo "📌 主进程 PID: $MAIN_PID"
echo ""

# 2. 查看主进程及其子进程的 CPU 使用
echo "=== 进程树和 CPU 使用 ==="
pstree -p $MAIN_PID 2>/dev/null || ps -ef | grep $MAIN_PID
echo ""

# 3. 统计所有相关进程的 CPU 使用
echo "=== 所有相关 Python 进程 ==="
ps aux | grep "[p]ython.*agibot" | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    cmd=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
    echo "PID: $pid | CPU: ${cpu}% | MEM: ${mem}% | $cmd"
done
echo ""

# 4. 计算总 CPU 使用率
echo "=== CPU 使用统计 ==="
total_cpu=$(ps aux | grep "[p]ython.*agibot" | awk '{sum+=$3} END {print sum+0}')
process_count=$(ps aux | grep "[p]ython.*agibot" | wc -l)
echo "总进程数: $process_count"
echo "总 CPU 使用率: ${total_cpu}%"
echo ""

# 5. 估算使用的 CPU 核心数（假设每个核心 100%）
cores_used=$(echo "$total_cpu / 100" | bc -l 2>/dev/null || echo "$total_cpu" | awk '{printf "%.1f", $1/100}')
echo "估算使用的 CPU 核心数: ${cores_used} 核"
echo ""

# 6. 查看 Ray 状态（如果可用）
if command -v ray &> /dev/null; then
    echo "=== Ray 集群状态 ==="
    ray status 2>/dev/null | head -20
    echo ""
fi

# 7. 系统整体 CPU 使用
echo "=== 系统整体 CPU 使用 ==="
if command -v top &> /dev/null; then
    top -bn1 | grep "Cpu(s)" | head -1
elif command -v vmstat &> /dev/null; then
    vmstat 1 2 | tail -1 | awk '{print "CPU 使用率:", 100-$15"%"}'
fi
echo ""

# 8. 实时监控模式提示
echo "💡 实时监控提示:"
echo "   运行 'watch -n 2 ./monitor_cpu.sh' 可以每 2 秒刷新一次"
echo "   或者运行 'htop' 查看实时进程情况"














