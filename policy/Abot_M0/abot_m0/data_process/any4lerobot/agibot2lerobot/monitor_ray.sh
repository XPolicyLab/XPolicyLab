#!/bin/bash
# Ray 集群监控脚本

echo "=== Ray 集群状态 ==="
ray status

echo ""
echo "=== Ray 资源使用情况 ==="
ray exec --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    # 如果有 ray exec 命令，可以使用
    echo "使用 'ray status' 查看详细信息"
else
    echo "运行 'ray status' 查看集群状态和资源使用"
fi

echo ""
echo "=== Python 进程 CPU 使用情况 ==="
ps aux | grep "[p]ython.*agibot_h5.py" | awk '{print "PID:", $2, "CPU%:", $3"%", "MEM%:", $4"%", "CMD:", $11, $12, $13}'

echo ""
echo "=== 总 CPU 使用统计 ==="
# 统计所有相关 Python 进程的 CPU 使用
total_cpu=$(ps aux | grep "[p]ython.*agibot_h5.py" | awk '{sum+=$3} END {print sum}')
echo "总 CPU 使用率: ${total_cpu}%"

# 统计进程数
process_count=$(ps aux | grep "[p]ython.*agibot_h5.py" | wc -l)
echo "运行中的任务进程数: $process_count"

echo ""
echo "=== 系统整体 CPU 使用 ==="
top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "CPU 空闲率:", $1"%", "使用率:", 100-$1"%"}'

