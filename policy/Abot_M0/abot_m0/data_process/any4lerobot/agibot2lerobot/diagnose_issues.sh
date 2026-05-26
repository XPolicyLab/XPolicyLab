#!/bin/bash
# 诊断脚本：检查任务卡住和异常的原因

echo "=========================================="
echo "  任务诊断报告"
echo "=========================================="
echo ""

# 1. 检查主进程状态
echo "【1】主进程状态:"
MAIN_PID=$(pgrep -f "python.*agibot_h5.py" | head -1)
if [ -n "$MAIN_PID" ]; then
    ps -p $MAIN_PID -o pid,pcpu,pmem,etime,state,cmd
    echo "  运行时间: $(ps -p $MAIN_PID -o etime --no-headers)"
else
    echo "  ❌ 主进程未运行"
fi
echo ""

# 2. 检查 Ray worker 进程
echo "【2】Ray Worker 进程:"
WORKER_COUNT=$(pgrep -f "ray::save_as_lerobot_dataset" | wc -l)
echo "  运行中的 worker 数: $WORKER_COUNT"
if [ $WORKER_COUNT -gt 0 ]; then
    echo "  Worker PIDs:"
    pgrep -f "ray::save_as_lerobot_dataset" | while read pid; do
        ps -p $pid -o pid,pcpu,pmem,etime,state --no-headers 2>/dev/null | awk '{print "    PID:", $1, "CPU:", $2"%", "MEM:", $3"%", "TIME:", $4, "STATE:", $5}'
    done
else
    echo "  ⚠️  没有运行中的 worker"
fi
echo ""

# 3. 检查错误日志
echo "【3】最近的错误 (output.txt 最后 5 个):"
if [ -f "output.txt" ]; then
    tail -20 output.txt | grep -E "task_|Exception|Error" | tail -5
    ERROR_COUNT=$(grep -c "exception details" output.txt 2>/dev/null || echo "0")
    echo "  总错误数: $ERROR_COUNT"
else
    echo "  ⚠️  output.txt 不存在"
fi
echo ""

# 4. 检查失败的任务类型
echo "【4】失败任务统计:"
if [ -f "output.txt" ]; then
    echo "  numpy 导入错误: $(grep -c "numpy.core\|Error importing numpy" output.txt 2>/dev/null || echo "0")"
    echo "  文件损坏错误: $(grep -c "file signature not found\|Unable to synchronously open" output.txt 2>/dev/null || echo "0")"
fi
echo ""

# 5. 检查 conda 环境
echo "【5】Conda 环境:"
if command -v conda &> /dev/null; then
    CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
    echo "  当前环境: $CURRENT_ENV"
    echo "  lerobot 环境路径: $(conda env list | grep lerobot | awk '{print $2}')"
else
    echo "  ⚠️  conda 未安装或未在 PATH 中"
fi
echo ""

# 6. 检查 Python 路径
echo "【6】Python 环境:"
echo "  Python 路径: $(which python)"
echo "  Python 版本: $(python --version 2>&1)"
echo "  PYTHONPATH: ${PYTHONPATH:-未设置}"
echo ""

# 7. 检查 Ray 状态
echo "【7】Ray 集群状态:"
if command -v ray &> /dev/null; then
    ray status 2>/dev/null | head -15 || echo "  ⚠️  无法获取 Ray 状态"
else
    echo "  ⚠️  ray 命令不可用"
fi
echo ""

# 8. 任务完成情况
echo "【8】任务完成情况:"
TOTAL_TASKS=$(ls /mnt/nas-data-4/bearbee/AgiBotWorld-Beta/task_info/*.json 2>/dev/null | wc -l)
COMPLETED=$(find /mnt/nas-data-1/yangyandan/lerobot/agibot/agibotworld -maxdepth 1 -type d 2>/dev/null | wc -l)
COMPLETED=$((COMPLETED - 1))
REMAINING=$((TOTAL_TASKS - COMPLETED))
echo "  总任务数: $TOTAL_TASKS"
echo "  已完成: $COMPLETED"
echo "  剩余: $REMAINING"
echo "  完成率: $(echo "scale=1; $COMPLETED * 100 / $TOTAL_TASKS" | bc 2>/dev/null || echo "计算中")%"
echo ""

echo "=========================================="
echo "  诊断完成"
echo "=========================================="
