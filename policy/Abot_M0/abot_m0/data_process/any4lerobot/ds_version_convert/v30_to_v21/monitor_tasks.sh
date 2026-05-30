#!/bin/bash
# 监控转换任务运行状态的脚本
# 可以在另一个终端运行此脚本来实时查看任务状态

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE_PATH="/mnt/workspace/vla_dataset/lerobot/agibot_convert_21/agibotworld"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 查找状态目录
STATUS_DIRS=$(find /tmp -maxdepth 1 -type d -name "convert_tasks_*" 2>/dev/null | sort -r)

if [ -z "$STATUS_DIRS" ]; then
    echo -e "${YELLOW}No conversion tasks found running.${NC}"
    echo ""
    echo "Method 1: Check Python processes"
    ps aux | grep -E "convert_v30_to_v21_simple.py" | grep -v grep | while read line; do
        pid=$(echo $line | awk '{print $2}')
        cmd=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
        task_id=$(echo "$cmd" | grep -oP '--task-id \K\S+' || echo "unknown")
        echo "  PID: $pid | Task: $task_id"
    done
    exit 0
fi

# 使用最新的状态目录
STATUS_DIR=$(echo "$STATUS_DIRS" | head -n1)
STATUS_INFO_FILE=$(find /tmp -maxdepth 1 -name "convert_status_*.info" 2>/dev/null | sort -r | head -n1)

if [ -f "$STATUS_INFO_FILE" ]; then
    source "$STATUS_INFO_FILE"
fi

echo -e "${BLUE}=== Conversion Task Monitor ===${NC}"
echo "Status directory: $STATUS_DIR"
echo ""

# 方法1: 从状态文件读取
if [ -d "$STATUS_DIR" ]; then
    running_tasks=()
    completed_tasks=()
    failed_tasks=()
    pending_tasks=()
    
    # 读取任务列表（从状态文件推断）
    if [ -f "$STATUS_INFO_FILE" ] && [ -n "$TOTAL" ]; then
        echo -e "${GREEN}Running Tasks:${NC}"
        for status_file in "$STATUS_DIR"/*.status; do
            if [ -f "$status_file" ]; then
                task_id=$(basename "$status_file" .status)
                status=$(cat "$status_file" 2>/dev/null || echo "UNKNOWN")
                
                case "$status" in
                    "RUNNING")
                        running_tasks+=("$task_id")
                        # 检查进程是否还在运行
                        if ps aux | grep -q "[p]ython3.*$SCRIPT_DIR/convert_v30_to_v21_simple.py.*--task-id $task_id"; then
                            echo -e "  ${YELLOW}[RUNNING]${NC} $task_id"
                        else
                            echo -e "  ${BLUE}[FINISHING]${NC} $task_id"
                        fi
                        ;;
                    "SUCCESS")
                        completed_tasks+=("$task_id")
                        ;;
                    "FAILED")
                        failed_tasks+=("$task_id")
                        ;;
                    *)
                        pending_tasks+=("$task_id")
                        ;;
                esac
            fi
        done
        
        echo ""
        echo -e "${GREEN}Completed: ${#completed_tasks[@]}${NC} | ${RED}Failed: ${#failed_tasks[@]}${NC} | ${YELLOW}Running: ${#running_tasks[@]}${NC} | Pending: ${#pending_tasks[@]}"
    else
        echo "Reading status files..."
        for status_file in "$STATUS_DIR"/*.status; do
            if [ -f "$status_file" ]; then
                task_id=$(basename "$status_file" .status)
                status=$(cat "$status_file" 2>/dev/null || echo "UNKNOWN")
                case "$status" in
                    "RUNNING")
                        echo -e "  ${YELLOW}[RUNNING]${NC} $task_id"
                        ;;
                    "SUCCESS")
                        echo -e "  ${GREEN}[DONE]${NC}    $task_id"
                        ;;
                    "FAILED")
                        echo -e "  ${RED}[FAILED]${NC}  $task_id"
                        ;;
                esac
            fi
        done
    fi
fi

echo ""
echo -e "${BLUE}=== Method 2: Python Processes ===${NC}"
ps aux | grep -E "convert_v30_to_v21_simple.py" | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    cmd=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
    task_id=$(echo "$cmd" | grep -oP '--task-id \K\S+' || echo "unknown")
    echo "  PID: $pid | CPU: ${cpu}% | MEM: ${mem}% | Task: $task_id"
done

echo ""
echo -e "${BLUE}=== Method 3: Completed Tasks (from markers) ===${NC}"
if [ -d "$OUTPUT_BASE_PATH" ]; then
    completed_count=0
    for task_dir in "$OUTPUT_BASE_PATH"/*; do
        if [ -d "$task_dir" ] && [ -f "$task_dir/.conversion_completed.json" ]; then
            ((completed_count++))
        fi
    done
    echo "Total completed: $completed_count"
fi

echo ""
echo -e "${YELLOW}Tip: Run this script repeatedly to monitor progress${NC}"
echo "Press Ctrl+C to exit"

