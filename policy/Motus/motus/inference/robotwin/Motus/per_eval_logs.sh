#!/bin/bash
# Parse evaluation logs and display task scores

# ============================================================================
# Configuration - UPDATE THIS PATH TO YOUR LOGS DIRECTORY
# ============================================================================
# Path to the logs directory you want to parse
# Log directory is created by auto_eval.sh as: policy/Motus/logs_YYYYMMDD_HHMMSS
LOG_DIR="..."

# ============================================================================
# Script starts here - No need to modify below
# ============================================================================

if [ -z "$LOG_DIR" ]; then
    echo "Error: LOG_DIR is not set."
    echo ""
    echo "Please edit this script and set LOG_DIR variable at the top."
    echo ""
    echo "Available log directories:"
    find "$(dirname "$0")" -maxdepth 1 -type d -name "logs_*" 2>/dev/null | sort -r | head -5
    exit 1
fi

if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory not found: $LOG_DIR"
    exit 1
fi

echo "================================================================"
echo "Parsing Evaluation Results"
echo "================================================================"
echo "Log Directory: $LOG_DIR"
echo ""

# Initialize counters
success_count=0
failed_count=0
total_score=0
task_count=0

# Arrays to store results
declare -A task_scores
declare -a task_names

# Parse each log file
for log_file in "$LOG_DIR"/*.log; do
    if [ ! -f "$log_file" ]; then
        continue
    fi
    
    task_name=$(basename "$log_file" .log)
    task_names+=("$task_name")
    
    # Try to extract success rate or score from log
    # Common patterns: "Success rate: XX/YY => XX.X%", "Success Rate: XX%", "success_rate: X.XX"
    score=""
    
    # Pattern 1: Success rate: XX/YY => XX.X% (with ANSI color codes)
    # Extract the percentage after "=>" from the last occurrence
    if grep -qi "Success rate:" "$log_file" 2>/dev/null; then
        # Get the last line with "Success rate:", remove ANSI codes, extract percentage
        last_line=$(grep -i "Success rate:" "$log_file" | tail -1 | sed 's/\x1b\[[0-9;]*m//g')
        # Extract percentage value after "=>" (e.g., "95.0%" or "95%")
        score=$(echo "$last_line" | grep -oP '=>\s*\K\d+\.?\d*(?=%)' | tail -1)
        if [ -z "$score" ]; then
            # Fallback: try to extract any percentage value on that line
            score=$(echo "$last_line" | grep -oP '\d+\.?\d*(?=%)' | tail -1)
        fi
    # Pattern 2: success_rate: X.XX (0-1 range)
    elif grep -q "success_rate:" "$log_file" 2>/dev/null; then
        score=$(grep "success_rate:" "$log_file" | tail -1 | grep -oP '\d+\.?\d*' | head -1)
        # Convert to percentage if it's 0-1 range
        if [ -n "$score" ]; then
            score=$(awk "BEGIN {printf \"%.1f\", $score * 100}")
        fi
    # Pattern 3: Check for failure markers (only if no success rate found)
    elif grep -q "failed with exit code\|Error:\|Traceback" "$log_file" 2>/dev/null; then
        score="0.0"
    else
        score="N/A"
    fi
    
    task_scores["$task_name"]="$score"
    
    # Count success/failure
    if [ "$score" != "N/A" ]; then
        ((task_count++))
        score_num=$(echo "$score" | sed 's/[^0-9.]//g')
        if [ -n "$score_num" ]; then
            total_score=$(awk "BEGIN {printf \"%.2f\", $total_score + $score_num}")
            if (( $(echo "$score_num >= 50.0" | bc -l) )); then
                ((success_count++))
            else
                ((failed_count++))
            fi
        fi
    fi
done

# Display results
echo "Task Results:"
echo "----------------------------------------------------------------"
printf "%-30s %10s\n" "Task Name" "Score"
echo "----------------------------------------------------------------"

for task_name in "${task_names[@]}"; do
    score="${task_scores[$task_name]}"
    
    # Color output based on score
    if [ "$score" = "N/A" ]; then
        printf "%-30s %10s\n" "$task_name" "N/A"
    else
        score_num=$(echo "$score" | sed 's/[^0-9.]//g')
        if (( $(echo "$score_num >= 80.0" | bc -l) )); then
            # Green for >= 80%
            printf "%-30s \033[32m%10.1f\033[0m%%\n" "$task_name" "$score_num"
        elif (( $(echo "$score_num >= 50.0" | bc -l) )); then
            # Yellow for >= 50%
            printf "%-30s \033[33m%10.1f\033[0m%%\n" "$task_name" "$score_num"
        else
            # Red for < 50%
            printf "%-30s \033[31m%10.1f\033[0m%%\n" "$task_name" "$score_num"
        fi
    fi
done

echo "----------------------------------------------------------------"

# Calculate average
if [ $task_count -gt 0 ]; then
    avg_score=$(awk "BEGIN {printf \"%.1f\", $total_score / $task_count}")
else
    avg_score="0.0"
fi

# Display summary
echo ""
echo "Summary Statistics:"
echo "----------------------------------------------------------------"
echo "Total Tasks:       $task_count"
echo "Success (>=50pct): $success_count"
echo "Failed (<50pct):   $failed_count"
echo "Average Score:     ${avg_score}%"
echo "================================================================"

# Create failed tasks file for re-run
failed_tasks_file="${LOG_DIR}/failed_tasks.txt"
> "$failed_tasks_file"

for task_name in "${task_names[@]}"; do
    score="${task_scores[$task_name]}"
    if [ "$score" != "N/A" ]; then
        score_num=$(echo "$score" | sed 's/[^0-9.]//g')
        if (( $(echo "$score_num < 50.0" | bc -l) )); then
            echo "$task_name" >> "$failed_tasks_file"
        fi
    fi
done

failed_count_file=$(wc -l < "$failed_tasks_file")
if [ $failed_count_file -gt 0 ]; then
    echo ""
    echo "Failed tasks saved to: $failed_tasks_file"
    echo "To re-run, copy to policy/Motus/ and set TASKS_FILE=\"failed_tasks.txt\""
fi

exit 0