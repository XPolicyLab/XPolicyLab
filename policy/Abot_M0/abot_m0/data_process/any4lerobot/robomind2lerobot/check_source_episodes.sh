#!/bin/bash
# 检查 RoboMIND 原始格式中每个 task 的 episode 数量
# 分别测试不同的 benchmark 和 embodiment 组合

SCRIPT="check_source_episodes.py"
SRC_PATH="/mnt/nas-data-4/gaowo.cyz/RoboMIND"
OUTPUT_DIR="./source_episodes_reports"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "检查 RoboMIND 源数据 episode 数量"
echo "=========================================="
echo ""

# # Benchmark 1.0
# echo "检查 benchmark1_0_compressed..."
# python "$SCRIPT" \
#     --src-path "$SRC_PATH" \
#     --benchmarks benchmark1_0_compressed \
#     --embodiments agilex_3rgb franka_1rgb franka_3rgb simulation tienkung_gello_1rgb tienkung_xsens_1rgb ur_1rgb \
#     --output "$OUTPUT_DIR/benchmark1_0_compressed.txt" \
#     --output-format table

# echo ""

# Benchmark 1.1
echo "检查 benchmark1_1_compressed..."
python "$SCRIPT" \
    --src-path "$SRC_PATH" \
    --benchmarks benchmark1_1_compressed \
    --embodiments agilex_3rgb franka_fr3_dual sim_tienkung_1rgb tienkung_prod1_gello_1rgb ur_1rgb franka_3rgb sim_franka_3rgb tienkung_gello_1rgb tienkung_xsens_1rgb \
    --output "$OUTPUT_DIR/benchmark1_1_compressed.txt" \
    --output-format table

# echo ""

# # Benchmark 1.2
# echo "检查 benchmark1_2_compressed..."
# python "$SCRIPT" \
#     --src-path "$SRC_PATH" \
#     --benchmarks benchmark1_2_compressed \
#     --embodiments franka_3rgb sim_franka_3rgb \
#     --output "$OUTPUT_DIR/benchmark1_2_compressed.txt" \
#     --output-format table

# echo ""

# 生成汇总报告
echo "=========================================="
echo "生成汇总报告..."
echo "=========================================="

# 合并所有报告到一个文件
SUMMARY_FILE="$OUTPUT_DIR/all_benchmarks_summary.txt"
echo "RoboMIND 源数据 Episode 数量汇总报告" > "$SUMMARY_FILE"
echo "生成时间: $(date)" >> "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for report_file in "$OUTPUT_DIR"/benchmark*.txt; do
    if [ -f "$report_file" ]; then
        echo "=== $(basename "$report_file") ===" >> "$SUMMARY_FILE"
        cat "$report_file" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    fi
done

echo "所有报告已保存到: $OUTPUT_DIR"
echo "汇总报告: $SUMMARY_FILE"
