#!/bin/bash
# 检查数据转换进度的便捷脚本
set -x
# 配置参数
SOURCE_JSON="/mnt/workspace/yangyandan/workspace/any4lerobot/robomind2lerobot/source_episodes_reports/all_benchmarks_summary.json"

# benchmark1_1_compressed 配置
BENCHMARK_11="benchmark1_1_compressed"
OUTPUT_PATH_11="/mnt/xlab-nas-2/vla_dataset/lerobot/robomind/"
EMBODIMENTS_11="agilex_3rgb franka_fr3_dual sim_tienkung_1rgb tienkung_prod1_gello_1rgb ur_1rgb franka_3rgb sim_franka_3rgb tienkung_gello_1rgb tienkung_xsens_1rgb"

# benchmark1_0_compressed 配置  
BENCHMARK_10="benchmark1_0_compressed"
OUTPUT_PATH_10="/mnt/xlab-nas-2/vla_dataset/lerobot/robomind_10_new_110"
EMBODIMENTS_10="agilex_3rgb franka_1rgb franka_3rgb simulation tienkung_gello_1rgb tienkung_xsens_1rgb ur_1rgb"

# benchmark1_2_compressed 配置
BENCHMARK_12="benchmark1_2_compressed"
OUTPUT_PATH_12="/mnt/xlab-nas-2/vla_dataset/lerobot/robomind_12"
EMBODIMENTS_12="franka_3rgb sim_franka_3rgb"

# 显示使用说明
show_usage() {
    echo "用法: $0 [benchmark]"
    echo ""
    echo "参数:"
    echo "  10 或 1.0    - 检查 benchmark1_0_compressed"
    echo "  11 或 1.1    - 检查 benchmark1_1_compressed (默认)"
    echo "  12 或 1.2    - 检查 benchmark1_2_compressed"
    echo "  all          - 检查所有 benchmarks"
    echo ""
    echo "示例:"
    echo "  $0           # 检查 benchmark1_1_compressed"
    echo "  $0 10        # 检查 benchmark1_0_compressed"
    echo "  $0 all       # 检查所有 benchmarks"
}

# 检查单个 benchmark
check_benchmark() {
    local benchmark=$1
    local output_path=$2
    shift 2
    local embodiments=("$@")
    
    echo "=========================================="
    echo "检查 ${benchmark}"
    echo "输出路径: ${output_path}"
    echo "=========================================="
    
    python check_conversion_progress.py \
        --source-json "${SOURCE_JSON}" \
        --output-path "${output_path}" \
        --benchmark "${benchmark}" \
        --embodiments "${embodiments[@]}"
    
    echo ""
}


# check_benchmark "${BENCHMARK_10}" "${OUTPUT_PATH_10}" ${EMBODIMENTS_10}
# echo ""
check_benchmark "${BENCHMARK_11}" "${OUTPUT_PATH_11}" ${EMBODIMENTS_11}
# echo ""
# check_benchmark "${BENCHMARK_12}" "${OUTPUT_PATH_12}" ${EMBODIMENTS_12}

