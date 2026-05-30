#!/bin/bash
# 批量解压指定路径下的压缩文件

TARGET_DIR="/mnt/xlab-nas-2/vla_dataset/lerobot/Galaxea/lerobot"
MAX_JOBS=10  # 最大并发数

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory does not exist: $TARGET_DIR"
    exit 1
fi

cd "$TARGET_DIR" || exit 1

echo "=========================================="
echo "批量解压文件: $TARGET_DIR"
echo "=========================================="
echo ""

# 统计压缩文件数量
TAR_GZ_COUNT=$(find . -maxdepth 1 -type f \( -name "*.tar.gz" -o -name "*.tgz" \) | wc -l)
TOTAL=$TAR_GZ_COUNT

echo "找到 .tar.gz 文件: $TOTAL 个"
echo ""

if [ $TOTAL -eq 0 ]; then
    echo "No compressed files found."
    exit 0
fi

# 解压函数
extract_file() {
    local file="$1"
    local basename=$(basename "$file")
    local dirname="${basename%.tar.gz}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始解压: $basename"
    
    if [ -d "$dirname" ]; then
        echo "  ⚠️  目录已存在，跳过: $dirname"
    else
        mkdir -p "$dirname"
        tar -xzf "$file" -C "$dirname" 2>&1 | head -5
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "  ✅ 解压完成: $basename -> $dirname"
        else
            echo "  ❌ 解压失败: $basename"
            rm -rf "$dirname"
        fi
    fi
}

# 导出函数以便在后台使用
export -f extract_file

# 并发解压
echo "开始解压（最大并发数: $MAX_JOBS）..."
echo ""

# 处理 .tar.gz 文件
find . -maxdepth 1 -type f \( -name "*.tar.gz" -o -name "*.tgz" \) | while read -r file; do
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        wait -n
    done
    extract_file "$file" &
done

# 等待所有后台任务完成
wait

echo ""
echo "=========================================="
echo "解压完成！"
echo "=========================================="
