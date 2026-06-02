#!/bin/bash

# Resize videos to 256x192 using ffmpeg
# Usage: ./preprocess.sh [dataset_name1] [dataset_name2] ...
# If no arguments, process all datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_BASE_DIR="${SCRIPT_DIR}/dataset"

if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg not found"
    exit 1
fi

process_dataset() {
    local dataset_name=$1
    local dataset_dir="${DATASET_BASE_DIR}/${dataset_name}"
    local videos_dir=""
    local output_dir=""
    
    if [ ! -d "$dataset_dir" ]; then
        echo "Warning: dataset directory not found: $dataset_dir"
        return 1
    fi

    if [ -d "${dataset_dir}/videos" ]; then
        videos_dir="${dataset_dir}/videos"
    elif [ -d "${dataset_dir}/video" ]; then
        videos_dir="${dataset_dir}/video"
    else
        mapfile -t matched_video_dirs < <(find "$dataset_dir" -type d \( -name "videos" -o -name "video" \) | sort)
        if [ ${#matched_video_dirs[@]} -eq 0 ]; then
            echo "Warning: no video/videos directory found under: $dataset_dir"
            return 1
        fi
        videos_dir="${matched_video_dirs[0]}"
    fi
    
    local resolved_dataset_name
    resolved_dataset_name="$(dirname "${videos_dir#${DATASET_BASE_DIR}/}")"
    output_dir="${DATASET_BASE_DIR}/${resolved_dataset_name}/videos_small"
    
    mapfile -t video_files < <(find "$videos_dir" -type f -name "*.mp4")
    local total_videos=${#video_files[@]}
    
    if [ $total_videos -eq 0 ]; then
        echo "No video files found in $dataset_name"
        return 0
    fi
    
    echo "Processing dataset: $resolved_dataset_name ($total_videos videos)"
    
    local processed=0
    local skipped=0
    local failed=0
    
    for i in "${!video_files[@]}"; do
        local video_path="${video_files[$i]}"
        local rel_path="${video_path#$videos_dir/}"
        local output_path="${output_dir}/${rel_path}"
        local output_dir_path="$(dirname "$output_path")"
        
        mkdir -p "$output_dir_path"
        
        local current=$((i + 1))
        local percent=$((current * 100 / total_videos))
        local bar_length=40
        local filled=$((current * bar_length / total_videos))
        
        local bar=""
        for ((j=0; j<bar_length; j++)); do
            if [ $j -lt $filled ]; then
                bar="${bar}█"
            else
                bar="${bar}░"
            fi
        done
        
        if [ -f "$output_path" ]; then
            skipped=$((skipped + 1))
            printf "\r[%3d%%] [%s] %d/%d | processed:%d skipped:%d failed:%d" \
                "$percent" "$bar" "$current" "$total_videos" "$processed" "$skipped" "$failed"
            continue
        fi
        
        printf "\r[%3d%%] [%s] %d/%d | processed:%d skipped:%d failed:%d" \
            "$percent" "$bar" "$current" "$total_videos" "$processed" "$skipped" "$failed"
        
        if ffmpeg -i "$video_path" \
            -vf "scale=256:192:force_original_aspect_ratio=decrease,pad=256:192:(ow-iw)/2:(oh-ih)/2" \
            -c:v libx264 -preset medium -crf 23 -y "$output_path" \
            -loglevel error 2>&1; then
            processed=$((processed + 1))
        else
            [ -f "$output_path" ] && rm -f "$output_path"
            failed=$((failed + 1))
        fi
        
        printf "\r[%3d%%] [%s] %d/%d | processed:%d skipped:%d failed:%d" \
            "$percent" "$bar" "$current" "$total_videos" "$processed" "$skipped" "$failed"
    done
    
    echo ""
    echo "Dataset $resolved_dataset_name: Total: $total_videos | Processed: $processed | Skipped: $skipped | Failed: $failed"
    echo ""
}

discover_dataset_names() {
    mapfile -t datasets < <(
        find "$DATASET_BASE_DIR" -type d \( -name "videos" -o -name "video" \) \
        | sed "s|^${DATASET_BASE_DIR}/||" \
        | xargs -r -I{} dirname "{}" \
        | sort -u
    )
}

discover_dataset_names_from_prefix() {
    local prefix="$1"
    local prefix_dir="${DATASET_BASE_DIR}/${prefix}"
    mapfile -t datasets < <(
        find "$prefix_dir" -type d \( -name "videos" -o -name "video" \) \
        | sed "s|^${DATASET_BASE_DIR}/||" \
        | xargs -r -I{} dirname "{}" \
        | sort -u
    )
}

if [ $# -eq 0 ]; then
    # Process all datasets
    if [ ! -d "$DATASET_BASE_DIR" ]; then
        echo "Error: dataset directory not found: $DATASET_BASE_DIR"
        exit 1
    fi
    
    discover_dataset_names
    
    if [ ${#datasets[@]} -eq 0 ]; then
        echo "No datasets with video/videos directory found in $DATASET_BASE_DIR"
        exit 0
    fi
    
    echo "Found ${#datasets[@]} dataset(s), processing all..."
    echo ""
    
    for dataset in "${datasets[@]}"; do
        process_dataset "$dataset"
    done
else
    # Process specified datasets
    for dataset_input in "$@"; do
        if [ ! -d "${DATASET_BASE_DIR}/${dataset_input}" ]; then
            echo "Warning: dataset path not found: ${DATASET_BASE_DIR}/${dataset_input}"
            continue
        fi
        
        discover_dataset_names_from_prefix "$dataset_input"
        if [ ${#datasets[@]} -eq 0 ]; then
            echo "Warning: no video/videos directory found under ${DATASET_BASE_DIR}/${dataset_input}"
            continue
        fi
        
        for dataset_name in "${datasets[@]}"; do
            process_dataset "$dataset_name"
        done
    done
fi
