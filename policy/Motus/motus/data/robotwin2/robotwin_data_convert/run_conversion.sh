#!/bin/bash
# Auto conversion script for RobotWin dataset

echo "Starting RobotWin dataset conversion at $(date)"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Load Configuration from config.yml
# ============================================================================
CONFIG_FILE="${SCRIPT_DIR}/config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please create config.yml with required paths."
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"

# Parse YAML configuration (improved - remove comments and extra whitespace)
SOURCE_ROOT=$(grep "^source_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
TARGET_ROOT=$(grep "^target_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
MAX_WORKERS=$(grep "^max_workers:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
LOG_LEVEL=$(grep "^log_level:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
ENABLE_T5=$(grep "^enable_t5_embeddings:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Default values if not in config
MAX_WORKERS=${MAX_WORKERS:-"4"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}
ENABLE_T5=${ENABLE_T5:-"false"}

# ============================================================================
# Validation
# ============================================================================
if [ -z "$SOURCE_ROOT" ]; then
    echo "Error: source_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$TARGET_ROOT" ]; then
    echo "Error: target_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$SOURCE_ROOT" ]; then
    echo "Error: Source root not found: $SOURCE_ROOT"
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_ROOT"

echo "Configuration loaded successfully:"
echo "  Source Root: $SOURCE_ROOT"
echo "  Target Root: $TARGET_ROOT"
echo "  Max Workers: $MAX_WORKERS"
echo "  Log Level: $LOG_LEVEL"
echo "  T5 Embeddings: $ENABLE_T5"

# ============================================================================
# Environment Setup
# ============================================================================
# Check if required Python packages are available
echo "Checking Python environment..."

# Use the python from current environment (conda or system)
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: No Python interpreter found"
    exit 1
fi

# Check if we're in a conda environment
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
    echo "Python executable: $(which $PYTHON_CMD)"
else
    echo "Using system Python: $(which $PYTHON_CMD)"
fi

# Verify torch is available
if ! $PYTHON_CMD -c "import torch" &> /dev/null; then
    echo "Error: PyTorch not found in current Python environment"
    echo "Please ensure you're in the correct conda environment with PyTorch installed"
    echo "Current environment: ${CONDA_DEFAULT_ENV:-system}"
    echo "Python path: $(which $PYTHON_CMD)"
    exit 1
fi

echo "Python environment check passed"

# ============================================================================
# Pre-conversion Checks
# ============================================================================
echo "Performing pre-conversion checks..."

# Check disk space
SOURCE_SIZE=$(du -sb "$SOURCE_ROOT" 2>/dev/null | cut -f1 || echo "0")
TARGET_PARENT=$(dirname "$TARGET_ROOT")
AVAILABLE_SPACE=$(df "$TARGET_PARENT" | awk 'NR==2 {print $4*1024}')

if [ "$SOURCE_SIZE" -gt 0 ] && [ "$AVAILABLE_SPACE" -gt 0 ]; then
    # Estimate required space (videos typically 50-80% of HDF5 size)
    REQUIRED_SPACE=$((SOURCE_SIZE * 70 / 100))
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        echo "Warning: May not have enough disk space"
        echo "  Estimated required: $(numfmt --to=iec $REQUIRED_SPACE)"
        echo "  Available: $(numfmt --to=iec $AVAILABLE_SPACE)"
        
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "Disk space check passed"
    fi
fi

# Check write permissions
if [ ! -w "$TARGET_PARENT" ]; then
    echo "Error: No write permission for target directory: $TARGET_PARENT"
    exit 1
fi

echo "Pre-conversion checks completed"

# ============================================================================
# Main Conversion Process
# ============================================================================
echo "Starting dataset conversion..."

# Create log directory
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/conversion_${TIMESTAMP}.log"

echo "Logs will be saved to: $LOG_FILE"

# Set up verbose flag
VERBOSE_FLAG=""
if [ "$LOG_LEVEL" = "DEBUG" ]; then
    VERBOSE_FLAG="--verbose"
fi

# Run the conversion
echo "Executing conversion script..."
cd "$SCRIPT_DIR"

$PYTHON_CMD robotwin_converter.py \
    --config "$CONFIG_FILE" \
    $VERBOSE_FLAG \
    2>&1 | tee "$LOG_FILE"

CONVERSION_STATUS=${PIPESTATUS[0]}

# ============================================================================
# Post-conversion Processing
# ============================================================================
if [ $CONVERSION_STATUS -eq 0 ]; then
    echo "Conversion completed successfully!"
    
    # ========================================================================
    # Final Report
    # ========================================================================
    echo ""
    echo "=========================================="
    echo "CONVERSION SUMMARY"
    echo "=========================================="
    echo "Start time: $(head -n 1 "$LOG_FILE" | grep -o "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]" || echo "Unknown")"
    echo "End time: $(date +%H:%M:%S)"
    echo "Source: $SOURCE_ROOT"
    echo "Target: $TARGET_ROOT"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Count converted files
    if [ -d "$TARGET_ROOT" ]; then
        VIDEO_COUNT=$(find "$TARGET_ROOT" -name "*.mp4" | wc -l)
        QPOS_COUNT=$(find "$TARGET_ROOT" -name "*.pt" | wc -l)
        META_COUNT=$(find "$TARGET_ROOT" -name "*.txt" | wc -l)
        
        echo "Converted files:"
        echo "  Videos: $VIDEO_COUNT"
        echo "  QPos files: $QPOS_COUNT" 
        echo "  Meta files: $META_COUNT"
        
        if [ "$ENABLE_T5" = "true" ]; then
            T5_COUNT=$(find "$TARGET_ROOT" -path "*/umt5_wan/*.pt" | wc -l)
            echo "  T5 embeddings: $T5_COUNT"
        fi
        
        # Calculate total size
        TARGET_SIZE=$(du -sh "$TARGET_ROOT" 2>/dev/null | cut -f1)
        echo "  Total size: ${TARGET_SIZE:-Unknown}"
    fi
    
    echo ""
    echo "Conversion completed successfully!"
    echo "=========================================="
    
else
    echo ""
    echo "=========================================="
    echo "CONVERSION FAILED"
    echo "=========================================="
    echo "Exit code: $CONVERSION_STATUS"
    echo "Check log file for details: $LOG_FILE"
    echo "=========================================="
    exit $CONVERSION_STATUS
fi

# ============================================================================
# Optional: Cleanup and Optimization
# ============================================================================
if grep -q "cleanup_temp_files.*true" "$CONFIG_FILE" 2>/dev/null; then
    echo "Cleaning up temporary files..."
    find /tmp -name "robotwin_*" -mtime +1 -delete 2>/dev/null || true
fi

echo "Script completed at $(date)"