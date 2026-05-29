# ============================================
# Load personal environment configuration
# ============================================
if [ -f "$PWD/.env.personal" ]; then
  echo "[env] Loading .env.personal"
  source "$PWD/.env.personal"
fi
# ============================================
# Activate Conda environment
# ============================================
if [ -n "$CONDA_ROOT" ] && [ -n "$CONDA_ENV" ]; then
  echo "[conda] Activating environment from $CONDA_ROOT: $CONDA_ENV"
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

checkpoint_dir=""
PORT=""
norm_stats_json_path=""
USE_NORM=0   # Whether to use normalization consistent with training (0: disabled, 1: enabled)

# Parse parameters like --weight --port --norm --norm_stats_json_path
while [ $# -gt 0 ]; do
  case "$1" in
    --weight)
      checkpoint_dir="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --norm)
      USE_NORM=1
      shift 1
      ;;
    --norm_stats_json_path)
      norm_stats_json_path="$2"
      shift 2
      ;;
    --*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      # Ignore positional arguments, use explicit --xxx format
      shift 1
      ;;
  esac
done

if [ -z "$checkpoint_dir" ]; then
  echo "Usage: $0 --weight <path> [--port <port>] [--norm_stats_json_path <path>]" >&2
  exit 1
fi

# If checkpoint_dir contains model.pt, use it directly;
# otherwise default to using latest-unsharded under it as checkpoint path.
if [ -f "$checkpoint_dir/model.pt" ]; then
  : # Keep checkpoint_dir unchanged
else
  checkpoint_dir="$checkpoint_dir/latest-unsharded"
fi

if [ -n "$PORT" ]; then
  : # Use user-specified port
else
  # Randomly select an unused port as API server port
  PORT=$(python - << 'EOF'
import socket, random
for _ in range(100):
    port = random.randint(10000, 19999)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)
raise SystemExit(1)
EOF
)
fi

echo "Starting API server on port $PORT"
if [ "$USE_NORM" -eq 1 ]; then
  # Normalized inference: auto-derive norm_stats_json_path from checkpoint_dir/norm_stat.json or checkpoint_dir/../norm_stat.json
  # and pass to api_server.py; no longer pass --no_norm.
  cand1="${checkpoint_dir}/norm_stat.json"
  cand2="$(dirname "$checkpoint_dir")/norm_stat.json"
  if [ -f "$cand1" ]; then
    norm_stats_json_path="$cand1"
  elif [ -f "$cand2" ]; then
    norm_stats_json_path="$cand2"
  else
    echo "[norm] Error: norm_stat.json not found" >&2
    echo "  Tried path 1: $cand1" >&2
    echo "  Tried path 2: $cand2" >&2
    echo "  Please confirm that norm_stat.json was generated under save_folder during training" >&2
    exit 1
  fi

  echo "[norm] Using norm_stats_json_path: $norm_stats_json_path"
  python deploy/api_server.py \
    --checkpoint "$checkpoint_dir" \
    --norm_stats_json_path "$norm_stats_json_path" \
    --normalization_type bounds \
    --host 0.0.0.0 \
    --port "$PORT"
else
  # No normalization: maintain backward compatibility, use default/manually specified norm_stats_json_path, and explicitly disable normalization
  python deploy/api_server.py \
    --checkpoint "$checkpoint_dir" \
    --norm_stats_json_path "$norm_stats_json_path" \
    --normalization_type bounds \
    --host 0.0.0.0 \
    --port "$PORT" \
    --no_norm
fi