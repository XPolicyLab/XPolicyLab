#!/usr/bin/env bash
# Shared SSH and rendezvous helpers; host rows are <ssh_host> <ip> <key>.

SSH_KEY_DIR=${SSH_KEY_DIR:-$HOME/.ssh}

# host_fields <line> -> "<ssh_host> <ip> <key>"
host_fields() { read -r h ip key _rest <<<"$1"; echo "$h $ip ${key:--}"; }

is_local_ip() { ip -4 -o addr show 2>/dev/null | grep -qw "$1"; }

# ssh_opts <key> -> ssh options for that node
ssh_opts() {
  local key=$1 opts="-o ConnectTimeout=15 -o LogLevel=ERROR"
  if [[ -n "$key" && "$key" != "-" ]]; then
    local path="$key"
    [[ "$path" == /* ]] || path="$SSH_KEY_DIR/$key"
    [[ -f "$path" ]] || { echo "missing ssh key $path" >&2; return 1; }
    opts="-i $path $opts"
  fi
  echo "$opts"
}

# node_exec <ssh_host> <ip> <key>  (script on stdin)
# Runs locally when the node is this machine, otherwise over ssh.
node_exec() {
  local ssh_host=$1 ip=$2 key=$3
  if is_local_ip "$ip"; then
    bash -s
  else
    # shellcheck disable=SC2046
    ssh $(ssh_opts "$key") "$ssh_host" "bash -s"
  fi
}

# Scan for a free rendezvous port on the master node.
pick_master_port() {
  local scan='for c in $(seq 29520 29599); do
      ss -ltn "sport = :$c" 2>/dev/null | grep -q LISTEN || { echo $c; exit 0; }
    done; exit 1'
  local port
  port=$(node_exec "$1" "$2" "$3" <<<"$scan")
  [[ -n "$port" ]] || return 1
  echo "$port"
}

# Load optional W&B credentials from the environment and forward them to nodes.
load_wandb_env() {
  if [[ -z "${WANDB_API_KEY:-}" && -n "${MMABC_WANDB_ENV:-}" ]]; then
    if [[ -f "$MMABC_WANDB_ENV" ]]; then
      set -a
      # shellcheck disable=SC1090
      source "$MMABC_WANDB_ENV"
      set +a
    else
      echo "wandb: MMABC_WANDB_ENV=$MMABC_WANDB_ENV does not exist" >&2
    fi
  fi
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    case " ${FORWARD_ENV:-} " in
      *" WANDB_API_KEY "*) : ;;
      *) FORWARD_ENV="${FORWARD_ENV:-} WANDB_API_KEY" ;;
    esac
    echo "wandb: WANDB_API_KEY set (forwarded to every node)"
  else
    echo "wandb: no WANDB_API_KEY (set it or MMABC_WANDB_ENV); logging to console only" >&2
  fi
}
