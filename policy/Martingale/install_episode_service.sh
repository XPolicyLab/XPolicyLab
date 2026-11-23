#!/usr/bin/env bash
# Bundle on a checkout; install on the Ubuntu Codex VM.
set -euo pipefail
adapter=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sources=(codex_cli config episode_protocol episode_server episode_client episode_smoke)
if [[ ${1:-} == --bundle ]]; then
    files=(install_episode_service.sh inference.json)
    for name in "${sources[@]}"; do
        files+=("runtime/src/martingale/inference/$name.py")
    done
    cat <<'HEADER'
#!/usr/bin/env bash
set -euo pipefail
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
base64 -d <<'MARTINGALE_BUNDLE' | tar -xz -C "$stage"
HEADER
    tar -C "$adapter" -czf - "${files[@]}" | base64
    printf '%s\n' 'MARTINGALE_BUNDLE' 'bash "$stage/install_episode_service.sh" --install'
    exit
fi
if [[ ${1:-} != --install || $EUID != 0 ]]; then
    echo "Bundle with --bundle on the checkout, or run --install as root on the VM." >&2
    exit 2
fi

state=/var/lib/martingale
dest=/opt/martingale-episode
id martingale >/dev/null
test -s "$state/endpoint.token"
curl -fsS --max-time 10 http://127.0.0.1:4500/readyz >/dev/null
apt-get install -y python3-venv
install -d -m 0755 "$dest/provider"
python3 -m venv "$dest/venv"
"$dest/venv/bin/python" -m pip install --disable-pip-version-check 'websockets==15.0.1'
for name in "${sources[@]}"; do
    install -m 0644 "$adapter/runtime/src/martingale/inference/$name.py" "$dest/provider/$name.py"
done
install -m 0644 "$adapter/inference.json" "$dest/inference.json"
runuser -u martingale -- sh -c 'umask 077; test -s /var/lib/martingale/robodojo.token || openssl rand -hex 32 > /var/lib/martingale/robodojo.token'
cat > /etc/systemd/system/martingale-episode.service <<'UNIT'
[Unit]
Description=RoboDojo episode service for Martingale
After=network-online.target martingale-codex.service
Wants=martingale-codex.service

[Service]
User=martingale
Group=martingale
WorkingDirectory=/opt/martingale-episode
RuntimeDirectory=martingale-episodes
RuntimeDirectoryMode=0700
ExecStart=/opt/martingale-episode/venv/bin/python -m provider.episode_server --config /opt/martingale-episode/inference.json --workspace-root /run/martingale-episodes
Restart=on-failure
RestartSec=3
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict
InaccessiblePaths=/var/lib/martingale/.codex

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable martingale-episode
systemctl restart martingale-episode
for attempt in {1..30}; do
    if [[ $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:4510/episode) == 401 ]]; then
        break
    fi
    sleep 1
done
if [[ $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:4510/episode) != 401 ]]; then
    journalctl -u martingale-episode -n 25 --no-pager >&2
    exit 1
fi
cd "$dest"
timeout 300 runuser -u martingale -- "$dest/venv/bin/python" -m provider.episode_smoke \
    --token-file "$state/robodojo.token" --config "$dest/inference.json"
