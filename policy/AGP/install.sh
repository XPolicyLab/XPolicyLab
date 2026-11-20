#!/usr/bin/env bash
# AGP environment setup on a machine with the official RoboDojo checkout and its Assets; nothing is installed into the
# checkout.
#
#   bash install.sh [--check] [--steps venv-policy,venv-agent,codex,venv-sim]
#
#   --check, --dry-run   report what is present and what would be done; change nothing
#   --steps LIST         comma-separated subset of venv-policy, venv-agent, codex, venv-sim (default: venv-policy,
#                        venv-agent, codex)
#   -h, --help           this text
#
# Steps: venv-policy ($AGP_HOME/.venv-policy, the stub policy server's interpreter), venv-agent ($AGP_HOME/.venv-agent,
# the MCP server's interpreter), codex (the pinned release 0.160.1 at deploy.yml agp.codex_binary, sha256-checked),
# venv-sim (only with --steps: $AGP_HOME/.venv-sim, the simulator runtime for a machine without RoboDojo's conda
# environment; installing Isaac Sim accepts NVIDIA's Omniverse EULA, as the official RoboDojo installer does).
# It also reports, without installing anything, what needs an administrator: the NVIDIA driver, the Vulkan ICD, the
# system libraries Isaac Sim loads, /usr/bin/python3 and the checkout's Assets, and whether a codex login exists.
#
# Environment:
#   AGP_HOME          default ~/.agp-robodojo: the venvs and everything the runtime writes (never inside the checkout)
#   AGP_SETUP_DIR     default .agp-setup next to AGP_HOME: uv and the CPython build the venvs run on (keep this folder)
#   AGP_RUNTIME_LIBS  system libraries unpacked without root
#
# Idempotent. Needs about 2 GB of free disk (about 50 GB with venv-sim), curl or wget, tar, sha256sum, git and
# /usr/bin/python3. Exit status: 0 when every selected step is done and nothing the runtime needs is missing; 1
# otherwise; 2 on usage.
set -Eeuo pipefail

PY_VERSION="3.11.16"
UV_VERSION="0.12.17"
UV_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
UV_SHA256="fa82fd8dde8e8eefdecada6aa0889666556cfceb690d06e0c3bca49eb3070a63"
CODEX_VERSION="0.160.1"
CODEX_TARGET="x86_64-unknown-linux-musl"
CODEX_ASSET="codex-package-${CODEX_TARGET}.tar.gz"
CODEX_URLS=("https://github.com/openai/codex/releases/download/rust-v${CODEX_VERSION}/${CODEX_ASSET}"
            "https://releases.openai.com/codex/releases/${CODEX_VERSION}/${CODEX_ASSET}")
CODEX_ASSET_SHA256="340801565906a7028f6baaa9ab6853addaef221f0016a1417a7c1ffdd96c21f0"
CODEX_BIN_SHA256="f34a4d2301892ae96c90097786bfe5dc269f187b6f69faf42a7b357b8c081e35"
CODEX_HOST_SHA256="b33e8a5283f3c65c2a0aca6d43a59cfe850f624d8fa16992e3cad4fcc27c14e1"
STEPS_ALL=(venv-policy venv-agent codex venv-sim)
STEPS_DEFAULT=(venv-policy venv-agent codex)
SYSPY=/usr/bin/python3
# soname:Ubuntu package that Isaac Sim 5.1 needs for headless rendering
SYSLIBS=(libGLU.so.1:libglu1-mesa libxml2.so.2:libxml2 libXt.so.6:libxt6 libGL.so.1:libgl1 libEGL.so.1:libegl1
         libOpenGL.so.0:libopengl0 libvulkan.so.1:libvulkan1 libglib-2.0.so.0:libglib2.0-0 libX11.so.6:libx11-6
         libXext.so.6:libxext6 libXrender.so.1:libxrender1 libSM.so.6:libsm6 libICE.so.6:libice6
         libXrandr.so.2:libxrandr2 libXi.so.6:libxi6 libXcursor.so.1:libxcursor1 libXinerama.so.1:libxinerama1
         libgomp.so.1:libgomp1)
VULKAN_ICDS=(/usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json)

usage() { sed -n '2,/^set -Eeuo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

CHECK=0; STEPS="default"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check|--dry-run) CHECK=1; shift ;;
        --steps) STEPS="${2:?--steps needs a value}"; shift 2 ;;
        --steps=*) STEPS="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done
if [[ "${STEPS}" == default ]]; then
    SELECTED=("${STEPS_DEFAULT[@]}")
else
    IFS=',' read -r -a SELECTED <<< "${STEPS}"
    for s in "${SELECTED[@]}"; do
        [[ " ${STEPS_ALL[*]} " == *" ${s} "* ]] || { echo "unknown step: ${s} (steps: ${STEPS_ALL[*]})" >&2; exit 2; }
    done
fi

# ------------------------------------------------------------------------------------------------------- helpers
ts() { date -u +%H:%M:%SZ; }
log() { printf '%s [install] %s\n' "$(ts)" "$*"; }
die() { printf '%s [install] FAILED: %s\n' "$(ts)" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
sha256() { sha256sum "$1" | cut -d' ' -f1; }
trap 'rc=$?; printf "%s [install] FAILED: line %s (exit %s)\n" "$(ts)" "${LINENO}" "${rc}" >&2' ERR
MISSING=()
ok() { printf '  ok       %-11s %s\n' "$1" "$2"; }
missing() { printf '  MISSING  %-11s %s\n' "$1" "$2"; MISSING+=("$1"); }
note() { printf '  note     %-11s %s\n' "$1" "$2"; }
inside() {   # inside PATH ROOT: PATH (resolved; it need not exist) is ROOT or lies in it
    local p r
    p="$(realpath -m -- "$1")"; r="$(realpath -m -- "$2")"
    [[ "${p}" == "${r}" || "${p}" == "${r}"/* ]]
}
abs_path() {   # a leading ~ is the home folder (as agp_core/paths.py reads AGP_HOME); relative paths are taken from here
    local p="$1"
    [[ "${p}" == "~" ]] && p="${HOME}"
    [[ "${p}" == "~/"* ]] && p="${HOME}/${p#\~/}"
    realpath -m -- "${p}"
}
existing_parent() {   # the nearest existing folder at or above a path
    local p; p="$(realpath -m -- "$1")"
    while [[ ! -d "${p}" ]]; do p="$(dirname "${p}")"; done
    echo "${p}"
}

# -------------------------------------------------------------------------------------------------------- layout
POLICY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
XPL="$(cd "${POLICY}/../.." && pwd -P)"
C="$(cd "${XPL}/.." && pwd -P)"                       # the official RoboDojo checkout
RUNTIME="${POLICY}/agp_runtime"
SIM_FREEZE="${RUNTIME}/env/sim/requirements-freeze.txt"
AGENT_FREEZE="${RUNTIME}/env/agent/requirements-freeze.txt"
POLICY_FREEZE="${RUNTIME}/env/policy/requirements-freeze.txt"
VERSIONS="${RUNTIME}/env/sim/versions.json"
export AGP_HOME="${AGP_HOME:-${HOME}/.agp-robodojo}"
AGP_HOME="$(abs_path "${AGP_HOME}")"
SETUP="$(abs_path "${AGP_SETUP_DIR:-$(dirname "${AGP_HOME}")/.agp-setup}")"
UV_BIN_DIR="${SETUP}/bin"
for f in "${SIM_FREEZE}" "${AGENT_FREEZE}" "${POLICY_FREEZE}" "${VERSIONS}" "${POLICY}/deploy.yml"; do
    [[ -f "${f}" ]] || die "${f} is missing: this policy folder is incomplete"
done
[[ -f "${C}/src/eval_client/main.py" && -d "${C}/third_party/IsaacLab/source/isaaclab" \
    && -d "${C}/third_party/curobo/curobo" ]] \
    || die "${C} is not a RoboDojo checkout with its submodules (install.sh must stay in XPolicyLab/policy/AGP)"
[[ -x "${SYSPY}" ]] || die "${SYSPY} is required (it reads the pin files here; the policy's freeze and account tools run on it)"
# deploy.yml agp.codex_binary, read without PyYAML
codex_pin="$(sed -n '/^[[:space:]]*codex_binary:/{s/^[[:space:]]*codex_binary:[[:space:]]*\([^[:space:]#]*\).*/\1/p;q;}' \
    "${POLICY}/deploy.yml")"
[[ -n "${codex_pin}" ]] || die "deploy.yml has no agp.codex_binary"
CODEX_BIN="$(abs_path "${codex_pin}")"
CODEX_DIR="$(dirname "$(dirname "${CODEX_BIN}")")"
[[ "$(basename "${CODEX_DIR}")" == "${CODEX_VERSION}-${CODEX_TARGET}" && "$(basename "${CODEX_BIN}")" == codex ]] \
    || die "deploy.yml agp.codex_binary (${codex_pin}) is not the release ${CODEX_VERSION}-${CODEX_TARGET} this script pins"

# Nothing is written into the checkout (this policy folder included): refuse every target inside it.
for target in "AGP_HOME=${AGP_HOME}" "AGP_SETUP_DIR=${SETUP}" "codex release=${CODEX_DIR}"; do
    if inside "${target#*=}" "${C}"; then
        die "${target%%=*} ${target#*=} lies in the checkout ${C}: install.sh writes nothing there (set it elsewhere)"
    fi
done
if inside "${SETUP}" "${AGP_HOME}"; then
    die "AGP_SETUP_DIR ${SETUP} lies in AGP_HOME ${AGP_HOME}: put it next to AGP_HOME (default: $(dirname "${AGP_HOME}")/.agp-setup)"
fi

json_get() {   # json_get FILE PYTHON-EXPRESSION-ON-d
    "${SYSPY}" -B -I -c "import json,sys; d=json.load(open(sys.argv[1])); print($2)" "$1"
}

# Isolation from the caller's environment: uv, Python and pip settings come from here only.
for v in $(compgen -e | grep -E '^(UV_|PIP_|CONDA_)' || true); do unset "${v}"; done
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV PYTHONSTARTUP
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="${SETUP}/uv-cache" UV_PYTHON_INSTALL_DIR="${SETUP}/python" UV_PYTHON_PREFERENCE=only-managed
export UV_NO_CONFIG=1 UV_HTTP_TIMEOUT=300 UV_PYTHON_DOWNLOADS=automatic UV_PYTHON_BIN_DIR="${SETUP}/python-bin"

fetch() {   # fetch URL OUT: download with retries to OUT (through OUT.part)
    local url="$1" out="$2"
    rm -f "${out}.part"
    if have curl; then
        curl -fL --retry 5 --retry-delay 10 --connect-timeout 30 -sS -o "${out}.part" "${url}" || return 1
    elif have wget; then
        wget -q --tries=5 --waitretry=10 --timeout=60 -O "${out}.part" "${url}" || return 1
    else
        die "neither curl nor wget is installed"
    fi
    mv -f "${out}.part" "${out}"
}

# ------------------------------------------------------------------------------------------------------------ uv
UV=""
find_uv() {   # the pinned uv: AGP_SETUP_DIR's, else one on PATH that reports exactly that version
    local candidate version
    for candidate in "${UV_BIN_DIR}/uv" "$(command -v uv || true)"; do
        [[ -n "${candidate}" && -x "${candidate}" ]] || continue
        version="$("${candidate}" --version 2>/dev/null || true)"
        version="${version%%$'\n'*}"
        if [[ "${version}" == "uv ${UV_VERSION}" || "${version}" == "uv ${UV_VERSION} "* ]]; then
            UV="${candidate}"
            return 0
        fi
    done
    return 1
}

ensure_uv() {
    find_uv && return 0
    log "installing uv ${UV_VERSION} into ${UV_BIN_DIR}"
    mkdir -p "${UV_BIN_DIR}" "${SETUP}/dl"
    local tgz="${SETUP}/dl/uv-${UV_VERSION}.tar.gz"
    [[ -f "${tgz}" && "$(sha256 "${tgz}")" == "${UV_SHA256}" ]] || fetch "${UV_URL}" "${tgz}" \
        || die "download of ${UV_URL} failed"
    [[ "$(sha256 "${tgz}")" == "${UV_SHA256}" ]] || die "uv archive sha256 mismatch: ${tgz}"
    tar -xzf "${tgz}" -C "${SETUP}/dl"
    install -m 0755 "${SETUP}/dl/uv-x86_64-unknown-linux-gnu/uv" "${UV_BIN_DIR}/uv"
    install -m 0755 "${SETUP}/dl/uv-x86_64-unknown-linux-gnu/uvx" "${UV_BIN_DIR}/uvx"
    find_uv || die "${UV_BIN_DIR}/uv does not report uv ${UV_VERSION}"
}

ensure_python() {
    ensure_uv
    "${UV}" python find "${PY_VERSION}" >/dev/null 2>&1 || "${UV}" python install "${PY_VERSION}"
    "${UV}" python find "${PY_VERSION}" >/dev/null || die "uv cannot provide CPython ${PY_VERSION}"
}

# ---------------------------------------------------------------------------------------------------------- venvs
venv_healthy() {   # venv_healthy DIR: its python runs and is the pinned version
    [[ -x "$1/bin/python" ]] \
        && [[ "$("$1/bin/python" -B -I -c 'import platform; print(platform.python_version())' 2>/dev/null)" == "${PY_VERSION}" ]]
}

freeze_diff() {   # freeze_diff VENV FREEZE [quiet]: installed distributions against the freeze file (editables apart)
    "$1/bin/python" -B -I - "$2" "${3:-}" <<'PY'
import json, re, sys
from importlib import metadata
def norm(name): return re.sub(r"[-_.]+", "-", name).lower()
want, editable = {}, {}
for line in open(sys.argv[1]):
    line = line.strip()
    m = re.match(r"#\s*editable:\s*([A-Za-z0-9_.\-]+)==(\S+)", line)
    if m:
        editable[norm(m.group(1))] = m.group(2)
    elif line and not line.startswith("#"):
        name, _, version = line.partition("==")
        want[norm(name)] = version.strip()
have, where = {}, {}
for dist in metadata.distributions():
    name = dist.metadata["Name"]
    if name:
        have[norm(name)] = dist.version
        try:
            where[norm(name)] = json.loads(dist.read_text("direct_url.json") or "{}").get("url", "")
        except ValueError:
            pass
problems = [f"{n}: want {v}, installed {have.get(n)}" for n, v in sorted(want.items()) if have.get(n) != v]
problems += [f"{n} (editable): want {v}, installed {have.get(n)}" for n, v in sorted(editable.items())
             if have.get(n) != v]
problems += [f"{n}: installed {have[n]}, not in the freeze file" for n in sorted(set(have) - set(want) - set(editable))]
if sys.argv[2] != "quiet":
    print(f"{len(want)} pinned + {len(editable)} editable distributions; {len(problems)} difference(s)")
    for p in problems[:30]:
        print("    " + p)
    for n in sorted(editable):
        if n in have:
            print(f"    editable {n} -> {where.get(n, '?').removeprefix('file://')}")
sys.exit(1 if problems else 0)
PY
}

venv_done() {   # venv_done DIR FREEZE
    venv_healthy "$1" && freeze_diff "$1" "$2" quiet >/dev/null 2>&1
}

check_venv_sim() { venv_done "${AGP_HOME}/.venv-sim" "${SIM_FREEZE}"; }
check_venv_agent() { venv_done "${AGP_HOME}/.venv-agent" "${AGENT_FREEZE}"; }
check_venv_policy() { venv_done "${AGP_HOME}/.venv-policy" "${POLICY_FREEZE}"; }

EDITABLES=("${C}/third_party/IsaacLab/source/isaaclab" "${C}/third_party/IsaacLab/source/isaaclab_assets"
           "${C}/third_party/IsaacLab/source/isaaclab_tasks" "${C}/third_party/curobo")

do_venv_sim() {
    ensure_python
    local V="${AGP_HOME}/.venv-sim"
    if ! venv_healthy "${V}"; then
        [[ ! -e "${V}" ]] || { log "${V} does not run CPython ${PY_VERSION}: recreated"; rm -rf "${V}"; }
        "${UV}" venv -q --python "${PY_VERSION}" --seed "${V}"
    fi
    log "installing env/sim/requirements-freeze.txt into ${V} (no deps; about 25 GB, 19 GB of wheels to download)"
    OMNI_KIT_ACCEPT_EULA=YES "${UV}" pip install --python "${V}/bin/python" --no-deps -r "${SIM_FREEZE}" \
        --extra-index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.nvidia.com \
        --index-strategy unsafe-best-match
    log "editable installs from ${C}/third_party (IsaacLab x3, cuRobo; no build isolation)"
    local stamp="${SETUP}/.editable-stamp" wrote
    : > "${stamp}"
    "${UV}" pip install --python "${V}/bin/python" --no-deps --no-build-isolation \
        -e "${EDITABLES[0]}" -e "${EDITABLES[1]}" -e "${EDITABLES[2]}" -e "${EDITABLES[3]}"
    wrote="$(find "${C}/third_party/IsaacLab/source" "${C}/third_party/curobo" -newer "${stamp}" \
        -not -path '*/.git/*' -print 2>/dev/null | head -20 || true)"
    rm -f "${stamp}"
    [[ -z "${wrote}" ]] || die "the editable installs wrote into the checkout (remove these, then report it): ${wrote}"
    freeze_diff "${V}" "${SIM_FREEZE}" || die "${V} differs from ${SIM_FREEZE} (remove ${V} and run install.sh again)"
}

do_venv_agent() {
    ensure_python
    local V="${AGP_HOME}/.venv-agent"
    [[ ! -e "${V}" ]] || { log "${V} differs from the freeze file: rebuilt"; rm -rf "${V}"; }
    "${UV}" venv -q --python "${PY_VERSION}" --prompt agp-agent "${V}"
    "${UV}" pip install --python "${V}/bin/python" --no-deps -r "${AGENT_FREEZE}"
    freeze_diff "${V}" "${AGENT_FREEZE}" || die "${V} differs from ${AGENT_FREEZE}"
}

do_venv_policy() {
    ensure_python
    local V="${AGP_HOME}/.venv-policy"
    [[ ! -e "${V}" ]] || { log "${V} differs from the freeze file: rebuilt"; rm -rf "${V}"; }
    "${UV}" venv -q --python "${PY_VERSION}" --prompt agp-policy "${V}"
    "${UV}" pip install --python "${V}/bin/python" --no-deps -r "${POLICY_FREEZE}"
    freeze_diff "${V}" "${POLICY_FREEZE}" || die "${V} differs from ${POLICY_FREEZE}"
}

# ---------------------------------------------------------------------------------------------------------- codex
check_codex() {   # the pinned release folder, by sha256 (codex itself is not started)
    [[ -d "${CODEX_DIR}" && ! -L "${CODEX_DIR}" && -f "${CODEX_BIN}" && ! -L "${CODEX_BIN}" ]] || return 1
    [[ -x "${CODEX_BIN}" && -x "${CODEX_DIR}/bin/codex-code-mode-host" && -f "${CODEX_DIR}/codex-package.json" ]] \
        || return 1
    [[ "$(sha256 "${CODEX_BIN}")" == "${CODEX_BIN_SHA256}" ]] || return 1
    [[ "$(sha256 "${CODEX_DIR}/bin/codex-code-mode-host")" == "${CODEX_HOST_SHA256}" ]]
}

do_codex() {
    mkdir -p "${SETUP}/dl" "$(dirname "${CODEX_DIR}")"
    local tgz="${SETUP}/dl/codex-${CODEX_VERSION}-${CODEX_ASSET}" url
    if [[ ! -f "${tgz}" || "$(sha256 "${tgz}")" != "${CODEX_ASSET_SHA256}" ]]; then
        for url in "${CODEX_URLS[@]}"; do
            log "downloading ${url}"
            fetch "${url}" "${tgz}" && [[ "$(sha256 "${tgz}")" == "${CODEX_ASSET_SHA256}" ]] && break
            log "download or sha256 check failed: ${url}"; rm -f "${tgz}"
        done
    fi
    [[ -f "${tgz}" ]] || die "cannot download ${CODEX_ASSET} ${CODEX_VERSION} with sha256 ${CODEX_ASSET_SHA256}"
    local stage; stage="$(dirname "${CODEX_DIR}")/.staging.$(basename "${CODEX_DIR}").$$"
    rm -rf "${stage}"; mkdir -p "${stage}"
    tar -xzf "${tgz}" -C "${stage}"
    chmod 0755 "${stage}/bin/codex" "${stage}/bin/codex-code-mode-host" "${stage}/codex-path/rg"
    [[ ! -f "${stage}/codex-resources/bwrap" ]] || chmod 0755 "${stage}/codex-resources/bwrap"
    ln -sfn bin/codex "${stage}/codex"                  # as the official installer lays it out
    [[ "$(sha256 "${stage}/bin/codex")" == "${CODEX_BIN_SHA256}" ]] || die "bin/codex sha256 differs from the pin"
    [[ "$(sha256 "${stage}/bin/codex-code-mode-host")" == "${CODEX_HOST_SHA256}" ]] \
        || die "bin/codex-code-mode-host sha256 differs from the pin"
    if [[ -e "${CODEX_DIR}" || -L "${CODEX_DIR}" ]]; then     # a differing folder of this version is kept aside
        local aside; aside="${CODEX_DIR}.replaced-$(date -u +%Y%m%dT%H%M%SZ)"
        log "${CODEX_DIR} differs from the pinned release: moved to ${aside}"
        mv "${CODEX_DIR}" "${aside}"
    fi
    mv "${stage}" "${CODEX_DIR}"
    check_codex || die "the installed codex release does not check out"
    log "$(CODEX_HOME="${HOME}/.codex" "${CODEX_BIN}" --version 2>/dev/null || true) at ${CODEX_DIR}"
}

# ---------------------------------------------------------------------------------------------- what is present
# `ldconfig -p`, read once and searched as a string
LDCONFIG_CACHE=""
LDCONFIG_READ=0
lib_present() {
    if [[ "${LDCONFIG_READ}" == 0 ]]; then
        local ldc
        ldc="$(command -v ldconfig || ls /usr/sbin/ldconfig /sbin/ldconfig 2>/dev/null | sed -n 1p || true)"
        [[ -z "${ldc}" ]] || LDCONFIG_CACHE="$("${ldc}" -p 2>/dev/null || true)"
        LDCONFIG_READ=1
    fi
    grep -q "^[[:space:]]*$1 (libc6,x86-64)" <<< "${LDCONFIG_CACHE}" && return 0
    [[ -e "${RUNTIME_LIBS}/usr/lib/x86_64-linux-gnu/$1" ]]
}
RUNTIME_LIBS="${AGP_RUNTIME_LIBS:-$(dirname "${AGP_HOME}")/robodojo/runtime-libs}"   # as activate_sim.sh derives it

report_system() {
    echo "System (checked only; installing these needs an administrator):"
    if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
        ok gpu "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed -n 1p) ($(nvidia-smi -L | wc -l) GPU(s))"
    else
        missing gpu "nvidia-smi does not work: no NVIDIA driver visible (Isaac Sim needs an RTX GPU)"
    fi
    local icd found=""
    for icd in "${VULKAN_ICDS[@]}"; do
        if [[ -f "${icd}" ]]; then found="${icd}"; break; fi
    done
    if [[ -n "${found}" ]]; then
        ok vulkan "${found} (activate_sim.sh pins it)"
    else
        missing vulkan "neither ${VULKAN_ICDS[0]} nor ${VULKAN_ICDS[1]} found (NVIDIA driver graphics libraries; in docker: NVIDIA_DRIVER_CAPABILITIES=all)"
    fi
    local entry absent=()
    for entry in "${SYSLIBS[@]}"; do lib_present "${entry%%:*}" || absent+=("${entry#*:}"); done
    if [[ ${#absent[@]} -eq 0 ]]; then
        ok syslibs "${#SYSLIBS[@]} libraries Isaac Sim loads"
    else
        missing syslibs "as root: apt-get install ${absent[*]} (or unpack them into ${RUNTIME_LIBS})"
    fi
    if "${SYSPY}" -B -I -c 'import yaml' 2>/dev/null; then
        ok python3 "${SYSPY} $("${SYSPY}" -B -I -c 'import sys, yaml; print(sys.version.split()[0], "with PyYAML", yaml.__version__)')"
    else
        note python3 "${SYSPY} has no PyYAML: agp_runtime/agp_core/freeze.py needs it (apt-get install python3-yaml)"
    fi
    local nofile; nofile="$(ulimit -n)"
    [[ "${nofile}" == unlimited || "${nofile}" -ge 65536 ]] || note ulimit "ulimit -n is ${nofile}; Isaac Sim wants >= 65536"
}

report_checkout() {
    echo "Checkout ${C}:"
    local key pin head bad=0
    for key in RoboDojo RoboDojo/XPolicyLab RoboDojo/third_party/IsaacLab RoboDojo/third_party/curobo; do
        pin="$(json_get "${VERSIONS}" "d['repositories']['${key}']")"
        head="$(git -C "${C}${key#RoboDojo}" rev-parse HEAD 2>/dev/null || echo '?')"
        if [[ "${head}" != "${pin}" ]]; then
            note "${key##*/}" "at ${head:0:8}; the system was frozen and tested at ${pin:0:8}"
            bad=1
        fi
    done
    [[ "${bad}" == 1 ]] || ok commits "RoboDojo $(git -C "${C}" rev-parse --short=8 HEAD) and its 3 submodules at the pinned commits"
    if [[ -d "${C}/Assets/Robots/x5" && -f "${C}/Assets/Robots/x5/curobo.yml" ]]; then
        ok assets "$(realpath "${C}/Assets") (Robots/x5/curobo.yml generated)"
    elif [[ -d "${C}/Assets/Robots/x5" ]]; then
        missing assets "Assets/Robots/x5/curobo.yml missing: run RoboDojo's utils/update_embodiment_config_path.py"
    else
        missing assets "${C}/Assets missing: download the RoboDojo Assets first (RoboDojo installation docs)"
    fi
}

report_access() {
    echo "Model access (install.sh never reads, copies or writes credentials):"
    local slots=0 d
    for d in "${AGP_HOME}/private/accounts/slots"/*/; do
        if [[ -f "${d}auth.json" ]]; then slots=$((slots + 1)); fi
    done
    [[ "${slots}" -eq 0 ]] || ok pool "${slots} account-pool slot(s) in ${AGP_HOME}/private/accounts (up to 3 episodes per login)"
    if [[ -f "${HOME}/.codex/auth.json" ]]; then
        if [[ "${slots}" -eq 0 ]]; then ok login "${HOME}/.codex/auth.json (your own codex login; used without a pool)"
        else note login "${HOME}/.codex/auth.json present (not used while the pool has slots)"; fi
    elif [[ "${slots}" -eq 0 ]]; then
        note login "no codex login: agent episodes need one (scripted test episodes do not)"
    fi
    cat <<EOF
  Log in once with your ChatGPT account (headless: a device code is shown):
    CODEX_HOME="\$HOME/.codex" ${CODEX_BIN} -c 'cli_auth_credentials_store="file"' login --device-auth
  Or an account pool, one login per slot (at most 3 slots per account):
    S="${AGP_HOME}/private/accounts/staging/a1s1"; mkdir -p -m 700 "\$S"
    CODEX_HOME="\$S" ${CODEX_BIN} -c 'cli_auth_credentials_store="file"' login --device-auth
    AGP_HOME="${AGP_HOME}" ${SYSPY} -B ${RUNTIME}/agp_core/accounts.py add a1s1 "\$S"
EOF
}

describe_step() {   # what a step would do
    case "$1" in
        venv-sim) echo "create ${AGP_HOME}/.venv-sim (CPython ${PY_VERSION}), install ${SIM_FREEZE#"${POLICY}/"} with --no-deps, then IsaacLab x3 + cuRobo editable from ${C}/third_party" ;;
        venv-agent) echo "create ${AGP_HOME}/.venv-agent (CPython ${PY_VERSION}) from ${AGENT_FREEZE#"${POLICY}/"}" ;;
        venv-policy) echo "create ${AGP_HOME}/.venv-policy (CPython ${PY_VERSION}) from ${POLICY_FREEZE#"${POLICY}/"}" ;;
        codex) echo "download ${CODEX_ASSET} ${CODEX_VERSION} (sha256 ${CODEX_ASSET_SHA256:0:12}...) into ${CODEX_DIR}" ;;
    esac
}

report_step() {   # the state of a step (after an install, or for --check)
    local V F out rc
    case "$1" in
        venv-sim|venv-agent|venv-policy)
            V="${AGP_HOME}/.$1"; F="${SIM_FREEZE}"
            [[ "$1" != venv-agent ]] || F="${AGENT_FREEZE}"
            [[ "$1" != venv-policy ]] || F="${POLICY_FREEZE}"
            if ! venv_healthy "${V}"; then
                missing "$1" "${V} absent or not CPython ${PY_VERSION}"
            else
                rc=0; out="$(freeze_diff "${V}" "${F}")" || rc=$?
                if [[ "${rc}" == 0 ]]; then
                    ok "$1" "${V}: CPython ${PY_VERSION}, ${out%%$'\n'*}"
                    if [[ "$1" == venv-sim ]]; then
                        printf '%s\n' "${out}" | sed -n 's/^    editable /             editable /p'
                        note "" "at run time IsaacLab and cuRobo are imported from ${C}/third_party (activate_eval.sh)"
                    fi
                else
                    missing "$1" "${V} differs from ${F#"${POLICY}/"}:"
                    printf '%s\n' "${out}" | sed -n '1,12s/^/             /p'
                fi
            fi
            ;;
        codex)
            if check_codex; then
                ok codex "${CODEX_BIN} (release ${CODEX_VERSION}, sha256 ${CODEX_BIN_SHA256:0:12}... as pinned)"
            else
                missing codex "${CODEX_DIR} absent or not the pinned release"
            fi
            ;;
    esac
    if [[ "${CHECK}" == 1 && " ${MISSING[*]} " == *" $1 "* ]]; then
        note "" "would $(describe_step "$1")"
    fi
    return 0
}

# ------------------------------------------------------------------------------------------------------------ main
main() {
    log "AGP install$([[ "${CHECK}" == 1 ]] && echo ' --check (nothing is changed)'): steps ${SELECTED[*]}"
    echo "  policy     ${POLICY}"
    echo "  AGP_HOME   ${AGP_HOME}"
    echo "  setup dir  ${SETUP}$([[ -d "${SETUP}" ]] || echo ' (not created yet)')"
    if [[ "${CHECK}" == 0 ]]; then
        local s
        for s in "${SELECTED[@]}"; do
            if "check_${s//-/_}" 2>/dev/null; then
                log "step ${s}: done, skipped"
                continue
            fi
            mkdir -p "${SETUP}"
            cd "${SETUP}"                                   # uv and the builds never run from inside the checkout
            log "step ${s}: $(describe_step "${s}")"
            "do_${s//-/_}"
            log "step ${s}: done"
        done
    else
        if find_uv; then echo "  uv         ${UV} (uv ${UV_VERSION}; needed only to build)"
        else echo "  uv         not found: a step that builds would install uv ${UV_VERSION} into ${UV_BIN_DIR}"; fi
        echo "  free disk  $(df -Pk "$(existing_parent "${AGP_HOME}")" | awk 'NR==2 {print int($4/1048576)}') GiB at AGP_HOME"
    fi
    echo "Steps:"
    local s
    for s in "${SELECTED[@]}"; do report_step "${s}"; done
    report_checkout
    report_system
    report_access
    if [[ ${#MISSING[@]} -eq 0 ]]; then
        log "ready: every selected step is done and nothing the runtime needs is missing"
        exit 0
    fi
    log "not ready: missing ${MISSING[*]}$([[ "${CHECK}" == 1 ]] && echo ' (without --check the steps are installed)')"
    exit 1
}

main
