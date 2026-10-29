# Configure the Codex API

Complete this step before launching an agent. The simulator does not need an API
key; the coding agent does. Run the shell examples in Bash from the repository root.

## 1. Choose a dedicated configuration directory

Keep credentials outside the checkout and separate from your interactive Codex:

```bash
export EMBODIEDEVO_CODEX_HOME="$HOME/.config/embodiedevo/codex"
install -d -m 700 "$EMBODIEDEVO_CODEX_HOME"
```

Select **one** template. The copy commands preserve an existing `config.toml`;
if you have configured this directory before, edit that file instead.

For the official OpenAI API:

```bash
cp -n configs/codex.openai.example.toml "$EMBODIEDEVO_CODEX_HOME/config.toml"
chmod 600 "$EMBODIEDEVO_CODEX_HOME/config.toml"
```

For a compatible gateway:

```bash
cp -n configs/codex.custom.example.toml "$EMBODIEDEVO_CODEX_HOME/config.toml"
chmod 600 "$EMBODIEDEVO_CODEX_HOME/config.toml"
```

Edit the gateway template's `base_url` to the endpoint supplied by your provider,
usually ending in `/v1`, without appending `/responses`. The gateway must support
the Responses API features used by Codex, including streaming, image input and
tool calls. A Chat Completions-only endpoint is insufficient for this example.
`wire_api = "responses"` selects that protocol. `requires_openai_auth = true`
uses the saved login credential. See the official
[provider configuration reference](https://developers.openai.com/codex/config-reference/)
and [custom provider guide](https://developers.openai.com/codex/config-advanced/).

Both examples explicitly select a model and reasoning effort. Check their
availability with your API provider; the shipped values match the task templates,
not an account entitlement check. Keep the selected directory exported in the
shell or job that starts each experiment.

## 2. Save an API key in that directory

Use the key for the selected endpoint. Enter it at the hidden prompt, with shell
tracing (`set -x`) disabled:

```bash
read -r -s -p "API key: " embodiedevo_api_key
printf '\n'
printf '%s' "$embodiedevo_api_key" | \
  CODEX_HOME="$EMBODIEDEVO_CODEX_HOME" codex login --with-api-key
unset embodiedevo_api_key
chmod 600 "$EMBODIEDEVO_CODEX_HOME/auth.json"
CODEX_HOME="$EMBODIEDEVO_CODEX_HOME" codex login status
```

The templates request file-based credential storage so the login cache is
`auth.json` in the selected directory. Passing `CODEX_HOME` on these commands
targets that directory without changing your usual Codex home. The official
[authentication guide](https://developers.openai.com/codex/auth/) documents
stdin API-key login and credential storage. Login status checks cached
authentication; it does not prove the endpoint or model is accessible.

The launcher stages `config.toml` provider settings and `auth.json` into a private
home for each agent. It does **not** forward arbitrary host environment variables
such as `OPENAI_API_KEY` or a custom `env_key`. Exporting a key on the host alone
therefore does not configure the container. Use the file-based setup above.

An existing working API-key home can be selected by setting
`EMBODIEDEVO_CODEX_HOME` to that directory instead. It must contain both files;
OS-keyring-only credentials cannot be copied by the launcher. Configuration in
a project `.codex/` directory or a separate named Codex profile is not staged.

## 3. Set the actual experiment model

Before creating a new run, set these fields in the appropriate task file:

```toml
[agent]
preset = "codex"
model = "gpt-6-astra"
reasoning_effort = "xhigh"
```

Edit the existing `[agent]` table rather than adding another one. The main task
uses `tasks/RoboDojo/task.toml`; the baseline uses
`tasks_baseline/RoboDojo/task.toml`. These two values are passed explicitly to
Codex and override the model defaults in `config.toml`. Keep them aligned for
paired comparisons. Changing model settings requires a new run directory.

## 4. Check credential staging without a model call

After installing the host environment, this checks the same staging function
used by real agents. It prints no credentials and launches no container:

```bash
.venv/bin/python - <<'PY'
import json
import tempfile
from pathlib import Path
from orchestrator.launch_agent import stage_home

with tempfile.TemporaryDirectory(prefix="embodiedevo-auth-check-") as directory:
    staged = stage_home(Path(directory) / "agent-home") / ".codex"
    auth = json.loads((staged / "auth.json").read_text())
    key = auth.get("OPENAI_API_KEY")
    assert isinstance(key, str) and key.strip(), "No saved API key in the selected home"
    assert (staged / "config.toml").is_file()
    print("API-key configuration staged successfully; no API request sent.")
PY
```

## 5. Check endpoint access before starting a simulator

This optional command sends a small real model request. Set its model and effort
to the same values as your task; it uses API quota but needs no GPU:

```bash
embodiedevo_probe_dir="$(mktemp -d)"
CODEX_HOME="$EMBODIEDEVO_CODEX_HOME" codex exec \
  --skip-git-repo-check --sandbox read-only --ephemeral \
  --cd "$embodiedevo_probe_dir" \
  --model gpt-6-astra -c 'model_reasoning_effort="xhigh"' \
  'Reply exactly OK. Do not use tools.'
rmdir "$embodiedevo_probe_dir"
```

A successful reply checks host-side API access. The agent container must also be
able to reach the endpoint. In a container, `localhost` refers to the container
itself; use a reachable gateway address or configure
`EMBODIEDEVO_AGENT_HTTPS_PROXY` for your network. Next build the
[thin agent image](INSTALL.md#thin-agent-container) and run the session diagnostic.

| Symptom | Check |
| --- | --- |
| Missing Codex config | The exported directory contains `config.toml` |
| Login prompt or missing API key | Login wrote `auth.json` to that same directory |
| HTTP 401/403 | Key validity and the provider's access permissions |
| Unknown model / unsupported reasoning effort | Both task files and the provider's supported values |
| HTTP 404 at `/responses` | The API base path and Responses support |
| Host works, container fails | Docker network, proxy and gateway reachability |

Keep real keys and provider configuration in the external directory. The
repository templates contain only public defaults and a placeholder endpoint.
