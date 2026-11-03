# Manage services

```bash
harness services [--json]               # every manifest: kind, protocol, port, VRAM
harness doctor [NAMES] --deploy mine    # can this machine run them?
harness fetch NAMES --deploy mine       # download their weights
harness up [NAMES] --deploy mine        # start the resident ones (and the broker)
harness status [NAMES] --deploy mine    # where each runs, whether it answers
harness down [NAMES] --deploy mine      # stop what harness up started
```

`--deploy` takes a name (`configs/deploy/<name>.toml`) or a path; without it,
`$HARNESS_DEPLOY`.

## doctor

Checks, for each service the deploy runs: the files its manifest `needs_files`, its
interpreter and working directory, that its GPU exists, that external URLs answer, and
that the GPUs have free memory for the services still to start (a swap group counted at
its largest member); also that the simulator GPUs exist and where the RoboDojo checkout
is. A port that already answers is noted (that service will be used as it is). It
prints the GPUs, problems and notes (`--json` for all of it) and exits 1 when there is a
problem.

## fetch

For each `[[weights]]` entry of a manifest, downloads into `<weights>/<dir>`:
`modelscope_model` / `modelscope_dataset` through `scripts/modelscope_download.py`,
`hf_model` / `hf_dataset` through `scripts/hf_mirror_download.py` (hf-mirror.com, large
files in parallel ranges). Files whose sha256 already matches are skipped; downloads
resume. An entry that is a symlink, or marked `local`, is left as it is: to use weights
you already have elsewhere, symlink the directory.

## up, status, down

`harness up` starts each resident service in its own process group with its log in
`<state>/logs/<name>.log` and a pid record, waits until each answers (up to three times
its `load_seconds`), then starts the broker when the deploy has on-demand services. A
service whose port already answers is left alone (another checkout, or started by
hand). `harness down` stops only what it started: SIGINT, then SIGTERM, then SIGKILL to
the process group, 15 s apart; without names, the broker too.

## On demand, swap groups, external

```toml
[services.g05]
gpu = 1
mode = "on_demand"
swap = "learned"        # services of one group share GPU 1, one loaded at a time
idle_seconds = 600      # stop it after 10 minutes without a request

[services.openwam]
gpu = 1
mode = "on_demand"
swap = "learned"

[services.molmo2]
mode = "external"       # someone else runs it
url = "http://10.0.0.7:8830"
```

The broker listens on `broker_port` (default 8899) and starts an on-demand service
when a client asks; clients ask before each call. `harness run` checks before the first
episode that resident and external services answer and starts the broker if needed.

## A service with its own interpreter or environment

```toml
[services.dm05]
python = "/data/venvs/opendm/bin/python"
env = { OPENPI_DATA_HOME = "/data/openpi-cache" }
gpu = 2
```

## When something does not start

- `missing files [...]; run harness fetch NAME` — the weights are not where the
  manifest expects them (`<weights>/<dir>`).
- `NAME exited before it answered; see <state>/logs/NAME.log` — read the log.
- `No GPU of [...] has N GB left for NAME` — the plan does not fit; move a service,
  put services on demand in a swap group, or turn one off.
- `Services A and B both use port P` — give one a `port` in the deploy.
