# Reproducibility

This directory contains Joy Future Academy's JoyAI-VL Agent adapter.

## Source identity

`metadata/variant_manifest.json` identifies the earlier prepared adapter and records
unchanged files, modified files and omissions. `metadata/source_manifest.json` and
`metadata/integration_manifest.json` retain the earlier source and packaging history.
`metadata/release_manifest.json` records the current release checksums.

The API request loop, robot controller, perception geometry and shared SAM3 service
are preserved. The shared service SHA-256 is
`7ee55899f1d6753f6ef9959ce9e1eba0c6e31e09c62c74eee37d1c6d41f485a1`.
See [development history](development.md) for the prompt and context changes.

## Deployment

The agent Docker recipe uses Python 3.11, Pillow 11.3.0 and NumPy 1.26.0.
The policy server uses RoboDojo's Isaac/cuRobo environment. SAM3 runs as a shared
service. `metadata/runtime_reference.json` retains reference environment identities;
record the actual rebuilt image ID and installed versions for each deployment.
The installer supplies a policy overlay with `websockets>=13`.

## Verification

```bash
python3 tools/verify_release.py
python3 -m unittest discover -s tests -v
```

The verifier checks the variant's source relationship, release checksums, absent
task assets and Python/shell syntax. Tests exercise context assembly, fixed tool
selection, configuration, API requests, RGB observations and evaluation entries.
Generated runtime files, Git metadata and private configuration are excluded from
the release manifest.
