# Model-bundle runtime provenance

The robot-specific model bundle contains a minimal inference runtime derived from the OpenGalaxea G05 source snapshot:

```text
commit: 7f8f78e6a6632d167365cc385f5555f2bae686a0
```

The XPolicyLab adapter loads only the runtime shipped with the selected model bundle. It never loads `policy/G05/G05` or an arbitrary external `G05_ROOT`.

The model bundle includes only the runtime surface needed to load and serve the released checkpoint:

- the imported subset of `src/g05/`
- `scripts/serve_policy.py`
- `configs/model/g05.yaml`
- the imported subsets of the checked-out `galaxea_dataset` and `galaxea_tokenizer` dependencies

Training launchers, cluster configuration, datasets, checkpoints, logs, credentials, and internal storage paths are excluded.
