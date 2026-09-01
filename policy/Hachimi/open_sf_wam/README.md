# open_sf_wam

Vendored training and inference code for the `Hachimi` policy.

## Provenance

This tree is a fork of the `openpi-SF/` subtree of
**Spatial Forcing** (<https://github.com/OpenHelix-Team/Spatial-Forcing>, arXiv:2510.12276),
which is itself built on **openpi** by Physical Intelligence
(<https://github.com/Physical-Intelligence/openpi>).

Our changes are limited to the action-conditioned future latent prediction head and the
training config for it, plus three small operational fixes; every changed file is listed
in the table in [`../README.md`](../README.md).

## Licenses

| File | Applies to |
|---|---|
| `LICENSE` | openpi — Apache License 2.0 |
| `LICENSE_SPATIAL_FORCING` | Spatial Forcing — MIT License, Copyright (c) 2025 Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, Haoang Li |
| `LICENSE_GEMMA.txt` | Gemma / PaliGemma weights reached through the `pi05_base` checkpoint — Gemma Terms of Use |

All three are retained unmodified. Model weights trained from `pi05_base` inherit the
Gemma Terms of Use.

## Usage

Do not invoke this tree directly. Installation, data, training and evaluation commands are
documented in the policy README at [`../README.md`](../README.md); the entry points are
`../install.sh`, `../train.sh` and `../eval.sh`.

The training config used for the submitted checkpoint is
`pi05sfwam_jax_robodojo_v21_merged`
(`src/openpi/external_config/pi05sfwam_jax_robodojo_v21_merged.py`). The variant without
the future head, `pi05sf_jax_robodojo_v21_merged`, is retained as the ablation config, and
`pi05sf_jax_robodojo_v21_offcache` is the released-data-only reproduction control.
