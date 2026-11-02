# RSIWAM

RSIWAM is RSIBot's recovery-adapted policy submission for RoboDojo. It adds a gated action-correction module and success133 Action LoRA to a frozen world-model foundation. The design aims to learn corrections after deviations while retaining the frozen policy's capabilities; overall recovery gains have not yet been established by the ongoing evaluation.

## Public code and model identifiers

| Item | Identifier |
| --- | --- |
| Policy / model name | `RSIWAM` |
| Code directory | `policy/RSIWAM` |
| XPolicyLab entrypoint | `XPolicyLab.policy.RSIWAM.model.Model` |
| Runtime distribution | `rsiwam-frozen-runtime` |
| Deployment policy | `policy_name: RSIWAM` |
| Checkpoint display name | `rsiwam-original32-lora33-step400` |

These names identify the composed policy. The evaluated inference implementation and learned weight bytes are unchanged by display-name or documentation changes.

## Fixed candidate

The recovery parent is frozen at seed 20261032. Action LoRA is frozen at seed 20261033, optimizer step 400 (119 original steps plus 281 resumed steps). Model sampling seed is 1 across tasks and environment seeds. The exact learned weights are included in this adapter:

| Component | SHA-256 |
| --- | --- |
| Recovery parent | `861529eb89e4279e4490d6156bf04a335eee410f6c02842d0ac3db2dc90431d3` |
| Action LoRA | `eec21ec2834ef95bffdac7b3df0b48092a215aa67e7bf6952ec1643ab511b361` |
| Base manifest identity | `ae66dbfc16eb3c75fa89111501412ecb14cd3ab62ce02ff9fc8a457bb3b19aba` |

The historical machine-readable candidate ID remains `recoverwam_success133_original_parent20261032_lora20261033` so original logs, checkpoints and results can be matched exactly.

## Component provenance

The foundation is the frozen official VPP2 `joint2b_s100000` model; upstream code is [video-prediction-policy-2](https://github.com/roboterax/video-prediction-policy-2). Base assets are fetched from [Haodong082399/VPP2](https://huggingface.co/Haodong082399/VPP2/tree/04824ddb3aae11f13db50984ed7d540c3b77dd9f) at the pinned revision in `bundle/base_weights.json`. The recovery component is the original RecoverWAM parent. RSIWAM is a composed adaptation of those components, not an independently pretrained foundation. Their notices, licenses, source identities and weight hashes are retained.

## Evidence and limitations

- Real CUDA plain/encoded synthetic debug passed at commit `7e5def6e61435e823f5b346f2b9d2dd5a7a78256`; see [the evidence summary](checks/gpu_debug/SUMMARY.json). Each mode executed 20 actions and returned an identical action chunk.
- Real installation, all 27 bundled dependency constraints and CPU imports passed at commit `776c9f3b177427c1078ee31daaa4db93f922008b` in a newly provisioned isolated environment after a corrective reinstall. This is separate from GPU evidence and does not claim a one-shot empty-environment installation; see [the installation record](checks/clean_install/SUMMARY.json).
- The literal plain standard launcher at `776c9f3b177427c1078ee31daaa4db93f922008b` completed 10 debug episodes / 200 actions. Its original overall attempt failed during resource-lease cleanup; encoded did not start in that attempt. [Original logs and the failure record](checks/standard_launcher_plain/SUMMARY.json) are retained. A later [independent cleanup closure](checks/standard_launcher_cleanup/SUMMARY.json) reconciled only the orphan reservation; the subsequent [literal encoded continuation](checks/standard_launcher_encoded/SUMMARY.json) completed 10 debug episodes / 200 actions, reached `[MAIN] eval finished`, exited 0 without a traceback, and released its resources. The two-mode acceptance is derived across these separate runs and the plain cleanup closure, not an uninterrupted two-mode run.
- Native evaluation of the same fixed candidate is ongoing. The [partial results attachment](checks/native_partial_325/README.txt) contains 325 accepted episodes from seven completed task blocks at evaluation seed 0: 137 successes (42.15% SR), mean Score 45.38/100. This completed-block selection is not representative of the full suite, has no paired baseline, and is not a recovery-specific success rate. The full 6,300-episode suite and official organizer verification are pending.
- Synthetic debug does not establish task success, recovery capability, clean installation, or full-episode parity with the historical implementation.
- Historical same-scene comparisons are partial development evidence. They are not an official score, and a failure-to-success pair alone does not prove recovery within an episode.
- Success133 includes Agent action proxy labels whose execution equivalence is not certified. Training-source and reproducibility limitations are disclosed in the [public companion artifact](https://github.com/modaxiansheng/XPolicyLab/releases/tag/rsiwam-training-companion-20261101).

Team: RSIBot (independent team). Contact: 22110110@bjtu.edu.cn. Technical report planned for 2026-11-25; evaluated training/deployment artifacts are committed for release within three days of verified listing.
