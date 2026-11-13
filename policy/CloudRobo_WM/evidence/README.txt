CloudRobo-WM evaluation evidence: 88.98%

Clean / seen:     4764 / 5000 = 95.28%
Random / unseen:  4134 / 5000 = 82.68%
Average:         8898 / 10000 = 88.98%

Recorded trial interval (UTC+08:00):
2026-11-11 10:40:52.091966 through 2026-11-11 20:45:38.460978
Archive directory: /data/robotwin_evaluation/20261111_104052

Contents:
- results/: aggregate and all 100 profile/task results.
- official_results/: native per-task result files.
- official_task_logs/: full-length simulator evaluation logs.
- policy_servers/: policy input and outcome event logs.
- evidence/INPUTS_AND_OUTCOMES.json: all 10,000 completed trials, including
  every success and failure, with task, profile, seed, prompt and outcome.
- PUBLIC_LOG_MANIFEST.json: file sizes and SHA256 checksums.

Checkpoint SHA256:
660bc6bd55d526366ea0f4abccb660d1b5aa1a29c8af1c7bfb1566a4b9e0462d
Input manifest SHA256:
d4e522a5988089d81be52425a3a48466488677c0280ffeeda55c03913a583b58

Evaluation protocol is documented in the policy README.md.
The separate runtime_validation_20261112 package contains entrypoint checks;
those trials are not included in this score.
