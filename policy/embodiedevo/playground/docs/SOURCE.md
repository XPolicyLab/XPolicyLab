# Source provenance and scope

This XPolicyLab bundle was prepared on 2026-10-14 from the supplied local
`EmbodiedRSI` checkout, whose project is now named EmbodiedEvo.
[SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json) records that checkout's commit,
each imported path and original SHA-256, and the current bundled path and hash.
Files are independent copies; execution does not import the original checkout.

The bundle includes the Playground/session implementation, simulator adapters,
optional native-evaluator transport, task templates, provisioning scripts, CPU
tests and documentation. It excludes virtualenvs, simulator assets, vendor trees,
credentials, local deployment profiles, results and artwork. The outer policy's
`Workspace/` separately contains the previously frozen Skills and Lessons for
42 tasks; those files are not initial Playground knowledge.

Local adaptations align the policy identifier with `embodiedevo`, use the outer
policy's `embodiedevo-agent:xpolicy` image tag, and select the optional native
transport's adapter before the outer Test adapter in its subprocess. The added
launcher and export command connect Playground to the existing Test entrypoint.
The protocol remains `workspace-session-v1`; task goals, execution budgets,
primitive manuals and third-party recipe text are unchanged.

The [recipe manifest](../tasks/RoboDojo/recipe/manifest.json), Apache-2.0 license
and [third-party notices](../THIRD_PARTY_NOTICES.md) preserve upstream attribution.
CPU tests validate session behavior and export boundaries. A live Playground
requires a separately provisioned RoboDojo runtime, Docker, GPU and provider
credentials; source inclusion alone does not establish benchmark performance.
