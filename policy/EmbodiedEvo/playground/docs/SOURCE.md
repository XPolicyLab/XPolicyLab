# EmbodiedEvo source and components

EmbodiedEvo provides the Playground and Test implementation in this directory.
[SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json) records the source revision,
source file hashes and packaged file hashes. Paths in the manifest are relative
to the Playground directory.

The implementation includes the Playground/session runtime, simulator adapters,
optional native-evaluator transport, task templates, provisioning scripts, CPU
tests and documentation. It excludes virtualenvs, simulator assets, vendor trees,
credentials, local deployment profiles, results and artwork. The outer policy's
`Workspace/` separately contains the previously frozen Skills and Lessons for
42 tasks; those files are not initial Playground knowledge.

The XPolicyLab policy identifier is `EmbodiedEvo`, and the agent image is
`embodiedevo-agent:xpolicy`. The optional native transport selects its own
adapter in a separate subprocess. The launcher and export command connect
Playground to the Test entrypoint. Both phases use `workspace-session-v1`.
Task goals and action limits come from RoboDojo; execution budgets are configured
in the task definition, and the primitive manuals describe the agent interface.

The [recipe manifest](../tasks/RoboDojo/recipe/manifest.json), Apache-2.0 license
and [third-party notices](../THIRD_PARTY_NOTICES.md) preserve upstream attribution.
CPU tests validate session behavior and export boundaries. A live Playground
requires a separately provisioned RoboDojo runtime, Docker, GPU and provider
credentials; source inclusion alone does not establish benchmark performance.
