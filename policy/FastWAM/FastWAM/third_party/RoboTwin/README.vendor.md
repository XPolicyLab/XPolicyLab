# Vendored RoboTwin

This directory vendors code from the upstream RoboTwin repository:

- Upstream project: https://github.com/RoboTwin-Platform/RoboTwin
- Upstream commit: `bf44be51cf5717a5595ce59447f2cf5263d2aa95`
- Upstream license: MIT License

License compliance notes:

- The original upstream license is preserved in [`LICENSE`](./LICENSE).
- Files copied from RoboTwin remain subject to the MIT License in this directory.
- For this project, the only locally maintained policy implementation is `policy/fastwam`.
- `policy/fastwam` contains project-specific code authored for this repository and is not copied from the other upstream `policy/*` subprojects.
- If code is later copied from any upstream subdirectory with an additional license notice, the corresponding license file and attribution must also be preserved.

Local modifications:

- RoboTwin is vendored under `third_party/RoboTwin` for easier integration with this project.
- Unused upstream policy implementations under `policy/` may be removed for redistribution.
- Additional local changes may be applied to adapt RoboTwin to this repository.
