# Official runtime and interruption handling

All 54 modes use the same configured official evaluator and the declared SDK
revision. Machine locations are bound in ignored `deploy.local.json` or explicit
location variables; installations use `install.sh` without changing the
required versions or external content checks. WAM inference keeps its own model environment and
EE action ABI; Rule modes keep their joint workflows. The SDK owns layouts,
episode counts, cameras, physics, horizons, success and reward.

Each task skill's `SKILL.md` documents its operation, inputs, results and calling
graphs or modules. These are maintenance instructions for people and development
agents. Runtime validation checks that declared skills have the file, but does
not read its body, send it to a model or select skills from its prose. Executable
bindings come from `composition.json`, `workflow.json` and the referenced Python
implementations. Updating a skill description alone does not change its actions.

The policy server supplies decoded RGB arrays. Policy inference keeps those
channels in RGB order, including the seven-channel RGB/difference input to the
sorting classifier. Camera/trajectory buffers use only XPolicyLab's official
codec; the policy never decodes them. `rsi/image_files.py` separately handles
authenticated static public PNG assets and local diagnostic image files. Its
file loader never receives observations or trajectory buffers, and its previews
are ordinary RGB media, never serialized XPolicyLab image-bit fields. Neither
file operation changes the frames used for perception or model inference.

The common launcher acquires standard physical GPU EX leases before models or
Isaac start. It selects one installed NVIDIA Vulkan ICD and queries Vulkan device
UUIDs and PCI addresses in a short-lived process. The renderer index uses Kit
GPU Foundation PCI ordering, resolved from the leased NVML UUID: a physical card number is not necessarily its Vulkan index. Ambiguous,
duplicate or missing device identities fail before model startup. No shared
driver files or environments are modified. `graphics-runtime.json` records the
selected driver and mapping in the external run output.

The current launcher owns the SDK client bootstrap even when an explicit
`policy_source` preserves an older immutable actor. Its independent Git identity
is recorded. Policy code, model settings and action ABI still come from that
actor source. `official-client-attempt-N.log` records each client process and is
also mirrored into the campaign log.

An SDK reset or new-episode marker discards the previous episode's action
counter. Only counters emitted after that marker describe the next episode: an
old terminal `1050/1050` cannot classify its first slow plan as a terminal stall.
This phase change creates no score or action progress, and a reset alone does
not clear an explicit CUDA failure.

The pinned official shell admits only bounded fatal client restarts (99/134/139).
The common launcher retains the same policy server, EX descriptors, public run
configuration and official result namespace. The SDK alone loads its checkpoint
and filters completed cases. Every previously scored success and failure must
remain unchanged. Normal completion requires the actual default result count;
a zero process exit alone does not mean a complete cohort.

Kit can terminate the process inside `SimulationApp.close()` before the SDK's
fatal handler reaches `os.execv` or `exit(99)`. That specific zero-exit case is
admitted only with the pinned SDK's fatal log marker and its matching incomplete
native checkpoint/public run identity. Policy exceptions, timeouts, arbitrary
zero exits or missing/mismatched checkpoints do not enter this path. Restart
attempts remain bounded and are recorded before a new client starts. This does
not promise recovery from every driver/hardware fault, uninterrupted actor RNG,
or permit replaying graded cases.

Aggregate task/campaign wall caps default to disabled. An explicit operator cap
is separate from the unchanged official action horizon. Startup, transport and
fatal-restart admission checks remain: removing them would hide broken processes.

For Rule bootstrap, an explicit Kit hang after its initial-reset message is
reported if no official episode, action or score has begun and the first Rule
runner directory has not been created. The explicit fault must persist for five
minutes. A runner creates that directory before perception/planning, so a slow
first plan is excluded, as are later resets and learned initialization. This is
failure detection, not an automatic retry or a task wall cap. An interrupted
bootstrap retains its diagnostics and absence of official scores; a separately recorded recovery
must preserve the actor and public seed/count. A scored invocation still requires
the authenticated SDK-checkpoint continuation described below.

For an already interrupted invocation, the programmatic `resume_from` path also
requires the same clean actor source, public deployment and original run ID; a
matching official result/checkpoint; and confirmed former-owner death on its
actual compute node. Acquire a fresh lease, preserve all old diagnostics, and
validate the scored prefix after continuation. An old idle snapshot is not a
lease. Never stop foreign processes or edit scores/layout IDs to fill gaps.

The installed SDK is never patched. The duplicate ICD diagnosis and distinction
between Vulkan and NVML numbering follow NVIDIA's
[Linux troubleshooting guide](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html).
Runtime fixes need a bounded native camera/action check in addition to CPU
contracts; neither one establishes a full task success rate.
