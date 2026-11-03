RSIWAM: local RoboDojo inference submission candidate
Prepared 2026-11-01. Display name: RSIWAM.

Release identity
  Candidate: recoverwam_success133_original_parent20261032_lora20261033
  Frozen official VPP2 joint2b_s100000 base, frozen RecoverWAM original
  recovery parent seed 20261032, then success133 Action-LoRA seed 20261033,
  committed optimizer step 400. Naming the method RSIWAM does not change its
  upstream origin, technical identity, training history or measured results.

  Base bundle manifest SHA256:
    ae66dbfc16eb3c75fa89111501412ecb14cd3ab62ce02ff9fc8a457bb3b19aba
  Parent weights/parent/adapter_final.pt SHA256:
    861529eb89e4279e4490d6156bf04a335eee410f6c02842d0ac3db2dc90431d3
  Action-LoRA weights/step000400/adapter.pt SHA256:
    eec21ec2834ef95bffdac7b3df0b48092a215aa67e7bf6952ec1643ab511b361
  Checkpoint metadata SHA256:
    38b9eaa31383a8a098dc68e568c125d5d22d3bb749742145ac38d7c2a5d3dbb6

What is included
  policy/RSIWAM: standard XPolicyLab model/deploy/config/install/eval/server/
    client interface. This is a NEW portable wrapper, awaiting full validation.
  src/recoverwam: exact 100-file registered Python source tree, SHA256 95feb351
    fc1211318d7edec9e3814679835029b7ba4314d04d25464b6a4691f0. Compute the digest
    from compact sorted JSON mapping paths relative to src/recoverwam to each
    Python file's SHA256, as in the original training verifier.
  runtime/recovery_action_lora.py: unmodified registered composition helper.
  runtime/rsiwam_native_wrapper.py and policy/RSIWAM/deploy.py: exact source
    bytes verified against the existing H800 deployment adapter, retained with
    original RecoverWAM internal identifiers and observation/history handling.
  reference_switch: unmodified switch_runtime.py and continuation_contract.py
    for reviewing original composition/admission behavior. This is a reference
    snapshot, not a portable shared-model scheduler or an active controller.
  weights: exact parent, recipe, adapter and committed checkpoint metadata.
    Optimizer training_state.pt and continuation sidecars referenced by that
    metadata are not bundled. They are not needed by this inference loader;
    this package alone is not a full training-resumption archive.
  base_weights.json: exact base filenames, byte counts where available, SHA256
    values, repository/revision locators. The full base is NOT bundled.
  evidence: byte-exact base manifest and dataset normalizer metadata.
  checks: reproducible CPU checks and verification results.

Loading order
  1. Verify local source, helper, native wrapper, parent, recipe, checkpoint
     and adapter hashes. Verify every external base asset hash and size.
  2. Construct the unadapted native VPP2 base policy using the registered
     RecoverWAM implementation and official base dataset_stats.json.
  3. Load the frozen original RecoverWAM parent, bound to the base identity.
  4. Install rank-8 / alpha-8 Action-LoRA on action self-attention q and v,
     then load exact A/B tensors and validate committed training metadata.
  5. Set recovery_mode=auto, freeze all parameters, enter eval mode and disable
     action/MoT gradient checkpointing. No training occurs on startup.
  This follows switch_runtime.load_candidate. The new entry is single-model;
  it does not bypass or replace any live shared-service admission/controller.

Prerequisites and installation (commands have NOT been executed here)
  Use an independently installed RoboDojo/XPolicyLab checkout and a separate
  policy environment. The simulator/assets remain separately installed under
  their own terms. The observed policy reference is torch 2.11.0+cu130 and
  torchvision 0.26.0+cu130. requirements.txt lists source dependencies;
  constraints-reference.txt records the packages bound by the H800 reference
  configuration. It is a partial constraint set, not a complete lock file.
  Local XPolicyLab reference checkout HEAD: 183b11a9528f2aa1dc6ca295480dcf2772b3bd7f.
  This records the inspected checkout, not a claim that all upstream files were
  pristine or that a clean checkout was tested.

  Keep this whole runtime_package directory together, then set:
    export RSIWAM_PACKAGE_ROOT=/absolute/path/runtime_package
    export RSIWAM_XPL_ROOT=/absolute/path/RoboDojo/XPolicyLab
    export RSIWAM_BASE_WEIGHTS=/absolute/path/checkpoints

  Make the standard policy import visible (only if that destination is absent):
    ln -s "$RSIWAM_PACKAGE_ROOT/policy/RSIWAM" "$RSIWAM_XPL_ROOT/policy/RSIWAM"
  Copying policy/RSIWAM alone is unsupported: Path.resolve() locates the complete
  package's src/runtime/weights siblings through the symlink. The launcher checks
  that XPolicyLab's registered policy path resolves to this exact directory.

  Activate the intended policy environment, install the matching CUDA torch
  wheel pair, then:
    bash "$RSIWAM_PACKAGE_ROOT/policy/RSIWAM/install.sh"

  RSIWAM_BASE_WEIGHTS must contain joint2b_s100000/{action.pt,video.pt,
  dataset_stats.json,manifest.json} and Wan2.1-I2V-14B-480P with the three
  encoder/VAE files and four tokenizer files in base_weights.json.
  Public URLs are pinned repository/revision locators derived from the frozen
  runtime provenance; they were not downloaded or HTTP-verified in this task.
  No private host endpoint, credential, private key or scheduler token is needed
  by the portable entry. Historical checkpoint provenance retains internal
  filesystem paths; these are evidence, not executable download locations.

  launch_policy.py exports CUBLAS_WORKSPACE_CONFIG=:4096:8 before the policy
  subprocess imports torch or initializes CUDA, matching the existing H800
  launch configuration. A direct Model constructor also requires this value
  to have been set before startup. Deterministic torch algorithms are enabled.

Data and inference contract
  RoboDojo / arx_x5 / absolute EE16 only. Inputs use vision.cam_head.color,
  vision.cam_left_wrist.color and vision.cam_right_wrist.color, plus state:
  left_ee_pose[7], left_ee_joint_state[1], right_ee_pose[7],
  right_ee_joint_state[1]. Instructions follow the native adapter.
  Camera composition is the official T-shape 240x416; history is 8 frames at
  stride 25. Action horizon 32; execute 24 actions per replan; 10 denoising
  steps; shift 1; model seed 1; BF16; automatic recovery gate threshold 0.9.
  Returned actions have the same four EE16 keys. The client records one native
  observation per executed action and supplies a per-episode client ID.
  The eval.sh positional seed controls the environment evaluation and may differ
  from 1. It is passed to the client; it never overrides the model inference seed.
  This release does not provide a new data conversion or full training command.
  Its exact checkpoint lineage is preserved: 119 online optimizer updates then
  281 exact-state cache-continuation updates, totaling 400. The parent had 160
  training updates and is frozen during Action-LoRA training/inference.

CPU review and intended evaluation
  With the package as current directory:
    PYTHONDONTWRITEBYTECODE=1 python3 checks/check_structure.py
    PYTHONDONTWRITEBYTECODE=1 python3 checks/check_launch_contract.py
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python checks/check_adapter_cpu.py
  The last command needs torch, but loads only the two small real artifacts
  into a synthetic structural skeleton whose base weights are scalar views.
  It never evaluates the base forward pass or initializes CUDA.

  Inspect the launcher without starting a server:
    python policy/RSIWAM/launch_policy.py --task stack_bowls --dry-run

  After installation and external base staging, intended local debug command:
    EVAL_ENV_TYPE=debug bash policy/RSIWAM/eval.sh RoboDojo stack_bowls rsiwam-original32-lora33-step400 arx_x5 ee 1 0 0 POLICY_ENV EVAL_ENV
  For simulation, set EVAL_ENV_TYPE=sim with the same argument structure.
  POLICY_ENV and EVAL_ENV are actual conda environment names; select appropriate
  policy/environment GPU IDs. These commands load the intended real checkpoint;
  they are documentation, not claims that either evaluation has run.

Verification scope and remaining work
  checks/verification.json records the checks performed here. Source syntax,
  shell syntax, artifact hashes, source-tree identity and real adapter/parent
  CPU structural composition have been checked. The CPU fixture verifies all
  160 A/B tensors, 80 q/v LoRA modules, 3,932,160 LoRA parameters, 1,597,697 parent
  parameters, unchanged base storage, eval mode and frozen parameters.
  This was not a clean environment install, full-model load, CUDA action parity
  check, official debug evaluation, or closed-loop simulator evaluation.
  Before submitting: resolve/download and verify all base locators; reproduce
  the policy environment; test the new XPolicyLab entry with real checkpoints;
  compare actions/history/client isolation to the accepted source runtime;
  run the required intended-checkpoint debug and official evaluation workflow;
  confirm redistribution terms for weights; then review and publish the repo/PR.
  No official leaderboard score is claimed by this package. Existing screening
  results belong in the separate submission evidence, with their original scope.

Licensing and provenance
  LICENSE, THIRD_PARTY_NOTICES.txt, licenses/ and retained source headers preserve
  FastWAM, Wan2.1/DiffSynth, LeRobot, RoboDojo and XPolicyLab terms. The upstream
  VPP2 base identity and citation records remain in provenance/ and manifest.json.
  The code license does not assign a license to model weights or datasets.
  Newly added portability files are identified separately in manifest.json.
