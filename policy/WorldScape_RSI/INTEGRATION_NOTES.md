# Company dev integration candidate, 2026-10-11

Install this folder as policy/WorldScape_RSI on the existing company dev branch.
Do not replace policy/WorldScape_Policy_2. No new Git branch is required.

Initial instruction + RGB + measured state -> episode-level local binary router -> WAM or Astra.
WAM reuses the adjacent WorldScape_Policy_2 Model, RoboDojo config, checkpoint resolver,
image transforms, action conversion and explicit memory commit/discard lifecycle.
One server process per environment; multi-GPU batching is deliberately disabled here.
Astra retains the existing command-envelope API, simulation-side motion executor,
original primitives, 43 skill files and 42-task ICL bank. No chunk switching or automatic
WAM-to-Astra finish helper is enabled. No reward enters the router features.

Server: WRSI_WAM_CHECKPOINT=/absolute/path/to/robodojo-checkpoint;
WRSI_JEV_CHECKPOINT=/absolute/path/to/trained-jev.pt; DEXBOTIC_SECRETS and VLM_AGENT_PROFILE
for Astra. WRSI_WAM_URL and WRSI_WAM_LOCK are no longer used by the active path.
Optional deploy.yml wam keys: checkpoint_path, device, worldscape_root only.
Checkpoint transforms and RoboDojo profile remain owned by the WAM adapter.
Client: WRSI_OUTPUT_DIR points to writable episode results. Both hosts need identical code.

No trained JEV weights are bundled. Random routing remains rejected. For explicit
development ablations only, use WRSI_ROUTE_MODE=wam or astra.
Restore ICL assets with import_icl_assets.py and the v2 ICL assets ZIP. Images remain
outside Git; every asset is hash-verified; the v2 archive includes the new balls demonstration.

Historical harness/wam_executor.py is retained as provenance, but the active episode
loop now calls wam_loop.py. Do not launch the historical WAM service for this entry.
The public Astra get_action command envelope still needs the custom deploy.py;
do not use a generic joint-action debug client as a substitute.
