from pathlib import Path
import json
k=Path('/ML-vePFS/model_partner/ken');p=k/'RoboDawn/evaluation/policies/vlm_agent_rsi1';c=k/'RoboDawn/task_control/pick_from_conveyor_by_image'
f=k/'RoboDawn/run_conveyor_ken.sh';s=f.read_text().replace('TASK=${1:-general_pickup}','TASK=${1:-pick_from_conveyor_by_image}\n[[ "$TASK" == "pick_from_conveyor_by_image" ]] || exit 2');f.write_text(s)
import sys
sys.path.insert(0,str(p))
import conveyor_coverage as m
r=m.rebuild();next_one=m.choose(r['catalogue'],r['successes'],m.attempts())
try:m.preflight(0);raise AssertionError('pause guard failed')
except RuntimeError as e:assert 'Paused' in str(e)
e={'tests_passed':11,'mode':'offline only','pause_guard':'verified rejection before run directory / GPU creation','next_candidate_not_launched':next_one,'old_run_logs_exists':(k/'run_logs').exists(),'coverage_counts':{x:r[x] for x in ('successful_files','known_unique_combinations','known_duplicate_files','unknown_identity_files')},'cases':['skip successful combination','no wrap to0','fresh before failures','retry failure with no success','unknown seed conservative exclusion','all unknown fail closed','version not new identity','pending successful delivery not retried','pause','actual reset mismatch','native layout7 remains7']}
(c/'dedup_test_evidence.json').write_text(json.dumps(e,indent=2))
(c/'OPERATIONS.txt').write_text('Paused. No new GPU authorization. Before a future authorized episode, coordinator must explicitly clear policy.json paused flag and assign a GPU slot. Launcher preflight rebuilds coverage from final HDF5 and native result metadata. Unresolved seed remains unknown; layout0 is conservatively excluded across seeds. Do not invent old run_logs data. New runs go under this task-control/runs only. Publish successful validated data via private conveyor_coverage.py publish TASK EPISODE ARCHIVE SOURCE_ROOT SOURCE_EPISODE; this refreshes coverage immediately. Actual seed/layout are stored in future HDF5 and lightweight attempts. One task slot must remain serialized; no concurrent launch of this task. File content hashes are provenance, not geometry coverage. Old run_logs evidence was user-deleted; consult preserved collected_data only.\n')
print(json.dumps(e,indent=2))
