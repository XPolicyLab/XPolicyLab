from pathlib import Path
import json,hashlib,py_compile
k=Path('/ML-vePFS/model_partner/ken');p=k/'RoboDawn/evaluation/policies/vlm_agent_rsi1';c=k/'RoboDawn/task_control/pick_from_conveyor_by_image';bank=p/'icl_bank_v2/pick_from_conveyor_by_image'
assert json.loads((c/'policy.json').read_text())['paused'] is True
f=p/'agent.py';s=f.read_text();backup=c/'agent_before_full_icl_v2.py';assert not backup.exists();backup.write_text(s)
anchor="        if self.cfg.icl['demo_count'] == 0:"
assert s.count(anchor)==1
hook='''        if task == "pick_from_conveyor_by_image":
            from .full_round_icl import load_messages
            bank = Path(__file__).parent / "icl_bank_v2" / task
            self.icl_msgs, proof = load_messages(bank,
                max_images=self.cfg.icl.get("max_images"),
                max_content_bytes=self.cfg.icl.get("max_content_bytes"))
            self.icl_demos = []
            self.icl_input_proof = proof
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                (self.log_dir / "full_icl_input_construction.json").write_text(json.dumps(proof, indent=2))
            return
'''
s=s.replace(anchor,hook+anchor);f.write_text(s)
f=p/'conveyor_coverage.py';s=f.read_text();anchor="    report=rebuild()\n    selected=choose(report['catalogue'],report['successes'],attempts())";assert s.count(anchor)==1
s=s.replace(anchor,"    bank=Path(__file__).parent/'icl_bank_v2'/TASK/'manifest.json'\n    if load(bank,{}).get('status')!='complete':raise RuntimeError('Full-round ICL v2 incomplete; source review required, no keyframe fallback')\n"+anchor);f.write_text(s)
rule={'status':'blocked_incomplete_source','new_episode_authorized':False,'source_root_required':'independent persistent OBS ICL source archive, isolated by task/run/decision, outside failed-media cleanup','preserve':['every original model response with all fields','all original full-resolution action-before camera images','exact decision_id/native control_step and full state','submitted commands/flow and request receipts','parsed executed primitives and complete feedback/errors','terminal observation labelled non-decision','full source SHA256 and export coverage manifest','actual desktop read/view coverage receipt distinct from constructed input'],'current_complete_source_exists':False,'capture_pipeline_completeness':'not yet validated; must verify before next authorized run','do_not_reconstruct_deleted_plans_from_hdf5':True,'do_not_restore_legacy_robodawn_demos':True,'protect_source_until_final_copy_sha_verified':True}
(c/'full_icl_source_preservation.json').write_text(json.dumps(rule,indent=2))
for name in ('agent.py','conveyor_coverage.py','full_round_icl.py'):py_compile.compile(str(p/name),doraise=True)
r={'task':'pick_from_conveyor_by_image','source_run':'robodawn-pick_from_conveyor_by_image-20261001-135129','bank':str(bank.parent),'loader':str(p/'full_round_icl.py'),'source_status':'incomplete','decisions_total':12,'complete_rounds_exported':0,'local_annotated_images':36,'raw_images_exported':0,'actual_full_v2_display_proven':False,'loader_tests':4,'online_validation':False,'paused':True,'new_gpu_run_started':False,'official_video_bank_preserved_but_not_used_as_silent_fallback':True,'future_capture_preservation':'requirements saved; implementation/validation pending before reactivation','modified_files':{str(p/name):hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ('agent.py','conveyor_coverage.py','full_round_icl.py')}}
(c/'icl_v2_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
