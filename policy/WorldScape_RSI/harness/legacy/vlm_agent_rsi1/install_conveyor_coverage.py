from pathlib import Path
import shutil,json,hashlib,py_compile
K=Path('/ML-vePFS/model_partner/ken');P=K/'RoboDawn/evaluation/policies/vlm_agent_rsi1';C=K/'RoboDawn/task_control/pick_from_conveyor_by_image'
C.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:cmd=(proc/'cmdline').read_bytes().decode(errors='replace')
 except OSError:continue
 if 'src/eval_client/main.py' in cmd and 'pick_from_conveyor_by_image' in cmd:raise RuntimeError('Episode active')
policy=C/'policy.json'
if not policy.exists():policy.write_text(json.dumps({'paused':True,'reason':'Ken global pause; only coordinator may resume after authorization','single_task_slot':True},indent=2))
launcher=K/'RoboDawn/run_conveyor_ken.sh';original=launcher.read_text()
backup=P/'coverage_preinstall';backup.mkdir(exist_ok=False)
shutil.copy2(launcher,backup/launcher.name)
gate='''# Persistent scene coverage preflight: BEFORE run directories or GPU processes.
"$ROOT/envs/rpent_robodojo/bin/python" "$ROOT/RoboDawn/evaluation/policies/vlm_agent_rsi1/conveyor_coverage.py" preflight "$SEED"
'''
original=original.replace('RUN="$ROOT/run_logs/',gate+'RUN="$ROOT/run_logs/',1)
original=original.replace('bash "$ROOT/RoboDojo/scripts/eval_policy.sh"','bash "$ROOT/RoboDawn/evaluation/policies/vlm_agent_rsi1/conveyor_eval_policy.sh"')
launcher.write_text(original)
client=K/'RoboDojo/src/eval_client/main.py'
text=client.read_text();anchor='    env = create_eval_env(env_cfg, simulation_app, resume_state=resume_state)'
assert text.count(anchor)==1
text=text.replace(anchor,anchor+'\n    from XPolicyLab.policy.vlm_agent_rsi1.conveyor_coverage import apply_queue\n    apply_queue(env)')
text=text.replace('import sys\n','import sys\nsys.path.insert(0,"/ML-vePFS/model_partner/ken/RoboDojo")\n',1)
(P/'conveyor_eval_client.py').write_text(text)
shell=(K/'RoboDojo/scripts/eval_policy.sh').read_text().replace('python -u src/eval_client/main.py',f'python -u {P}/conveyor_eval_client.py')
(P/'conveyor_eval_policy.sh').write_text(shell)
rec=P/'sft_recording.py';shutil.copy2(rec,backup/rec.name);text=rec.read_text()
text=text.replace('        self.rows = 0','''        from .conveyor_coverage import identity
        for key, value in identity(self.env).items():
            if value is not None:
                self.sft.attrs[key] = value
        self.rows = 0''',1)
text=text.replace('        self.sft.close()','''        from .conveyor_coverage import record
        record(self.env, self.sft_path, dict(self.sft.attrs), str(reason))
        self.sft.close()''',1)
rec.write_text(text)
for name in ('conveyor_coverage.py','conveyor_eval_client.py','sft_recording.py'):py_compile.compile(str(P/name),doraise=True)
report={'scope':'private conveyor launcher/client/recorder only','paused':True,'gpu_started':False,
 'official_seed_manager':'sorted matching files enumerated from0; seed chooses official layout directory',
 'official_client_source_sha256':sha(client),'official_shell_source_sha256':sha(K/'RoboDojo/scripts/eval_policy.sh'),
 'paths':[str(launcher),str(P/'conveyor_coverage.py'),str(P/'conveyor_eval_client.py'),str(P/'conveyor_eval_policy.sh'),str(rec)],
 'control_directory':str(C),'publication_command':'python conveyor_coverage.py publish TASK EPISODE ARCHIVE SOURCE_ROOT SOURCE_EPISODE',
 'online_validated':False,'old_run_logs_restored':False}
(C/'installation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
