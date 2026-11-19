"""CPU-only admission for an explicitly bound online-to-cache execution lineage.

An old online checkpoint, even with a relabeled producer flag, is never accepted.
The source and continuation identities are checked in full, including all fields
unknown to this evaluator. Runtime evidence is checked independently of identity.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

SOURCE_FORMAT = 'recoverwam-success133-action-lora-training-v1'
FORMAT = 'recoverwam-success133-native-cache-continuation-v1'
QUEUE_SCHEMA = 'recoverwam-h800-cache-continuation-first400-queue-v1'
PRODUCER = 'vpp2-robodojo-action-lora-v1'
CONSUMER = 'recoverwam-robodojo-action-lora-v1'
MODE = 'EXPLICIT_FORMAT_ADAPTER_COMPLETE_CACHE_EXACT_NATIVE_ACCEPTANCE_REQUIRED'
ONLINE_TRAINER = '808bc58925fa39e580809569c031a79b8347a984c63d0d2b4831a15b525f321b'
CONTINUATION_TRAINER = 'e9e14a0a9e6d7ffd78ea583b0fa701f7fc962dc2fb468f1c34b899f3c703675a'
CONTINUATION_SUPPORT = '1d95436acf8ae4598c5ad59792573824bd435d8339bfb0691863f000ba6b5d92'
ONLINE_ENCODER = '7adb4deb019c849731d086d0885d727e363008e2cc443f5e3fa5fb73c1b21ba4'
ONLINE_SCOPES = {
    'recoverwam_success133_original_parent20261032_lora20261033': 'f68004790c2b25fac3940234ebd1ba8c4885f4ab2277df0e809b88df8033d22f',
    'recoverwam_success133_state_anchor_v2_parent20261032_lora20261033': '0acc3dfdd8e6ac5db922ecd78176fcacb9d39906ff5d6c9487061c4dcc5c764c',
}
SIDECARS = (
    'source_handoff_checkpoint.json', 'continuation_manifest.json',
    'feature_transfer_evidence.json', 'acceptance_zero_adapter.json',
    'resume_parity.json', 'source_training_prefix.jsonl',
    'continuation_training_suffix.jsonl',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def json_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def hash_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def checked(binding):
    require(set(binding) == {'path', 'sha256'} and hash_string(binding['sha256']), 'Invalid file binding')
    path = Path(binding['path'])
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents))
            and path.is_file() and sha(path) == binding['sha256'], 'Bound file is missing, aliased, or changed')
    return path


def validate_source(identity, candidate_id):
    require(candidate_id in ONLINE_SCOPES and identity.get('candidate_id') == candidate_id,
            'Only the registered original online pair may continue')
    expected = dict(format=SOURCE_FORMAT, trainer_sha256=ONLINE_TRAINER,
        producer_mode='same_runtime_online_official_encoders', online_feature_module_sha256=ONLINE_ENCODER,
        online_scope_sha256=ONLINE_SCOPES[candidate_id], feature_cache_sha256=None,
        compatibility_sha256=None, shared_merged_cache_accepted=False,
        initial_lora_reference_mode='same_device_official_helper_reference')
    require(all(k in identity and type(identity[k]) is type(v) and identity[k] == v
                for k, v in expected.items()), 'Source is not the original online lineage')
    require(hash_string(identity.get('initial_lora_sha256')), 'Source initializer tensor identity missing')


def validate_identity(identity, row):
    source = row['expected_source_identity']
    validate_source(source, row['candidate_id'])
    require(canonical(identity) == canonical(row['expected_continuation_identity']),
            'Full continuation scientific/execution identity differs')
    require(identity.get('format') == FORMAT and identity.get('shared_merged_cache_accepted') is True
            and identity.get('producer_mode') == MODE, 'Old online provenance is not a cached continuation')
    for key in ('trainer_sha256', 'feature_cache_sha256', 'compatibility_sha256', 'online_feature_module_sha256'):
        require(hash_string(identity.get(key)), 'Missing actual continuation binding: ' + key)
    require(identity['trainer_sha256'] == CONTINUATION_TRAINER
            and identity['online_feature_module_sha256'] == CONTINUATION_SUPPORT,
            'Continuation must use the frozen exact-resume r3 scientific code')
    lineage = identity['execution_lineage']
    require(lineage.get('schema') == 'recoverwam-online-exact-continuation-lineage-v1'
            and canonical(lineage.get('source_training_identity')) == canonical(source)
            and lineage.get('source_training_identity_sha256') == json_sha(source), 'Full original identity was lost')
    n = lineage.get('source_optimizer_step')
    require(type(n) is int and 1 <= n < 400 and lineage.get('first_resumed_step') == n + 1
            and type(lineage.get('total_optimizer_updates')) is int and lineage['total_optimizer_updates'] == 400
            and lineage.get('new_optimizer_updates') == 400 - n, 'Source prefix plus continuation must total400')
    require(lineage.get('source_results_preserved') is True
            and lineage.get('state_policy') == 'RESTORE_EXACT_ADAPTER_ADAMW_PYTHON_NUMPY_CPU_CUDA_RNG_AND_NEXT_DRAW'
            and lineage.get('producer_format') == PRODUCER and lineage.get('consumer_format') == CONSUMER
            and lineage.get('tensor_or_label_conversion') is False, 'Continuation changes state or tensor semantics')
    require(lineage.get('execution_variant') == 'b1_checkpoint_off_useful_kv_prefetch',
            'Only the exact sidecar-qualified B1 useful-KV variant is admitted')
    require(lineage.get('compatibility_manifest_sha256') == identity['compatibility_sha256']
            and lineage.get('support_sha256') == identity['online_feature_module_sha256'], 'Nested execution hashes differ')
    changed = {'format', 'trainer_sha256', 'producer_mode', 'feature_cache_sha256', 'compatibility_sha256',
               'online_feature_module_sha256', 'shared_merged_cache_accepted', 'execution_lineage'}
    require({k: v for k, v in identity.items() if k not in changed}
            == {k: v for k, v in source.items() if k not in changed}, 'Scientific identity changed during continuation')
    require(all(identity.get(k) == v for k, v in row['expected_identity'].items()),
            'Registered parent, initializer, seeds or schedule differs')
    return lineage


def validate_commit(row, commit):
    require(type(commit.get('step')) is int and commit['step'] == 400, 'Require committed total step400')
    validate_identity(commit['identity'], row)
    require(all(hash_string(commit.get(k)) for k in ('adapter_sha256', 'training_state_sha256')),
            'Missing committed adapter/AdamW-RNG file hashes')


def validate_steps(path, first, last, row):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    require([r.get('step') for r in records] == list(range(first, last + 1))
            and all(type(r.get('step')) is int for r in records), 'Optimizer log has a gap, duplicate, or wrong boundary')
    proof = row['schedule_proof']
    for record in records:
        pair = [{k: component[k] for k in proof['draw_fields']} for component in record['components']]
        require(len(pair) == 2 and json_sha(pair) == proof['pair_sha256'][record['step'] - 1],
                'Logged window, label, sampler order or noise seed differs from the frozen schedule')
    return records


def strict_zero(value):
    return type(value) in (int, float) and value == 0


def validate_native_evidence(transfer, zero, identity, manifest):
    require(transfer.get('status') == 'PASS_EXACT_CANONICAL_CACHE_NATIVE_REENCODING'
            and transfer.get('source_cache_sha256') == identity['feature_cache_sha256']
            and transfer.get('shared_merged_cache_accepted') is True,
            'Fresh real-RGB canonical feature acceptance missing or mismatched')
    ids = manifest['native_rgb_acceptance_window_ids']
    require(len(ids) == 3 and len(set(ids)) == 3, 'Require exactly three frozen native RGB windows')
    samples = transfer.get('samples', [])
    require(len(samples) == 3 and {r['window_id'] for r in samples} == set(ids),
            'Native RGB acceptance windows differ')
    for row in samples:
        differences = row.get('differences', {})
        require(row.get('normalization_exact') is True
                and set(differences) == {'condition_latents', 'clip_fea', 'context'}
                and all(set(v) == {'max_abs_error', 'rms_error'}
                        and all(strict_zero(number) for number in v.values()) for v in differences.values()),
                'Feature or normalization comparison is not bitwise exact')
    require(zero.get('initial_lora_sha256') == identity['initial_lora_sha256']
            and zero.get('compared_to') == 'EXACT_SAME_PARENT_ORIGINAL_NATIVE_INFERENCE'
            and zero.get('modes') == ['off', 'on', 'auto'], 'Wrong zero-LoRA/native-parent comparison')
    checks = zero.get('checks', [])
    require(len(checks) == 6, 'Require fresh RGB and cached parity for all three native windows')
    seen = set()
    for row in checks:
        require(row.get('window_id') in ids and len(row.get('checks', [])) == 3, 'Wrong parent parity window')
        modes, kind = set(), set()
        for check in row['checks']:
            require(check.get('mode') in ('off', 'on', 'auto')
                    and check.get('zero_lora_exact') is True and check.get('same_parent_native_exact') is True
                    and type(check.get('real_rgb')) is bool and strict_zero(check.get('max_abs_error')),
                    'Native parent action parity is not exact')
            modes.add(check['mode']); kind.add(check['real_rgb'])
        require(modes == {'off', 'on', 'auto'} and len(kind) == 1, 'Repeated or mixed parent parity modes')
        key = (row['window_id'], next(iter(kind)))
        require(key not in seen, 'Repeated parent parity window/mode evidence')
        seen.add(key)
    require(seen == {(name, rgb) for name in ids for rgb in (True, False)}, 'Missing native or cached parent parity')


def validate_handoff(manifest, evidence):
    """Preserve root's completed source boundary; preparation checks exact raw bytes."""
    status, request = evidence['status'], evidence['request']
    cid = manifest['source_training_identity']['candidate_id']
    n = manifest['source_optimizer_step']
    require(manifest.get('source_handoff_status_binding') == evidence['status_binding']
            and manifest.get('source_handoff_request_binding') == evidence['request_binding'],
            'Manifest handoff differs from frozen admitted source evidence')
    require(status.get('phase') == 'paused_checkpoint_boundary'
            and type(status.get('optimizer_updates')) is int and status['optimizer_updates'] == n
            and status.get('candidate_id') == cid and request.get('candidate_id') == cid
            and isinstance(request.get('request_id'), str) and request['request_id'],
            'Source did not finish the explicitly requested checkpoint boundary')
    require(status.get('handoff', {}).get('sha256') == evidence['request_binding']['sha256']
            and status['handoff'].get('request') == request,
            'Source did not acknowledge the exact root handoff request')
    require(str(Path(status['final_checkpoint'])/'checkpoint.json') == manifest['source_checkpoint_binding']['path']
            and manifest.get('execution_relationship') == 'CONTINUE_EXACT_ROOT_REQUESTED_LATEST_COMMITTED_HANDOFF'
            and manifest.get('acceptance_only') is False and manifest.get('execution_authorized') is True,
            'Diagnostic or earlier checkpoint replay cannot become formal continuation')
    return evidence


def exact_comparison(value):
    return value.get('exact') is True and strict_zero(value.get('max_abs')) and value.get('mismatches') == []


def validate_gate_coverage(resume, manifest, row):
    selections = manifest.get('gate_coverage_pairs', [])
    reports = resume.get('gate_coverage', [])
    expected = [(15, 'recovery'), (30, 'normal')]
    require([(v.get('schedule_optimizer_step'),v.get('expected_active_role')) for v in selections] == expected
            and [(v.get('schedule_optimizer_step'),v.get('expected_active_role')) for v in reports] == expected,
            'Both predeclared real recovery/normal gate-on pairs are required')
    n = manifest['source_optimizer_step']; proof = row['schedule_proof']
    for selection, report, (step, role) in zip(selections, reports, expected):
        pair = selection['pair']; pair_hash = json_sha(pair)
        require(len(pair) == 2 and pair_hash == proof['pair_sha256'][step-1]
                and report.get('registered_pair') == pair and report.get('registered_pair_sha256') == pair_hash,
                'Gate coverage changed original registered draws, noise, labels or pair identity')
        require(report.get('schema') == 'recoverwam-nonzero-lora-gate-coverage-v1'
                and report.get('status') == 'PASS_EXACT_REAL_GATE_ON_UPDATE_AND_ROLLBACK'
                and report.get('source_optimizer_step') == n and report.get('comparison_optimizer_step') == n+1
                and report.get('restored_sampler_next_draw') == n*2
                and type(report.get('nonzero_source_lora_B_elements')) is int and report['nonzero_source_lora_B_elements'] > 0
                and type(report.get('optimizer_updates_committed')) is int and report['optimizer_updates_committed'] == 0
                and report.get('original_training_cursor_advanced') is False,
                'Gate coverage used zero LoRA or changed the actual training cursor')
        for name, fields in [('comparisons', {'parameters','clipped_gradients','adam','rng','result'}),
                             ('restoration_comparisons', {'parameters','adam','rng'})]:
            values = report.get(name, {})
            require(set(values) == fields and all(exact_comparison(v) for v in values.values()),
                    'Gate-on update or rollback is not bitwise exact')
        byte_rows = report.get('native_cache_bytes', [])
        require(len(byte_rows) == 2 and set(selection['native_reference_bindings']) == {d['window_id'] for d in pair},
                'Gate coverage is missing native input references')
        for value, draw in zip(byte_rows, pair):
            require(value.get('window_id') == draw['window_id']
                    and value.get('native_binding') == selection['native_reference_bindings'][draw['window_id']]
                    and hash_string(value['native_binding']['sha256'])
                    and set(value.get('comparisons',{})) == {'features','action_raw','action_norm','action_mask','metadata','window_sha256','feature_sha256'}
                    and all(exact_comparison(v) for v in value['comparisons'].values()),
                    'Gate-on native/cache inputs differ')
        branches = report.get('branches', {})
        require(set(branches) == {'native_b1_checkpoint_on','cache_b1_checkpoint_off_two_consumed_conditions'},
                'Unexpected gate-on execution branch')
        native_result = branches['native_b1_checkpoint_on'].get('result')
        for name, branch in branches.items():
            require(0 < branch.get('peak_allocated_gib',0) < 70 and 0 < branch.get('peak_reserved_gib',0) < 70
                    and branch.get('no_artificial_memory_floor') is True
                    and branch.get('result') == native_result, 'Gate coverage changed actual result or exceeded memory ceiling')
            components = branch['result'].get('components', [])
            require(len(components) == 2
                    and json_sha([{k:c[k] for k in proof['draw_fields']} for c in components]) == pair_hash,
                    'Gate coverage result belongs to another pair')
            active = [c for c in components if c['role'] == role]
            require(len(active) == 1, 'Expected real gated role absent')
            gate = active[0].get('gate', {})
            probabilities, mask = gate.get('probabilities', []), gate.get('active', [])
            require(gate.get('threshold') == .9 and probabilities and len(probabilities) == len(mask)
                    and all(type(v) in (int,float) and 0 <= v <= 1 for v in probabilities)
                    and all(type(v) is bool for v in mask) and any(mask)
                    and mask == [v >= .9 for v in probabilities], 'Predeclared real gate-on path was not active')
            if name.startswith('cache'):
                require(branch.get('generated_conditions') == 2 and branch.get('consumed_conditions') == 2,
                        'Gate diagnostic must consume exactly two useful conditions')


def validate_bundle(directory, row, commit=None):
    """Verify every portable sidecar before import, enqueue, dispatch and switch."""
    directory = Path(directory)
    commit = read(directory / 'checkpoint.json') if commit is None else commit
    validate_commit(row, commit)
    for name, key in [('adapter.pt', 'adapter_sha256'), ('training_state.pt', 'training_state_sha256')]:
        checked(dict(path=str(directory/name), sha256=commit[key]))
    identity = commit['identity']; lineage = identity['execution_lineage']
    n = lineage['source_optimizer_step']
    sidecars = commit.get('continuation_sidecars', {})
    require(set(sidecars) == set(SIDECARS), 'All seven exact checkpoint evidence sidecars are required')
    for name in SIDECARS:
        value = sidecars[name]
        require(set(value) == {'filename', 'sha256', 'bytes'} and value['filename'] == name
                and type(value['bytes']) is int and value['bytes'] > 0, 'Invalid portable sidecar descriptor')
        path = checked(dict(path=str(directory/name), sha256=value['sha256']))
        require(path.stat().st_size == value['bytes'], 'Sidecar size differs')
    require(sidecars['source_handoff_checkpoint.json']['sha256'] == lineage['source_checkpoint_binding']['sha256']
            and sidecars['source_training_prefix.jsonl']['sha256'] == lineage['source_training_prefix_binding']['sha256']
            and sidecars['continuation_manifest.json']['sha256'] == identity['compatibility_sha256'],
            'Copied origin or compatibility bytes differ from immutable identity')
    origin = read(directory/'source_handoff_checkpoint.json')
    require(type(origin.get('step')) is int and origin['step'] == n
            and canonical(origin.get('identity')) == canonical(row['expected_source_identity'])
            and hash_string(origin.get('adapter_sha256')) and hash_string(origin.get('training_state_sha256')),
            'Wrong original committed optimizer/adapter boundary')
    manifest = read(directory/'continuation_manifest.json')
    require(manifest.get('schema') == 'recoverwam-online-cache-continuation-manifest-v1'
            and canonical(manifest.get('source_training_identity')) == canonical(row['expected_source_identity'])
            and manifest.get('source_optimizer_step') == n and manifest.get('total_optimizer_updates') == 400
            and manifest.get('source_results_preserved') is True and manifest.get('execution_authorized') is True
            and manifest.get('acceptance_only') is False, 'CPU-only or diagnostic manifest cannot qualify formal training')
    for key in ('source_checkpoint_binding', 'source_training_prefix_binding', 'source_trainer_binding', 'execution_variant'):
        require(manifest.get(key) == lineage[key], 'Manifest disagrees with execution lineage: '+key)
    require(manifest['cache_binding']['sha256'] == identity['feature_cache_sha256']
            and manifest['support_sha256'] == identity['online_feature_module_sha256']
            and manifest['continuation_trainer_sha256'] == identity['trainer_sha256']
            and manifest.get('prefetch') == dict(target_gib=60., ceiling_gib=65., hard_peak_gib=70., max_entries=32),
            'Actual cache, code, or bounded useful-prefetch policy differs')
    validate_handoff(manifest, row['expected_handoff'])
    proof = row['schedule_proof']
    require(proof['schedule_sha256'] == identity['schedule_sha256'] and len(proof['pair_sha256']) == 400
            and all(hash_string(h) for h in proof['pair_sha256']) and proof['draw_fields'], 'Missing original400 schedule proof')
    prefix = validate_steps(directory/'source_training_prefix.jsonl', 1, n, row)
    suffix = validate_steps(directory/'continuation_training_suffix.jsonl', n+1, 400, row)
    require(len(prefix) + len(suffix) == 400, 'Combined optimizer steps do not total400')
    for record in suffix:
        require(record.get('gpu_uuid') == row['expected_gpu_uuid']
                and record.get('frozen_base_parent_gradients_absent') is True
                and 0 < record.get('peak_allocated_gib',0) < 70 and 0 < record.get('peak_reserved_gib',0) < 70,
                'A formal update exceeded the70GiB ceiling or changed source GPU/frozen parameters')
        prefetch = record.get('useful_prefetch', {})
        cursor = (record['step'] - 1) * 2
        end = prefetch.get('next_build')
        require(type(end) is int and cursor + 2 <= end <= 800
                and prefetch.get('allocation_purpose') == 'REGISTERED_FUTURE_FROZEN_VIDEO_KV_CONSUMED_IN_ORIGINAL_ORDER'
                and prefetch.get('cursor') == cursor and prefetch.get('consumed') == cursor - n*2
                and prefetch.get('generated') == end - n*2 and prefetch.get('queued_future_draws') == end - cursor,
                'Continuation did not consume useful future KV in the exact original order')
    require(suffix[-1]['useful_prefetch']['next_build'] == 800,
            'Final update did not consume the final two registered draws')
    transfer = read(directory/'feature_transfer_evidence.json')
    zero = read(directory/'acceptance_zero_adapter.json')
    resume = read(directory/'resume_parity.json')
    validate_native_evidence(transfer, zero, identity, manifest)
    validate_gate_coverage(resume, manifest, row)
    require(transfer.get('schema') == 'recoverwam-native-cache-rgb-acceptance-v1'
            and resume.get('schema') == 'recoverwam-exact-next-update-resume-parity-v1'
            and resume.get('status') == 'PASS_EXACT_NATIVE_CACHE_PREFETCH_NEXT_UPDATE',
            'Missing actual exact GPU resume acceptance')
    for evidence in (transfer, resume):
        require(evidence.get('actual_gpu_uuid') == row['expected_gpu_uuid']
                and evidence.get('runtime_sha256') == row['expected_runtime_sha256']
                and evidence.get('continuation_manifest_sha256') == identity['compatibility_sha256']
                and evidence.get('source_checkpoint_binding') == lineage['source_checkpoint_binding'],
                'GPU evidence belongs to another runtime, checkpoint, manifest or GPU')
    require(isinstance(resume.get('torch_version'), str) and resume['torch_version']
            and resume['torch_version'] == transfer.get('torch_version'), 'GPU evidence runtime versions differ')
    require(canonical(resume.get('source_training_identity')) == canonical(row['expected_source_identity'])
            and resume.get('source_optimizer_step') == n and resume.get('next_optimizer_step') == n+1
            and resume.get('restored_sampler_next_draw') == n*2
            and resume.get('execution_variant') == lineage['execution_variant']
            and resume.get('source_state_restored_after_acceptance') is True
            and type(resume.get('optimizer_updates_committed_by_acceptance')) is int
            and resume['optimizer_updates_committed_by_acceptance'] == 0
            and resume.get('frozen_parent_versions_unchanged') is True,
            'Acceptance consumed an update, changed frozen state, or restored a different cursor')
    pair = resume.get('next_pair')
    require(isinstance(pair, list) and len(pair) == 2 and json_sha(pair) == proof['pair_sha256'][n],
            'Resume parity did not use the actual next two frozen draws')
    comparisons = resume.get('comparisons', {})
    require(set(comparisons) == {'parameters', 'clipped_gradients', 'adam', 'rng', 'result'},
            'Resume parity omits optimizer, parameters, gradients, RNG or result')
    require(all(exact_comparison(c) for c in comparisons.values()), 'Actual next update is not bitwise identical')
    restoration = resume.get('restoration_comparisons', {})
    require(set(restoration) == {'parameters', 'adam', 'rng'}
            and all(exact_comparison(c) for c in restoration.values())
            and resume.get('source_gradients_cleared_after_acceptance') is True,
            'Post-acceptance state restoration was not actually verified')
    byte_rows = resume.get('native_cache_bytes', [])
    require(len(byte_rows) == 2, 'Actual next native/cache inputs are not both verified')
    for evidence, draw in zip(byte_rows, pair):
        require(evidence.get('draw') == draw and evidence.get('window_id') == draw['window_id']
                and set(evidence.get('comparisons', {})) == {'features', 'action_raw', 'action_norm', 'action_mask',
                                                          'metadata', 'window_sha256', 'feature_sha256'}
                and all(exact_comparison(c) for c in evidence['comparisons'].values())
                and hash_string(evidence.get('feature_sha256')) and hash_string(evidence.get('window_sha256')),
                'Actual next input tensors, metadata, labels or masks differ')
    branches = resume.get('branches', {})
    require(set(branches) == {'original_online_b1_checkpoint_on', 'cache_b1_checkpoint_off_useful_kv_prefetch'},
            'Unknown comparison branch; B2 is not admitted')
    changed = branches['cache_b1_checkpoint_off_useful_kv_prefetch']
    prefetch = changed.get('useful_prefetch', {})
    require(prefetch.get('allocation_purpose') == 'REGISTERED_FUTURE_FROZEN_VIDEO_KV_CONSUMED_IN_ORIGINAL_ORDER'
            and prefetch.get('cursor') == n*2 and type(prefetch.get('generated')) is int and prefetch['generated'] >= 2
            and (prefetch.get('allocated_gib', 0) >= 56 or prefetch.get('tail_draining') is True)
            and 0 < changed.get('peak_allocated_gib', 0) < 70 and 0 < changed.get('peak_reserved_gib', 0) < 70,
            'Useful GPU prefetch acceptance missing or exceeded original70GiB ceiling')
    return dict(status='PASS_BOUND_EXACT_CACHE_CONTINUATION_STEP400', source_optimizer_updates=n,
                continuation_optimizer_updates=400-n, total_optimizer_updates=400,
                gpu_uuid=row['expected_gpu_uuid'], execution_variant=lineage['execution_variant'])


def admitted_row(manifest_binding, candidate_id):
    manifest = read(checked(manifest_binding))
    require(manifest.get('schema') == QUEUE_SCHEMA, 'Wrong independent continuation admission manifest')
    rows = [r for r in manifest['candidates'] if r['candidate_id'] == candidate_id]
    require(len(rows) == 1, 'Candidate missing or duplicated in frozen continuation admission')
    row = rows[0]
    validate_identity(row['expected_continuation_identity'], row)
    return row
