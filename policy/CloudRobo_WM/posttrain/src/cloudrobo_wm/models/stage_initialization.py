from collections import Counter
import logging
import torch
logger = logging.getLogger(__name__)

def load_stage_payload(model, payload, checkpoint):
    model._validate_checkpoint_payload(payload, strict_shape=False)
    model._ms_rebuild_after_load = {}
    if 'mot' not in payload:
        raise ValueError('Policy stage initialization requires a mot checkpoint')
    report = dict(version=2, checkpoint=str(checkpoint), strict=False, source_step=payload.get('step'), loaded=[], skipped=[], missing=[], unexpected=[], partial=[], loaded_details=[], fresh_optimizer_scheduler_step=True, source_stats_hash=(payload.get('multisource_contract') or {}).get('stats_hash'), destination_stats_hash=(model.multisource_contract or {}).get('stats_hash'))
    keys = tuple(dict.fromkeys(('mot', 'proprio_encoder', *model._dino_ckpt_keys(), *model._mode_ckpt_keys())))
    for key in keys:
        module = getattr(model, key, None)
        saved = payload.get(key, {})
        if not isinstance(saved, dict):
            raise ValueError('Invalid checkpoint state dictionary: ' + key)
        if module is None:
            report['unexpected'].extend((key + '.' + name for name in saved))
            continue
        current = module.state_dict()
        selected = {}
        for name, target in current.items():
            full = key + '.' + name
            source = saved.get(name)
            info = dict(key=full, target_shape=list(target.shape), source_shape=list(source.shape) if torch.is_tensor(source) else None)
            if source is None:
                report['missing'].append(full)
                report['skipped'].append(dict(info, reason='missing_in_checkpoint'))
                continue
            if not torch.is_tensor(source) or source.shape != target.shape:
                report['skipped'].append(dict(info, reason='shape_mismatch' if torch.is_tensor(source) else 'not_tensor'))
                continue
            if key == 'multisource_conditioning' and name == 'conditions.weight':
                old = (payload.get('multisource_contract') or {}).get('conditions', {})
                new = (model.multisource_contract or {}).get('conditions', {})
                fresh = target.detach().clone()
                matched = []
                for new_id, condition in new.items():
                    candidates = [int(i) for i, item in old.items() if item == condition]
                    if len(candidates) > 1:
                        raise ValueError('Ambiguous semantic condition mapping')
                    if candidates:
                        old_id = candidates[0]
                        if not (0 <= old_id < source.shape[0] and 0 <= int(new_id) < target.shape[0]):
                            raise ValueError('Condition index exceeds embedding capacity')
                        fresh[int(new_id)].copy_(source[old_id].to(fresh))
                        matched.append([old_id, int(new_id)])
                report['matched_condition_rows'] = matched
                report['reinitialized_condition_rows'] = [int(i) for i in new if int(i) not in [j for _, j in matched]]
                report['condition_mapping'] = dict(source=old, destination=new, unused_rows='keep_destination_initialization')
                if matched:
                    selected[name] = fresh
                    report['partial'].append(dict(info, reason='semantic_condition_rows', matched_rows=matched))
                else:
                    report['skipped'].append(dict(info, reason='no_matching_condition_semantics'))
                continue
            selected[name] = source
            report['loaded'].append(full)
            report['loaded_details'].append(dict(info, numel=target.numel(), source_dtype=str(source.dtype), target_dtype=str(target.dtype)))
        report['unexpected'].extend((key + '.' + name for name in saved if name not in current))
        module.load_state_dict(selected, strict=False)
    known = set(keys)
    for key, value in payload.items():
        if key not in known and isinstance(value, dict):
            report['unexpected'].extend((key + '.' + name for name, item in value.items() if torch.is_tensor(item)))
    report['reinitialized'] = [item['key'] for item in report['skipped']]
    report['summary'] = dict(loaded_tensors=len(report['loaded']), loaded_numel=sum((item['numel'] for item in report['loaded_details'])), partial_tensors=len(report['partial']), skipped_tensors=len(report['skipped']), skipped_by_reason=dict(Counter((item['reason'] for item in report['skipped']))), missing_tensors=len(report['missing']), unexpected_tensors=len(report['unexpected']))
    model.multisource_initialization_report = report
    logger.info('Policy stage initialization: %s', report['summary'])
    return payload

def load_stage_checkpoint(model, path):
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    return load_stage_payload(model, payload, path)
