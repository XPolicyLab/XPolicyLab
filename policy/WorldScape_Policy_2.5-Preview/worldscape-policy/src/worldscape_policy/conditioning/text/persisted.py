"""Portable 5B T5 features generated from the actual native initialization weights."""
import hashlib
import json
import os
from pathlib import Path
import uuid

import torch
from torch import nn
from safetensors import safe_open

from worldscape_policy.checkpoint.weights_io import checkpoint_weight_files
from worldscape_policy.conditioning.text.t5 import T5InstructionEncoder

PREFIX = "condition_router.interactive.t5."
SCHEMA = "robodojo-wan22-text-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def tokenizer_identity(directory):
    root = Path(directory)
    # Do not hash HF model-weight shards bundled alongside tokenizer assets.
    files = [p for p in root.iterdir() if p.is_file() and
             (p.suffix in {'.json', '.model', '.txt'} and 'index' not in p.name)
             and p.name != 'generation_config.json']
    if not any(p.name in {'tokenizer.json','spiece.model','tokenizer.model'} for p in files):
        raise ValueError('Missing tokenizer assets')
    return {p.name: sha(p) for p in sorted(files)}


def checkpoint_identity(directory):
    root = Path(directory).resolve()
    files = checkpoint_weight_files(root)
    # Fast startup check: no reading the 10GB T5 payload on every training rank.
    return {'path': str(root), 'files': [
        {'name': p.name, 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
         **({'sha256': sha(p)} if p.suffix == '.json' else {})} for p in files]}


def filtered_checkpoint(directory, *, include=None, exclude=None):
    """Validate all safetensor keys, materialize only selected tensors."""
    files = checkpoint_weight_files(directory)
    indexed = files[0].suffix == '.json'
    expected = json.loads(files[0].read_text())['weight_map'] if indexed else None
    seen, result = {}, {}
    for file in files[1:] if indexed else files:
        with safe_open(str(file), framework='pt', device='cpu') as stream:
            for key in stream.keys():
                if key in seen:
                    raise ValueError('Duplicate checkpoint tensor: '+key)
                seen[key] = file.name
                if (include is None or key.startswith(include)) and (exclude is None or not key.startswith(exclude)):
                    result[key] = stream.get_tensor(key)
    if expected is not None and seen != expected:
        raise ValueError('Checkpoint index does not match shard keys')
    return result


def load_live_encoder(directory, tokenizer, *, max_length=512, device='cuda', **kwargs):
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        encoder = T5InstructionEncoder(tokenizer_path=str(tokenizer), max_length=max_length,
                                       force_eval=True, **kwargs)
    state = {k[len(PREFIX):]: v for k,v in filtered_checkpoint(directory, include=PREFIX).items()}
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode()); digest.update(tensor_digest(tensor).encode())
    encoder.load_state_dict(state, strict=True, assign=True)
    encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    return encoder, digest.hexdigest()


def validate_payload(metadata, features, *, capacity, max_length, dim):
    prompts = metadata.get('prompts', [])
    prompt_mode = metadata.get('prompt_mode', 'template')
    if (metadata.get('schema') != SCHEMA or metadata.get('max_length') != max_length
            or metadata.get('capacity') != capacity or metadata.get('dim') != dim
            or not isinstance(prompts, list) or not 0 < len(prompts) <= capacity
            or any(not isinstance(p, str) for p in prompts) or len(set(prompts)) != len(prompts)
            or metadata.get('padding') != 'zero' or prompt_mode not in {'raw', 'template'}
            or metadata.get('lowercase_instruction') is not (prompt_mode == 'template')
            or (prompt_mode == 'raw' and metadata.get('template') is not None)):
        raise ValueError('5B text cache prompt/shape contract mismatch')
    if features.shape != (capacity,max_length,dim) or features.dtype != torch.bfloat16:
        raise ValueError('5B text cache feature shape/dtype mismatch')
    if not bool(torch.isfinite(features).all()) or tensor_digest(features) != metadata.get('features_sha256'):
        raise ValueError('5B text cache corrupt/nonfinite features')
    if (not isinstance(metadata.get('source'), dict) or not isinstance(metadata.get('tokenizer'), dict)
            or len(metadata.get('t5_weights_sha256','')) != 64):
        raise ValueError('5B text cache missing source provenance')


def save_features(path, metadata, features):
    validate_payload(metadata, features, capacity=features.shape[0], max_length=features.shape[1], dim=features.shape[2])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        torch.save({'metadata':metadata,'features':features.cpu()}, tmp)
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_cache_extension(old_metadata, old_features, new_metadata, new_features):
    """A dataset expansion may add prompts, but cannot change existing features."""
    validate_payload(old_metadata, old_features, capacity=old_metadata['capacity'],
                     max_length=old_metadata['max_length'], dim=old_metadata['dim'])
    for key in ('source', 'tokenizer', 't5_weights_sha256', 'max_length', 'dim',
                'template', 'lowercase_instruction', 'padding'):
        if old_metadata.get(key) != new_metadata.get(key):
            raise ValueError(f'Text cache extension changes {key}')
    if old_metadata.get('prompt_mode', 'template') != new_metadata.get('prompt_mode', 'template'):
        raise ValueError('Text cache extension changes prompt_mode')
    positions = {p: i for i, p in enumerate(new_metadata['prompts'])}
    if any(p not in positions for p in old_metadata['prompts']):
        raise ValueError('Text cache extension removes existing prompts')
    indices = torch.tensor([positions[p] for p in old_metadata['prompts']], device=new_features.device)
    selected = new_features.index_select(0, indices).cpu()
    if not torch.equal(old_features[:len(indices)].cpu(), selected):
        raise ValueError('Text cache extension changes existing prompt features')


class PrecomputedT5InstructionEncoder(nn.Module):
    """No parameters or T5 allocation in training; portable feature buffers.

    Inputs are exact trainer strings (raw in unscheduled interactive mode). Unseen
    training text is an error. Eval can use the original live encoder lazily.
    """
    def __init__(self, *, text_feature_cache, tokenizer_path, source_checkpoint=None,
                 cache_capacity=64, max_length=512, dim=4096, load_features=True,
                 allow_live_fallback=False, text_encoder_pretrained_path=None,
                 force_eval=True, refresh_cache_on_load=False,
                 cache_source_checkpoint=None, **encoder_kwargs):
        super().__init__()
        if not force_eval:
            raise ValueError('Offline T5 requires force_eval=true')
        self.capacity,self.max_length,self.dim=cache_capacity,max_length,dim
        self.tokenizer_path=tokenizer_path
        self.allow_live_fallback=allow_live_fallback
        self.encoder_kwargs=encoder_kwargs
        self.refresh_cache_on_load=refresh_cache_on_load
        self._metadata=None;self._live=None
        self.register_buffer('features',torch.zeros(cache_capacity,max_length,dim,dtype=torch.bfloat16))
        self.register_buffer('metadata_bytes',torch.zeros(262144,dtype=torch.uint8))
        if load_features or refresh_cache_on_load:
            payload=torch.load(text_feature_cache,map_location='cpu',weights_only=True)
            metadata,features=payload['metadata'],payload['features']
            self._validate(metadata,features)
            cache_source = cache_source_checkpoint if refresh_cache_on_load else source_checkpoint
            if cache_source is None or metadata['source']!=checkpoint_identity(cache_source):
                raise ValueError('5B cache source differs from initialization checkpoint; regenerate cache')
            if metadata['tokenizer']!=tokenizer_identity(tokenizer_path):
                raise ValueError('5B cache tokenizer differs; regenerate cache')
            self.features.copy_(features)
            encoded=json.dumps(metadata,sort_keys=True).encode()
            if len(encoded)>self.metadata_bytes.numel():
                raise ValueError('Text cache metadata exceeds capacity')
            self.metadata_bytes[:len(encoded)]=torch.tensor(list(encoded),dtype=torch.uint8)
            self._metadata=metadata
        self.register_load_state_dict_post_hook(self._validate_restored)

    def _validate(self, metadata, features):
        validate_payload(metadata,features,capacity=self.capacity,max_length=self.max_length,dim=self.dim)

    def _load_from_state_dict(self,state_dict,prefix,*args,**kwargs):
        if self.refresh_cache_on_load:
            old_metadata = json.loads(bytes(state_dict[prefix+'metadata_bytes'].cpu().tolist()).rstrip(b'\0'))
            validate_cache_extension(old_metadata, state_dict[prefix+'features'],
                                     self._metadata, self.features)
            # Only these non-parameter buffers change. Model parameters and
            # DeepSpeed optimizer partitions retain strict checkpoint loading.
            state_dict[prefix+'features'] = self.features
            state_dict[prefix+'metadata_bytes'] = self.metadata_bytes
        self._metadata=None;self._live=None
        return super()._load_from_state_dict(state_dict,prefix,*args,**kwargs)

    def _validate_restored(self,module,incompatible_keys):
        metadata=json.loads(bytes(self.metadata_bytes.cpu().tolist()).rstrip(b'\0'))
        self._validate(metadata,self.features)
        self._metadata=metadata

    def encode_text(self,instructions):
        if self._metadata is None:
            raise RuntimeError('Text features not loaded')
        table={p:i for i,p in enumerate(self._metadata['prompts'])}
        if any(p not in table for p in instructions):
            if self.training or not self.allow_live_fallback:
                missing = list(dict.fromkeys(p for p in instructions if p not in table))
                raise ValueError(
                    'Unknown instruction in embedded T5 feature cache; live text encoding is disabled. '
                    f'Cache mode={self._metadata.get("prompt_mode", "template")}; '
                    f'missing examples={missing[:3]!r}. Use a checkpoint containing the exact task instruction.')
            return self._encode_live(instructions)
        ids=torch.tensor([table[p] for p in instructions],device=self.features.device,dtype=torch.long)
        return self.features.index_select(0,ids)

    @torch.no_grad()
    def _encode_live(self,instructions):
        if self._live is None:
            source=self._metadata['source']['path']
            if checkpoint_identity(source)!=self._metadata['source'] or tokenizer_identity(self.tokenizer_path)!=self._metadata['tokenizer']:
                raise ValueError('Evaluation T5 assets differ from offline features')
            encoder,digest=load_live_encoder(source,self.tokenizer_path,max_length=self.max_length,
                                             dim=self.dim,device=self.features.device,**self.encoder_kwargs)
            if digest!=self._metadata['t5_weights_sha256']:
                raise ValueError('Evaluation T5 tensors differ from offline features')
            object.__setattr__(self,'_live',encoder)
        self._live.to(device=self.features.device)
        with torch.autocast(self.features.device.type,dtype=torch.bfloat16):
            return self._live.encode_text(instructions)
