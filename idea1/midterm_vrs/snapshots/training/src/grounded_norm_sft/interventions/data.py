"""Bounded serialization of pinned manifests; no model loading or launch policy."""
from dataclasses import asdict
import json
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
import torch
from .cache import digest, provenance, read_record
from ..hashing import sha256_file, sha256_bytes
from ..runtime import data_io


def fingerprint(item):
    """Hash exact tensor bytes, dtype and shape without a pixel-sized JSON tree."""
    fields = dict(vars(item))
    inputs = {}
    for key, value in item.model_inputs.items():
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        inputs[key] = dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                           sha256=sha256_bytes(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()))
    fields['model_inputs'] = inputs
    return digest(fields)


def serializer_policy(serializer):
    """Snapshot active native preprocessing; test adapters explicitly declare policy.

    No tensor data or vocabulary-sized backend JSON is needed on each access.
    Artifact pins bind vocabulary files; this binds mutable runtime configuration.
    """
    from ..serializer import Serializer
    if type(serializer) is not Serializer:
        policy = serializer.serialization_policy()
        if not isinstance(policy, dict) or not policy:
            raise ValueError('adapter must declare a nonempty serialization policy')
        return deepcopy(policy)
    processor, tokenizer = serializer.processor, serializer.tokenizer
    policy = dict(serializer={k: getattr(serializer, k) for k in
        ('max_tokens', 'visual_range', 'image_token', 'merge_size', 'image_token_id',
         'vision_start_id', 'vision_end_id')}, markers=asdict(serializer.markers),
        image_processor=processor.image_processor.to_dict(),
        chat_template=getattr(processor, 'chat_template', None),
        tokenizer=dict(init_kwargs={k: v for k, v in tokenizer.init_kwargs.items()
                                   if k not in ('name_or_path', '_name_or_path')},
            special_tokens_map=tokenizer.special_tokens_map,
            **{k: getattr(tokenizer, k, None) for k in
               ('padding_side', 'truncation_side', 'model_max_length',
                'clean_up_tokenization_spaces', 'chat_template')}))
    # HF config dictionaries can contain AddedToken values; their string content
    # is deterministic and the artifact pin separately binds full token metadata.
    return json.loads(json.dumps(policy, default=str, sort_keys=True))


def _stamp(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class ManifestInputs(Mapping):
    """Immutable source view with at most ``capacity`` serialized examples resident.

    Manifest rows are small host metadata, never processed pixels. Serializer pins
    are supplied audited artifact hashes. A per-key digest catches serializer drift
    on eviction/reload; returned mutable Serialized objects are checked on use.
    """
    def __init__(self, path, expected_sha256, serializer, *, task, split, pins,
                 image_dir=None, capacity=2):
        if task not in ('normlens', 'normbank') or split not in ('train', 'dev', 'test'):
            raise ValueError('explicit task and manifest split required')
        if type(capacity) is not int or capacity < 1:
            raise ValueError('positive finite cache capacity required')
        if not pins or any(not isinstance(v, str) or len(v) != 64 or
                           any(c not in '0123456789abcdef' for c in v) for v in pins.values()):
            raise ValueError('immutable serializer artifact SHA256 pins required')
        self._path, self._sha = Path(path), expected_sha256
        rows = getattr(data_io, 'load_' + task)(path, expected_sha256)
        selected = data_io.split_rows(rows, split)
        view = getattr(data_io, task + '_input')
        self._rows = {r['example_id']: MappingProxyType(view(r)) for r in selected}
        self.targets = MappingProxyType({r['example_id']: tuple(getattr(data_io, task + '_target')(r)) for r in selected})
        self._serializer, self.task, self.split = serializer, task, split
        self._policy_sha256 = digest(serializer_policy(serializer))
        self._capacity, self._cache, self._hashes = capacity, OrderedDict(), {}
        self._manifest_stamp = _stamp(self._path)
        self._images = {}
        if task == 'normlens':
            if image_dir is None:
                raise ValueError('NormLens requires image directory')
            from ..data.images import resolve_image_path
            self._loader = data_io.ImageLoader(image_dir)
            for row in selected:
                image = resolve_image_path(image_dir, row['image'])
                raw = sha256_file(image)
                # Actual schema lists all deduplicated source image hashes.
                allowed = row.get('image_file_sha256s')
                if allowed is not None and raw not in allowed:
                    raise ValueError('raw image hash differs from manifest')
                self._images[row['example_id']] = (image, raw)
        self._identity = dict(version='manifest_inputs_v1', manifest_sha256=expected_sha256,
                              task=task, split=split, pins=dict(pins), serializer_policy_sha256=self._policy_sha256,
                              ids_sha256=digest(list(self)),
                              raw_images_sha256=digest({k: v[1] for k, v in self._images.items()}))

    @property
    def identity(self):
        return deepcopy(self._identity)

    @property
    def image_sha256(self):
        return MappingProxyType({k: v[1] for k, v in self._images.items()})

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __contains__(self, key):
        return key in self._rows

    def _verify_source(self, key):
        if digest(serializer_policy(self._serializer)) != self._policy_sha256:
            raise ValueError('bound serializer policy changed')
        if _stamp(self._path) != self._manifest_stamp:
            if sha256_file(self._path) != self._sha:
                raise ValueError('pinned manifest changed')
            self._manifest_stamp = _stamp(self._path)
        if key in self._images:
            path, want = self._images[key]
            if sha256_file(path) != want:
                raise ValueError('pinned raw image changed')

    def verify_item(self, item):
        self._verify_source(item.example_id)
        if fingerprint(item) != self._hashes.get(item.example_id):
            raise ValueError('bound serialized input changed after cache validation')

    def __getitem__(self, key):
        row = self._rows[key]
        self._verify_source(key)
        if key in self._cache:
            item = self._cache.pop(key)
            self.verify_item(item)
        else:
            if self.task == 'normlens':
                image = self._loader.load(row)
                try:
                    item = self._serializer.normlens(key, row['action'], image, text_only=False)
                finally:
                    image.close()
            else:
                item = self._serializer.normbank(key, row['setting'], row['behavior'], row['constraints'])
            if item.example_id != key or item.task != self.task:
                raise ValueError('serializer example/task mismatch')
            value = fingerprint(item)
            if key in self._hashes and value != self._hashes[key]:
                raise ValueError('serializer changed on reload')
            self._hashes[key] = value
        self._cache[key] = item
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return item


class SupportInputs(Mapping):
    """Lazy provenance validation with cache byte identities bound before training."""
    def __init__(self, inputs, paths, *, pins, image_sha256, alignment_ids):
        self.inputs = inputs
        self._paths = {k: Path(paths[k]) for k in inputs}
        self._hashes = {k: sha256_file(v) for k, v in self._paths.items()}
        self._pins, self._images, self._alignment = dict(pins), dict(image_sha256), dict(alignment_ids)

    @property
    def identity(self):
        return digest(self._hashes)

    def __len__(self):
        return len(self._paths)

    def __iter__(self):
        return iter(self._paths)

    def __getitem__(self, key):
        path = self._paths[key]
        if sha256_file(path) != self._hashes[key]:
            raise ValueError('bound support cache changed')
        item = self.inputs[key]
        expected = provenance(item, **dict(self._pins, image_sha256=self._images[key]), **self._alignment)
        return read_record(path, expected_provenance=expected)
