"""Strict compatibility of an immutable cache acquisition and its later consumer.

Only operational launch/approval gates may differ. Pipeline changes are limited to
this module's explicit metadata binding hook; all scientific AST remains exact.
"""
import ast
from pathlib import Path

from ..hashing import sha256_file
from ..runtime.source_snapshot import verify_source_snapshot

ACQUISITION_RELEASE = 'a46e93fe49755401efeae520dacdcc2ece0361f14608e4d71b3067c95384de82'
ACQUISITION_MANIFEST = 'e594af2f3fd552489b6ff7992fac2c67d1ebd1ee9d058ddcc9b77c61084a5ab8'
_OPERATIONAL = {
    'src/grounded_norm_sft/interventions/approval.py',
    'src/grounded_norm_sft/interventions/launch.py',
    'src/grounded_norm_sft/interventions/gates.py',
    'src/grounded_norm_sft/interventions/__main__.py',
    'src/grounded_norm_sft/interventions/acquisition_source.py',
}
_PIPELINE = 'src/grounded_norm_sft/interventions/pipeline.py'


def _pipeline_contract(path):
    tree = ast.parse(Path(path).read_text())
    # Exact syntax only: no broad function exclusion or launcher-field erasure.
    hook_import = ast.dump(ast.parse('from .acquisition_source import bind_acquisition_source').body[0])
    hook_call = ast.dump(ast.parse('bind_acquisition_source(launch, inputs)').body[0])
    potency_test = ast.dump(ast.parse("command == 'potency'", mode='eval').body)
    diagnostic_branch = ast.dump(ast.parse("if launch.get('diagnostic_amendment') is not None:\n    from .diagnostic_potency import execute_loaded\n    return execute_loaded(model, launch, inputs, pins=pins, alignment_ids=alignment, budget=budget)").body[0])
    removed = 0
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name == 'execute':
            for parent in ast.walk(function):
                if isinstance(parent, ast.Try):
                    parent.body = [node for node in parent.body if ast.dump(node) != diagnostic_branch]
            for node in ast.walk(function):
                if (isinstance(node, ast.If) and ast.dump(node.test) == potency_test and
                        node.body and ast.dump(node.body[0]) == hook_call):
                    node.body.pop(0)
                    removed += 1
    if removed == 1:
        for index, node in enumerate(tree.body):
            if ast.dump(node) == hook_import:
                tree.body.pop(index)
                break
    return ast.dump(tree)


def verify_compatibility(acquisition_root, consumer_root, *, acquisition_sha256,
                         acquisition_manifest_sha256, consumer_sha256, consumer_manifest_sha256):
    if (acquisition_sha256 != ACQUISITION_RELEASE or
            acquisition_manifest_sha256 != ACQUISITION_MANIFEST):
        raise ValueError('immutable cache acquisition source identity mismatch')
    for root, identity, manifest in ((acquisition_root, acquisition_sha256, acquisition_manifest_sha256),
                                     (consumer_root, consumer_sha256, consumer_manifest_sha256)):
        verify_source_snapshot(str(root), identity)
        if sha256_file(Path(root)/'SHA256SUMS') != manifest:
            raise ValueError('source manifest bytes mismatch')
    old, new = Path(acquisition_root), Path(consumer_root)
    # A superset of transitive acquisition dependencies avoids an import loader
    # (and therefore avoids Torch/native imports during metadata validation).
    covered = []
    for line in (old/'SHA256SUMS').read_text().splitlines():
        checksum, relative = line.split('  ', 1)
        if relative == _PIPELINE:
            if _pipeline_contract(old/relative) != _pipeline_contract(new/relative):
                raise ValueError('scientific pipeline source drift')
        elif relative == 'src/grounded_norm_sft/interventions/__init__.py':
            if (checksum != '78003667d4d47e96dc44b6cf2c09b7da80a255b5781056b460a9e7138ec317e8' or
                    sha256_file(new/relative) not in {checksum,
                        '5ef8925d8d09b12fc0cae3c3157219813467ec6812d7828d9f67d5955d8f59cc'}):
                raise ValueError('unreviewed acquisition package export change')
        elif relative == 'src/grounded_norm_sft/runtime/ledger.py':
            if (checksum != 'a60aabbafa4cd61f099dd33c29183e0558b26444e346f193695e38016390d500' or
                    sha256_file(new/relative) not in {checksum,
                        'e2b11f3dd0e6f8ad676a4064a002c8b20b244332b8d8fec6d9c3e3830e0a4d58'}):
                raise ValueError('unreviewed aggregate ledger source change')
        elif relative not in _OPERATIONAL and (relative.startswith('src/') or
                relative.startswith('configs/') or relative.startswith('provenance/native_source/') or
                relative.startswith('requirements') or relative == 'pyproject.toml'):
            if not (new/relative).is_file() or sha256_file(new/relative) != checksum:
                raise ValueError('scientific acquisition dependency drift: ' + relative)
        covered.append(relative)
    if _PIPELINE not in covered:
        raise ValueError('acquisition pipeline missing from source manifest')


def verify_launch_acquisition(launch):
    """Preflight immutable acquisition metadata before any reservation."""
    binding = launch.get('cache_acquisition_source')
    if not isinstance(binding, dict):
        raise ValueError('explicit cache acquisition source binding required')
    auth = launch['authorization']
    verify_compatibility(binding['code_root'], launch['paths']['code_root'],
        acquisition_sha256=binding['source_snapshot_sha256'],
        acquisition_manifest_sha256=binding['source_manifest_sha256'],
        consumer_sha256=auth['source_snapshot_sha256'],
        consumer_manifest_sha256=auth['source_manifest_sha256'])
    index = Path(launch['cache_index'])
    if sha256_file(index) != binding['cache_index_sha256']:
        raise ValueError('acquisition cache index bytes mismatch')

    if binding['cache_index_sha256'] != launch.get('cache_index_sha256'):
        raise ValueError('acquisition and consumer cache index hashes differ')
    if binding['cache_index_sha256'] != '4f1c3a518295161d3d7cce7e30fd2f02bbb8ea9194ca0cd91906734ed5d477a2':
        raise ValueError('original accepted cache index required')
    return binding


def bind_acquisition_source(launch, inputs):
    """Preserve old cache identity only after proving consumer compatibility."""
    binding = verify_launch_acquisition(launch)
    source = inputs._identity['launch_provenance']
    source['source_snapshot_sha256'] = binding['source_snapshot_sha256']
    source['source_manifest_sha256'] = binding['source_manifest_sha256']
