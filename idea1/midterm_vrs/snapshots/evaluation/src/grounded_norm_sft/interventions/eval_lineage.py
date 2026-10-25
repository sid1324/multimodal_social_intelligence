"""Held-out native evaluation consumes a verified export, preserving trained identity."""
from copy import deepcopy
import ast
from pathlib import Path
from ..hashing import sha256_file
from ..runtime.source_snapshot import verify_source_snapshot
from .export_lineage import verify_export_consumer

ALLOWED = frozenset('src/grounded_norm_sft/interventions/'+name for name in
    ('evaluation.py','eval_lineage.py','pipeline.py','__main__.py','gates.py','launch.py'))
# Accounting-only successor repair; scientific implementations remain byte-exact.
AGGREGATE_PATH = 'src/grounded_norm_sft/interventions/aggregate.py'
AGGREGATE_SUCCESSOR_REPAIR = (
    '2ee54b5ebccc686b80890df6eccf2125ec064d216398dcfc5917d5af693791e9',
    '1b927141f276b568e2390fb32ddfe6570af54926e5e9535f9d2a3f2d858facd8',
)
SOURCE_KEYS = {'code_root','source_snapshot_sha256','source_manifest_sha256'}


def execution_root(launch):
    return launch.get('eval_consumer', {}).get('code_root', launch['paths']['code_root'])


def verify_eval_consumer(launch):
    if launch['phase'] != 'hybrid_eval' or set(launch.get('eval_consumer', {})) != SOURCE_KEYS:
        raise ValueError('explicit evaluation-only consumer source identity required')
    exporter=launch['eval'].get('export_consumer', {})
    if set(exporter) != SOURCE_KEYS or 'export_consumer' in launch:
        raise ValueError('distinct actual export and evaluation consumer bindings required')
    # Authenticate the exporter against the original trained source, without
    # rewriting old Seal B or authorizing changes to its scientific dependencies.
    shadow=deepcopy(launch);shadow['phase']='hybrid_export';shadow['export_consumer']=exporter
    verify_export_consumer(shadow)
    old,new=Path(exporter['code_root']),Path(launch['eval_consumer']['code_root'])
    consumer=launch['eval_consumer']
    verify_source_snapshot(str(new),consumer['source_snapshot_sha256'])
    if sha256_file(new/'SHA256SUMS')!=consumer['source_manifest_sha256']:
        raise ValueError('evaluation consumer source manifest bytes mismatch')
    required=set()
    for root in (old,new):
        required.update(str(p.relative_to(root)) for pattern in
            ('src/**/*.py','scripts/*.py','configs/runtime_pins.json','requirements*',
             'provenance/native_source/**/*','provenance/hybrid_norm_sft_plan_FINAL_v1.2.md')
            for p in root.glob(pattern) if p.is_file())
    if not required:
        raise ValueError('evaluation consumer scientific source inventory empty')
    for relative in sorted(required-ALLOWED):
        a,b=old/relative,new/relative
        if a.is_file() and b.is_file():
            pair=(sha256_file(a),sha256_file(b))
            if pair[0]==pair[1] or (relative==AGGREGATE_PATH and pair==AGGREGATE_SUCCESSOR_REPAIR):
                continue
        raise ValueError('evaluation consumer scientific dependency drift: '+relative)
    # The dispatcher lives beside cache acquisition. A filename-level exception
    # must not silently authorize changes to Budget, cache_inputs or other code.
    relative='src/grounded_norm_sft/interventions/pipeline.py'
    def acquisition(root):
        tree=ast.parse((root/relative).read_text())
        tree.body=[node for node in tree.body if not (
            isinstance(node,ast.FunctionDef) and node.name=='execute')]
        return ast.dump(tree)
    if acquisition(old)!=acquisition(new):
        raise ValueError('evaluation consumer acquisition pipeline contract drift')
