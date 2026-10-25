"""Export-only consumer revision; the trained source/Seal B identity stays immutable."""
from pathlib import Path
from ..hashing import sha256_file
from ..runtime.source_snapshot import verify_source_snapshot

# Only export reconstruction, its explicit execution gate and script dispatch may
# differ. All training, objectives, schedules, serializers, LoRA and acquisition
# implementation bytes remain identical to the source that produced the run.
ALLOWED = frozenset('src/grounded_norm_sft/interventions/'+name for name in
    ('export_pipeline.py','export_lineage.py','gates.py','launch.py'))


# Exact reviewed FP32 adapter-slot promotion repair; exporting.py is not freely mutable.
EXPORTER_PATH = 'src/grounded_norm_sft/interventions/exporting.py'
EXPORTER_DTYPE_REPAIR = (
    '5df8557a3bd50cd9f982534f42e2327ccdf4c5a3cfa917baf87d84f83c986d55',
    'e85142e2d57e97adf770eeb5b36763c16ab59f58ebc1796228db645546f2a6a6',
)


def execution_root(launch):
    return launch.get('export_consumer', {}).get('code_root', launch['paths']['code_root'])


def verify_export_consumer(launch):
    spec = launch.get('export_consumer')
    if spec is None:
        return
    if launch['phase'] != 'hybrid_export' or set(spec) != {
            'code_root','source_snapshot_sha256','source_manifest_sha256'}:
        raise ValueError('explicit export-only consumer source identity required')
    old, new = Path(launch['paths']['code_root']), Path(spec['code_root'])
    for root, identity in ((old,launch['authorization']),(new,spec)):
        verify_source_snapshot(str(root),identity['source_snapshot_sha256'])
        if sha256_file(root/'SHA256SUMS') != identity['source_manifest_sha256']:
            raise ValueError('export consumer source manifest bytes mismatch')
    required=set()
    for root in (old,new):
        required.update(str(p.relative_to(root)) for pattern in
            ('src/**/*.py','scripts/*.py','configs/runtime_pins.json','requirements*',
             'provenance/native_source/**/*','provenance/hybrid_norm_sft_plan_FINAL_v1.2.md')
            for p in root.glob(pattern) if p.is_file())
    if not required:
        raise ValueError('export consumer scientific source inventory empty')
    for relative in sorted(required-ALLOWED):
        a,b=old/relative,new/relative
        if a.is_file() and b.is_file():
            pair = (sha256_file(a), sha256_file(b))
            if pair[0] == pair[1] or (relative == EXPORTER_PATH and pair == EXPORTER_DTYPE_REPAIR):
                continue
        raise ValueError('export consumer scientific dependency drift: '+relative)
