"""Per-cache acquisition provenance with exact scientific dependency compatibility."""
import ast
from copy import deepcopy
from pathlib import Path
from ..hashing import sha256_file
from ..runtime.source_snapshot import verify_source_snapshot


_EXPORT_INITIALIZER = 'src/grounded_norm_sft/interventions/__init__.py'
_EAGER_EXPORT_SHA256 = '78003667d4d47e96dc44b6cf2c09b7da80a255b5781056b460a9e7138ec317e8'
_LAZY_EXPORT_SHA256 = '5ef8925d8d09b12fc0cae3c3157219813467ec6812d7828d9f67d5955d8f59cc'


def _approved_initializer_transition(relative, producer, consumer):
    # Previously reviewed metadata-only lazy export transition (also pinned in
    # acquisition_source.py). Reference acquisition imports reference/cache/data
    # directly; none accesses the three endpoint exports. The eager initializer
    # imported endpoints only to bind those names, whereas the lazy initializer
    # binds identical names on access. No selector/serializer/native code changes
    # are exempted. Every other initializer byte transition remains an error.
    return (relative == _EXPORT_INITIALIZER and
            sha256_file(producer) == _EAGER_EXPORT_SHA256 and
            sha256_file(consumer) == _LAZY_EXPORT_SHA256)


def _closure(root):
    """Conservative local import closure; never import/execute audited source."""
    roots = ['interventions.reference', 'interventions.data', 'interventions.cache',
             'serializer', 'runtime.model', 'runtime.pins', 'hashing', 'io_utils']
    pending = ['grounded_norm_sft.'+name for name in roots]
    covered = set()
    while pending:
        module = pending.pop()
        relative = 'src/'+module.replace('.', '/')+'.py'
        path = root/relative
        if not path.is_file():
            relative = 'src/'+module.replace('.', '/')+'/__init__.py'
            path = root/relative
        if relative in covered or not path.is_file():
            continue
        covered.add(relative)
        # Package initializers are checked as bytes, but their lazy APIs are not
        # called by acquisition and do not expand the scientific closure.
        if path.name == '__init__.py':
            continue
        package = module.split('.')[:-1]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(a.name for a in node.names if a.name.startswith('grounded_norm_sft.'))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[:len(package)-node.level+1]
                    target = '.'.join(base+([node.module] if node.module else []))
                else:
                    target = node.module or ''
                if target.startswith('grounded_norm_sft'):
                    pending.append(target)
                    pending.extend(target+'.'+a.name for a in node.names)
        # Every traversed package initializer is itself an exact byte dependency.
        for i in range(1,len(module.split('.'))):
            init = 'src/'+'/'.join(module.split('.')[:i])+'/__init__.py'
            if (root/init).is_file():
                covered.add(init)
    return covered


def verify_cache_compatibility(producer_root, consumer_root, binding, consumer):
    producer_root, consumer_root = Path(producer_root), Path(consumer_root)
    for root, identity in ((producer_root,binding),(consumer_root,consumer)):
        verify_source_snapshot(str(root),identity['source_snapshot_sha256'])
        if sha256_file(root/'SHA256SUMS') != identity['source_manifest_sha256']:
            raise ValueError('cache acquisition source manifest bytes mismatch')
    dependencies = _closure(producer_root) | _closure(consumer_root)
    # Runtime pins, requirements and preserved native-source receipts are part of
    # native acquisition even when referenced through data rather than imports.
    for root in (producer_root,consumer_root):
        dependencies.update(str(p.relative_to(root)) for pattern in
            ('configs/runtime_pins.json','requirements*','provenance/native_source/**/*')
            for p in root.glob(pattern) if p.is_file())
    for relative in sorted(dependencies):
        a,b=producer_root/relative,consumer_root/relative
        if (not a.is_file() or not b.is_file() or
            (sha256_file(a) != sha256_file(b) and not _approved_initializer_transition(relative,a,b))):
            raise ValueError('cache acquisition scientific dependency drift: '+relative)
    relative='src/grounded_norm_sft/interventions/pipeline.py'
    def nodes(root):
        tree=ast.parse((root/relative).read_text())
        return {n.name:ast.dump(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
                and n.name in ('cache_inputs','Budget','BudgetExpired')}
    if len(nodes(producer_root)) != 3 or nodes(producer_root) != nodes(consumer_root):
        raise ValueError('cache acquisition pipeline scientific contract drift')


def bind_cache_source(launch, inputs, kind):
    """Bind each actual cache to its explicit acquisition after compatibility proof."""
    if launch.get('training_amendment') is None:
        return
    binding=launch['cache_acquisition_sources'][kind]
    index_key=kind+'_cache_index'
    if (binding['cache_index_sha256'] != launch[index_key+'_sha256'] or
        sha256_file(launch[index_key]) != binding['cache_index_sha256']):
        raise ValueError('per-cache acquisition index bytes mismatch')
    verify_cache_compatibility(binding['code_root'],launch['paths']['code_root'],binding,launch['authorization'])
    # All non-code source, policy, serializer and manifest identities are still
    # compared exactly by validate_index. Only audited acquisition release fields
    # are rebound, independently for train and dev.
    provenance=deepcopy(inputs._identity['launch_provenance'])
    for key in ('source_snapshot_sha256','source_manifest_sha256'):
        provenance[key]=binding[key]
    inputs._identity['launch_provenance']=provenance


def verify_cache_sources(launch):
    """Validate both explicit producer releases during launch preflight."""
    for kind in ('train','dev'):
        binding=launch['cache_acquisition_sources'][kind]
        key=kind+'_cache_index'
        if (binding['cache_index_sha256'] != launch[key+'_sha256'] or
            sha256_file(launch[key]) != binding['cache_index_sha256']):
            raise ValueError('per-cache acquisition index bytes mismatch')
        verify_cache_compatibility(binding['code_root'],launch['paths']['code_root'],binding,launch['authorization'])
