"""Frozen sentence embeddings, event-specific hard negatives and retrieval."""
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.manifold import TSNE

from .data import save_json
from .statistics import bar_plot, cluster_mean_ci, figure_save, groups_for
import matplotlib.pyplot as plt

LOG = logging.getLogger(__name__)


def text_rows(sample, humans):
    rows = []
    for col in ['norm_text', 'reference_evidence', 'reference_verbal_evidence', 'reference_nonverbal_evidence']:
        for r in sample.to_dict('records'):
            if r[col].strip():
                rows.append({'key': json.dumps([r['sample_id'], col]), 'sample_id': r['sample_id'], 'representation': col, 'text': r[col]})
    for r in humans[humans.response_status.eq('completed')].to_dict('records'):
        if r['human_evidence'].strip():
            col = 'human_evidence_' + r['annotator']
            rows.append({'key': json.dumps([r['sample_id'], col]), 'sample_id': r['sample_id'], 'representation': col, 'text': r['human_evidence']})
    return pd.DataFrame(rows).sort_values('key').reset_index(drop=True)


def cache_key(rows, model, revision):
    blob = json.dumps({'rows': rows.to_dict('records'), 'model': model, 'revision': revision}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_cache(folder, rows, model, revision):
    manifest = json.loads((folder / 'manifest.json').read_text())
    expected = cache_key(rows, model, revision)
    saved = pd.read_csv(folder / 'rows.csv', dtype=str, keep_default_na=False)
    if manifest['fingerprint'] != expected or not saved.equals(rows):
        raise ValueError('Embedding cache is stale or row IDs/text/order changed.')
    vectors = np.load(folder / 'embeddings.npy', allow_pickle=False)
    if vectors.ndim != 2 or len(vectors) != len(rows) or not np.isfinite(vectors).all():
        raise ValueError('Embedding arrays do not align with the row manifest.')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('Expected unit-normalized sentence embeddings.')
    checksum = hashlib.sha256(vectors.tobytes()).hexdigest()
    if manifest['array_sha256'] != checksum:
        raise ValueError('Embedding data checksum mismatch.')
    return vectors


def embed(config, sample, humans):
    cfg = config['embedding']
    if not cfg.get('language_reviewed'):
        raise ValueError('Review actual text languages and model suitability, then set embedding.language_reviewed=true.')
    revision = cfg.get('revision')
    if not revision or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Pin embedding.revision to the exact 40-character Hugging Face commit SHA.')
    rows = text_rows(sample, humans)
    if rows.empty or rows.key.duplicated().any():
        raise ValueError('Need nonempty unique text embedding rows.')
    folder = Path(config['output_dir']) / 'embeddings' / cache_key(rows, cfg['model'], revision)[:16]
    if (folder / 'manifest.json').exists():
        return rows, load_cache(folder, rows, cfg['model'], revision)
    from sentence_transformers import SentenceTransformer
    import torch
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.set_num_threads(2)
    model = SentenceTransformer(cfg['model'], revision=revision, device=cfg['device'],
                                cache_folder='.cache/sentence_transformers', trust_remote_code=False)
    model.eval()
    vectors = model.encode(rows.text.tolist(), batch_size=cfg['batch_size'], normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
    lengths = [len(model.tokenizer.encode(t, truncation=False)) for t in rows.text]
    folder.mkdir(parents=True, exist_ok=True)
    rows.to_csv(folder / 'rows.csv', index=False)
    np.save(folder / 'embeddings.npy', vectors, allow_pickle=False)
    pd.DataFrame({'key': rows.key, 'token_length': lengths, 'truncated': np.array(lengths) > model.max_seq_length}).to_csv(folder / 'token_lengths.csv', index=False)
    save_json({'model': cfg['model'], 'revision': revision, 'fingerprint': cache_key(rows, cfg['model'], revision),
               'array_sha256': hashlib.sha256(vectors.tobytes()).hexdigest(),
               'seed': config['seed'], 'rows': len(rows), 'dimension': vectors.shape[1],
               'max_seq_length': model.max_seq_length, 'normalized': True,
               'packages': {p: importlib.metadata.version(p) for p in ['sentence-transformers', 'torch', 'transformers', 'numpy']}}, folder / 'manifest.json')
    LOG.info('Embedded %d actual text rows; %d exceed model token limit.', len(rows), sum(n > model.max_seq_length for n in lengths))
    return rows, load_cache(folder, rows, cfg['model'], revision)


def build_negatives(sample, norms):
    """Negatives chosen using norms/metadata only, never human evidence similarity."""
    pairs = []
    for i, anchor in sample.iterrows():
        pool = sample[(sample.sample_id != anchor.sample_id) &
                      sample.gold_norm_judgment.eq(anchor.gold_norm_judgment) &
                      sample.reference_evidence.ne('') &
                      sample.reference_evidence.ne(anchor.reference_evidence)]
        if anchor.source_video_id:
            pool = pool[pool.source_video_id.ne(anchor.source_video_id)]
        if anchor.language:
            pool = pool[pool.language.eq(anchor.language)]
        same_norm = pool[pool.norm_text.eq(anchor.norm_text)]
        same_category = pool[pool.category.eq(anchor.category)] if anchor.category else pool.iloc[:0]
        if len(same_norm):
            pool, strategy = same_norm, 'same_label_exact_norm_different_source'
        elif len(same_category):
            pool, strategy = same_category, 'same_label_category_nearest_norm'
        else:
            strategy = 'same_label_nearest_norm'
        if pool.empty or not anchor.reference_evidence:
            pairs.append({'sample_id': anchor.sample_id, 'negative_id': '', 'strategy': 'no_eligible_negative'})
            continue
        scores = norms[pool.index] @ norms[i]
        # Stable tie order is sample order; independent of human responses.
        j = pool.index[int(np.argmax(scores))]
        pairs.append({'sample_id': anchor.sample_id, 'negative_id': sample.loc[j, 'sample_id'],
                      'strategy': strategy, 'norm_similarity': float(scores.max())})
    return pd.DataFrame(pairs)


def retrieval_rank(query, references, true_index):
    """Conservative tied rank: every tied candidate counts ahead of true target."""
    similarities = references @ query
    return int(np.count_nonzero(similarities >= similarities[true_index] - 1e-7))


def summarize_pairs(pairs, config):
    results = {}
    for annotator, frame in pairs.groupby('annotator'):
        delta = frame.positive_similarity - frame.hard_negative_similarity
        result = {'difference': cluster_mean_ci(delta, groups_for(frame), config['seed'], config['bootstrap_replicates']),
                  'positive': {'mean': float(frame.positive_similarity.mean()), 'median': float(frame.positive_similarity.median()),
                               'std': float(frame.positive_similarity.std(ddof=1)) if len(frame) > 1 else None},
                  'negative': {'mean': float(frame.hard_negative_similarity.mean()), 'median': float(frame.hard_negative_similarity.median()),
                               'std': float(frame.hard_negative_similarity.std(ddof=1)) if len(frame) > 1 else None},
                  'fraction_positive_gt_negative': float((delta > 0).mean())}
        # Avoid treating multiple records from one source as independent observations.
        clustered = pd.DataFrame({'group': groups_for(frame), 'delta': delta.to_numpy()}).groupby('group').delta.mean()
        result['wilcoxon_cluster_means_p'] = float(wilcoxon(clustered, method='approx').pvalue) if len(clustered) >= 10 and (clustered != 0).any() else None
        result['wilcoxon_note'] = 'Exploratory; tests source-mean differences, requires >=10 sources; shared negatives still induce dependence.'
        results[annotator] = result
    return results


def language_analysis(config, sample, humans, rows, vectors):
    out = Path(config['output_dir'])
    tables = out / 'tables'
    lookup = {r.key: vectors[i] for i, r in rows.iterrows()}
    def vec(sid, kind):
        return lookup[json.dumps([sid, kind])]
    sample = sample.reset_index(drop=True)
    norms = np.stack([vec(sid, 'norm_text') for sid in sample.sample_id])
    negative = build_negatives(sample, norms)
    negative.to_csv(tables / 'hard_negatives.csv', index=False)
    review_path = tables / 'hard_negative_review.csv'
    if not review_path.exists():
        review = negative.copy()
        review['negative_supported_by_anchor_scene'] = ''
        review['notes'] = ''
        review.to_csv(review_path, index=False)
    reference = sample[sample.reference_evidence.ne('')].copy()
    if reference.sample_id.duplicated().any():
        raise ValueError('Duplicate reference record IDs would corrupt retrieval.')
    # Repeated reference wording is retained as distinct real records and reported.
    refs = np.stack([vec(sid, 'reference_evidence') for sid in reference.sample_id]) if len(reference) else np.empty((0, vectors.shape[1]))
    reference[['sample_id', 'source_video_id', 'reference_evidence']].to_csv(tables / 'retrieval_candidates.csv', index=False)
    reference[reference.reference_evidence.duplicated(keep=False)].to_csv(tables / 'duplicate_reference_text.csv', index=False)
    ref_positions = {sid: i for i, sid in enumerate(reference.sample_id)}
    pairs, ranks, exclusions = [], [], []
    neg_lookup = negative.set_index('sample_id')
    for h in humans[humans.response_status.eq('completed')].itertuples():
        if h.sample_id not in ref_positions:
            exclusions.append({'sample_id': h.sample_id, 'annotator': h.annotator, 'issue': 'no_reference_evidence'})
            continue
        query = vec(h.sample_id, 'human_evidence_' + h.annotator)
        pos = ref_positions[h.sample_id]
        rank = retrieval_rank(query, refs, pos)
        ranks.append({'sample_id': h.sample_id, 'source_video_id': h.source_video_id, 'annotator': h.annotator,
                      'rank': rank, 'recall_at_1': int(rank <= 1), 'recall_at_5': int(rank <= 5),
                      'reciprocal_rank': 1 / rank, 'candidate_count': len(reference)})
        nid = neg_lookup.loc[h.sample_id, 'negative_id']
        if nid:
            pairs.append({'sample_id': h.sample_id, 'source_video_id': h.source_video_id, 'annotator': h.annotator,
                          'negative_id': nid, 'positive_similarity': float(query @ refs[pos]),
                          'hard_negative_similarity': float(query @ vec(nid, 'reference_evidence'))})
        else:
            exclusions.append({'sample_id': h.sample_id, 'annotator': h.annotator, 'issue': 'no_eligible_hard_negative'})
    pd.DataFrame(exclusions, columns=['sample_id', 'annotator', 'issue']).to_csv(tables / 'language_exclusions.csv', index=False)
    pair_frame = pd.DataFrame(pairs, columns=['sample_id', 'source_video_id', 'annotator', 'negative_id', 'positive_similarity', 'hard_negative_similarity'])
    pair_frame.to_csv(tables / 'similarity_pairs.csv', index=False)
    retrieval = pd.DataFrame(ranks, columns=['sample_id', 'source_video_id', 'annotator', 'rank', 'recall_at_1', 'recall_at_5', 'reciprocal_rank', 'candidate_count'])
    retrieval.to_csv(tables / 'retrieval_ranks.csv', index=False)
    retrieval_summary = {}
    for annotator, frame in retrieval.groupby('annotator'):
        retrieval_summary[annotator] = {m: cluster_mean_ci(frame[m], groups_for(frame), config['seed'], config['bootstrap_replicates'])
                                        for m in ['recall_at_1', 'recall_at_5', 'reciprocal_rank']}
        retrieval_summary[annotator]['median_rank'] = float(frame['rank'].median())
        bar_plot({m: retrieval_summary[annotator][m]['mean'] for m in ['recall_at_1', 'recall_at_5']}, out, f'B3_retrieval_{annotator}', 'Recall (conservative tie ranks)')
    save_json({'status': 'pending_human_evidence' if retrieval.empty else 'exploratory',
               'candidate_count': len(reference), 'similarity': summarize_pairs(pair_frame, config),
               'retrieval': retrieval_summary}, tables / 'language_statistics.json')
    if len(pair_frame):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([pair_frame.positive_similarity, pair_frame.hard_negative_similarity], labels=['Same example', 'Hard negative'])
        ax.set_ylabel('Cosine similarity (all completed annotators)')
        figure_save(fig, out, 'B1_similarity')
    specificity = []
    same_norm_pairs = []
    for i, r in sample.iterrows():
        if not r.reference_evidence:
            continue
        evidence = vec(r.sample_id, 'reference_evidence')
        pool = sample[sample.norm_text.ne(r.norm_text)]
        if r.language:
            pool = pool[pool.language.eq(r.language)]
        if len(pool):
            j = pool.index[int(np.argmax(norms[pool.index] @ norms[i]))]
            specificity.append({'sample_id': r.sample_id, 'other_id': sample.loc[j, 'sample_id'],
                                'evidence_own_norm': float(evidence @ norms[i]), 'evidence_nearby_other_norm': float(evidence @ norms[j])})
        same = sample[(sample.index > i) & sample.norm_text.eq(r.norm_text) & sample.reference_evidence.ne('')]
        if r.source_video_id:
            same = same[same.source_video_id.ne(r.source_video_id)]
        else:
            same = same.iloc[:0]  # Cannot claim different video without IDs.
        for other in same.itertuples():
            same_norm_pairs.append({'sample_id': r.sample_id, 'other_id': other.sample_id,
                                    'similarity': float(evidence @ vec(other.sample_id, 'reference_evidence'))})
    pd.DataFrame(specificity, columns=['sample_id', 'other_id', 'evidence_own_norm', 'evidence_nearby_other_norm']).to_csv(tables / 'norm_event_specificity.csv', index=False)
    pd.DataFrame(same_norm_pairs, columns=['sample_id', 'other_id', 'similarity']).to_csv(tables / 'same_norm_different_video.csv', index=False)
    specific_frame = pd.DataFrame(specificity)
    specificity_summary = {'matched_norm_comparisons': len(specificity), 'same_norm_cross_video_pairs': len(same_norm_pairs)}
    if len(specific_frame):
        specific_frame = specific_frame.merge(sample[['sample_id', 'source_video_id']], validate='one_to_one')
        delta = specific_frame.evidence_own_norm - specific_frame.evidence_nearby_other_norm
        specificity_summary['own_minus_nearby_norm'] = cluster_mean_ci(delta, groups_for(specific_frame), config['seed'], config['bootstrap_replicates'])
    if same_norm_pairs:
        specificity_summary['same_norm_cross_video_mean_similarity'] = float(np.mean([p['similarity'] for p in same_norm_pairs]))
    save_json(specificity_summary, tables / 'norm_event_statistics.json')
    if len(reference) >= 4:
        perplexity = min(30, max(2, (len(reference) - 1) / 3))
        coordinates = TSNE(n_components=2, perplexity=perplexity, random_state=config['seed'], init='pca', learning_rate='auto', max_iter=1000).fit_transform(refs)
        coords = reference[['sample_id', 'gold_norm_judgment']].copy()
        coords['x'], coords['y'] = coordinates.T
        coords.to_csv(tables / 'tsne_coordinates.csv', index=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        for label in sorted(coords.gold_norm_judgment.unique()):
            sub = coords[coords.gold_norm_judgment.eq(label)]
            ax.scatter(sub.x, sub.y, label=label, alpha=.8)
        ax.set(xlabel='t-SNE 1', ylabel='t-SNE 2', title='Exploratory evidence representation')
        ax.legend()
        figure_save(fig, out, 'B2_tsne')
        save_json({'seed': config['seed'], 'perplexity': perplexity, 'n': len(reference), 'max_iter': 1000,
                   'warning': 'Exploratory visualization, not quantitative evidence of grounding.'}, tables / 'tsne_parameters.json')
    LOG.info('Language analysis: %d candidate references, %d pairs, %d human queries.', len(reference), len(pair_frame), len(retrieval))
