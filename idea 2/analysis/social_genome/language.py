"""Unsupervised public-language exploration and optional cross-human comparisons."""
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score

from analysis.idea2.data import save_json
from analysis.idea2.statistics import bar_plot, cluster_mean_ci, figure_save
from analysis.idea2.language import cache_key, load_cache, retrieval_rank
import matplotlib.pyplot as plt

LOG = logging.getLogger(__name__)


def language_rows(sample, humans):
    rows = []
    for r in sample.to_dict('records'):
        for field in ['question', 'transcript', 'a0', 'a1', 'a2', 'a3']:
            text = r[field]
            if field == 'transcript':
                text = re.sub(r'\[\d+(?:\.\d+)?–\d+(?:\.\d+)?s\]\s*', '', text)
            rows.append({'key': json.dumps([r['sample_id'], field]), 'sample_id': r['sample_id'], 'representation': field, 'text': text})
    for r in humans[humans.response_status.eq('completed')].to_dict('records'):
        kind = 'human_' + r['annotator']
        rows.append({'key': json.dumps([r['sample_id'], kind]), 'sample_id': r['sample_id'], 'representation': kind, 'text': r['human_evidence']})
    return pd.DataFrame(rows).sort_values('key').reset_index(drop=True)


def embed_texts(config, rows):
    cfg = config['embedding']
    revision = cfg['revision']
    if not revision or not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Pin encoder revision to a 40-character model commit.')
    if rows.key.duplicated().any() or rows.text.str.strip().eq('').any():
        raise ValueError('Embedding rows must have unique keys and nonempty text.')
    # Pooling scheme is part of the cache namespace; no stale single-window vectors.
    folder = Path(config['output_dir']) / 'embeddings' / ('chunkmean-v1-' + cache_key(rows, cfg['model'], revision)[:16])
    if (folder / 'manifest.json').exists():
        return load_cache(folder, rows, cfg['model'], revision)
    from sentence_transformers import SentenceTransformer
    import torch
    torch.manual_seed(config['seed'])
    torch.set_num_threads(2)
    model = SentenceTransformer(cfg['model'], revision=revision, device=cfg['device'],
                                cache_folder='.cache/sentence_transformers', trust_remote_code=False)
    model.eval()
    width = model.max_seq_length - model.tokenizer.num_special_tokens_to_add(pair=False)
    all_chunks, ranges, diagnostics = [], [], []
    for row in rows.itertuples():
        tokens = model.tokenizer.encode(row.text, add_special_tokens=False, truncation=False)
        chunks = [model.tokenizer.decode(tokens[i:i+width], skip_special_tokens=True) for i in range(0, len(tokens), width)]
        ranges.append((len(all_chunks), len(all_chunks)+len(chunks)))
        all_chunks.extend(chunks)
        diagnostics.append({'key': row.key, 'tokens': len(tokens), 'chunks': len(chunks), 'text_truncated': False})
    # Decode/re-tokenize can alter boundaries; fail rather than silently truncate.
    if any(len(model.tokenizer.encode(x, truncation=False)) > model.max_seq_length for x in all_chunks):
        raise ValueError('Chunk re-tokenization exceeded encoder limit; inspect tokenizer behavior.')
    encoded = model.encode(all_chunks, batch_size=cfg['batch_size'], normalize_embeddings=True, show_progress_bar=True)
    vectors = np.stack([encoded[a:b].mean(axis=0) for a, b in ranges])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    folder.mkdir(parents=True, exist_ok=True)
    rows.to_csv(folder / 'rows.csv', index=False)
    np.save(folder / 'embeddings.npy', vectors, allow_pickle=False)
    pd.DataFrame(diagnostics).to_csv(folder / 'text_chunks.csv', index=False)
    save_json({'fingerprint': cache_key(rows, cfg['model'], revision),
               'array_sha256': hashlib.sha256(vectors.tobytes()).hexdigest(), 'model': cfg['model'], 'revision': revision,
               'pooling': 'nonoverlapping token windows; equal mean of normalized chunk embeddings, then normalize',
               'max_tokens_per_chunk': width, 'seed': config['seed'], 'rows': len(rows), 'chunks': len(all_chunks),
               'packages': {p: importlib.metadata.version(p) for p in ['sentence-transformers', 'transformers', 'torch', 'numpy']}}, folder / 'manifest.json')
    return load_cache(folder, rows, cfg['model'], revision)


def nearest_other(questions, source_ids):
    similarities = questions @ questions.T
    result = []
    for i, source in enumerate(source_ids):
        eligible = np.array([j for j, other in enumerate(source_ids) if j != i and other != source])
        if not len(eligible):
            raise ValueError('Need at least two distinct sources for comparison.')
        result.append(int(eligible[np.argmax(similarities[i, eligible])]))
    return result


def cross_human(config, sample, humans, lookup, negative_indices):
    out = Path(config['output_dir'])
    tables = out / 'tables'
    completed = humans[humans.response_status.eq('completed')]
    summary = {'status': 'pending_humans', 'directions': {},
               'interpretation': 'Cross-human consistency, not gold-reference accuracy or proven grounding.'}
    if len(config['annotators']) < 2:
        summary.update(status='not_applicable_single_annotator', interpretation='One annotator: no independent second-human evidence or cross-human retrieval is available.')
    comparison, retrieval_rows, exclusions = [], [], []
    def vector(sid, name):
        return lookup[json.dumps([sid, 'human_' + name])]
    for anchor_name in config['annotators']:
        for other_name in config['annotators']:
            if other_name == anchor_name:
                continue
            anchors = completed[completed.annotator.eq(anchor_name)]
            targets = completed[completed.annotator.eq(other_name)].sort_values('sample_id')
            target_ids = targets.sample_id.tolist()
            if not target_ids:
                continue
            refs = np.stack([vector(sid, other_name) for sid in target_ids])
            for h in anchors.itertuples():
                if h.sample_id not in target_ids:
                    exclusions.append({'sample_id': h.sample_id, 'direction': anchor_name+'->'+other_name, 'reason': 'other_human_response_missing'})
                    continue
                query = vector(h.sample_id, anchor_name)
                rank = retrieval_rank(query, refs, target_ids.index(h.sample_id))
                same_answer = h.selected_answer == targets.set_index('sample_id').loc[h.sample_id, 'selected_answer']
                retrieval_rows.append({'sample_id': h.sample_id, 'direction': anchor_name+'->'+other_name, 'rank': rank,
                                       'recall_at_1': int(rank <= 1), 'recall_at_5': int(rank <= 5),
                                       'reciprocal_rank': 1/rank, 'candidate_count': len(target_ids), 'same_answer': same_answer})
                pos = sample.index[sample.sample_id.eq(h.sample_id)][0]
                negative = sample.iloc[negative_indices[pos]].sample_id
                if negative not in target_ids:
                    exclusions.append({'sample_id': h.sample_id, 'direction': anchor_name+'->'+other_name, 'reason': 'preselected_negative_response_missing'})
                    continue
                comparison.append({'sample_id': h.sample_id, 'direction': anchor_name+'->'+other_name,
                                   'negative_id': negative, 'positive': float(query @ vector(h.sample_id, other_name)),
                                   'negative': float(query @ vector(negative, other_name)), 'same_answer': same_answer})
    pairs = pd.DataFrame(comparison, columns=['sample_id', 'direction', 'negative_id', 'positive', 'negative', 'same_answer'])
    ranks = pd.DataFrame(retrieval_rows, columns=['sample_id', 'direction', 'rank', 'recall_at_1', 'recall_at_5', 'reciprocal_rank', 'candidate_count', 'same_answer'])
    pairs.to_csv(tables / 'cross_human_similarity.csv', index=False)
    ranks.to_csv(tables / 'cross_human_retrieval.csv', index=False)
    pd.DataFrame(exclusions, columns=['sample_id', 'direction', 'reason']).to_csv(tables / 'cross_human_exclusions.csv', index=False)
    for direction, group in ranks.groupby('direction'):
        summary['status'] = 'exploratory_human_results'
        values = {m: cluster_mean_ci(group[m], group.sample_id, config['seed'], config['bootstrap_replicates']) for m in ['recall_at_1', 'recall_at_5', 'reciprocal_rank']}
        values['median_rank'] = float(group['rank'].median())
        matched = pairs[pairs.direction.eq(direction)]
        if len(matched):
            delta = matched.positive - matched.negative
            values['paired_difference'] = cluster_mean_ci(delta, matched.sample_id, config['seed'], config['bootstrap_replicates'])
            values['fraction_positive_gt_negative'] = float((delta > 0).mean())
            values['positive_description'] = {k: None if pd.isna(v) else float(v) for k, v in matched.positive.describe().items()}
            values['negative_description'] = {k: None if pd.isna(v) else float(v) for k, v in matched.negative.describe().items()}
            values['wilcoxon_p_exploratory'] = float(wilcoxon(delta, method='approx').pvalue) if len(delta) >= 10 and delta.ne(0).any() else None
            same = matched[matched.same_answer]
            values['same_answer_only_difference'] = cluster_mean_ci(same.positive-same.negative, same.sample_id, config['seed'], config['bootstrap_replicates'])
        summary['directions'][direction] = values
        bar_plot({m: values[m]['mean'] for m in ['recall_at_1', 'recall_at_5']}, out, 'B3_retrieval_'+direction.replace('->','_to_'), 'Cross-human retrieval recall')
    if len(pairs):
        fig, ax = plt.subplots(figsize=(6,4))
        ax.boxplot([pairs.positive, pairs.negative], labels=['Same question', 'Nearby question, other video'])
        ax.set_ylabel('Cross-human evidence cosine similarity')
        figure_save(fig, out, 'B1_cross_human_similarity')
    save_json(summary, tables / 'cross_human_statistics.json')


def analyze_language(config, sample, humans):
    sample = sample.reset_index(drop=True)
    rows = language_rows(sample, humans)
    vectors = embed_texts(config, rows)
    lookup = {r.key: vectors[i] for i, r in rows.iterrows()}
    def matrix(kind):
        return np.stack([lookup[json.dumps([sid, kind])] for sid in sample.sample_id])
    qvec, tvec = matrix('question'), matrix('transcript')
    negative = nearest_other(qvec, sample.source_video_id)
    tables = Path(config['output_dir']) / 'tables'
    comparison = pd.DataFrame({'sample_id': sample.sample_id, 'other_id': sample.iloc[negative].sample_id.to_numpy(),
                               'own_transcript_similarity': (qvec*tvec).sum(axis=1),
                               'other_transcript_similarity': (qvec*tvec[negative]).sum(axis=1),
                               'question_similarity': (qvec*qvec[negative]).sum(axis=1)})
    comparison.to_csv(tables / 'question_transcript_comparison.csv', index=False)
    pd.DataFrame({'sample_id': sample.sample_id, 'negative_id': sample.iloc[negative].sample_id.to_numpy(),
                  'selection': 'nearest question embedding among other sources; independent of human responses'}).to_csv(tables / 'candidate_hard_negatives.csv', index=False)
    summary = {'n': len(sample), 'labels': 'no gold labels used; cluster IDs are algorithmic assignments',
               'question_own_minus_other_transcript': cluster_mean_ci(comparison.own_transcript_similarity-comparison.other_transcript_similarity,
                                                                      sample.source_video_id, config['seed'], config['bootstrap_replicates']),
               'note': 'Question/transcript semantic association is not evidence correctness.', 'representations': {}}
    for kind, embeddings in [('question', qvec), ('transcript', tvec)]:
        selected = rows[rows.representation.eq(kind)].set_index('sample_id').loc[sample.sample_id]
        texts = selected.text.tolist()
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=3000)
        lexical = tfidf.fit_transform(texts)
        k = min(4, len(sample)-1)  # Predeclared descriptive setting, never tuned against outcomes.
        if k < 2:
            raise ValueError('At least three sampled sources needed for clustering.')
        labels = KMeans(n_clusters=k, n_init=10, random_state=config['seed']).fit_predict(embeddings)
        lexical_labels = KMeans(n_clusters=k, n_init=10, random_state=config['seed']).fit_predict(lexical)
        assignments = pd.DataFrame({'sample_id': sample.sample_id, 'embedding_cluster': labels, 'tfidf_cluster': lexical_labels, 'text': texts})
        assignments.to_csv(tables / f'{kind}_clusters.csv', index=False)
        terms = []
        for cluster in sorted(set(labels)):
            weights = np.asarray(lexical[labels==cluster].mean(axis=0)).ravel()
            top = tfidf.get_feature_names_out()[np.argsort(weights)[-10:][::-1]]
            terms.append({'cluster': int(cluster), 'n': int((labels==cluster).sum()), 'top_lexical_terms': '; '.join(top)})
        pd.DataFrame(terms).to_csv(tables / f'{kind}_cluster_terms.csv', index=False)
        summary['representations'][kind] = {'k': k, 'neural_vs_tfidf_adjusted_rand_index': float(adjusted_rand_score(labels, lexical_labels)),
                                            'neural_cosine_silhouette': float(silhouette_score(embeddings, labels, metric='cosine')) if 1 < len(set(labels)) < len(labels) else None}
        perplexity = min(10, (len(sample)-1)/3)
        xy = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=config['seed'], max_iter=1000).fit_transform(embeddings)
        pd.DataFrame({'sample_id': sample.sample_id, 'x': xy[:,0], 'y': xy[:,1], 'cluster': labels}).to_csv(tables / f'{kind}_tsne.csv', index=False)
        fig, ax = plt.subplots(figsize=(6,5))
        for label in sorted(set(labels)):
            ax.scatter(xy[labels==label,0], xy[labels==label,1], label=f'Cluster {label}', alpha=.85)
        ax.set(xlabel='t-SNE 1', ylabel='t-SNE 2', title=f'{kind.capitalize()} embeddings: exploratory clusters')
        ax.legend()
        figure_save(fig, config['output_dir'], 'B2_'+kind+'_tsne')
        summary['representations'][kind]['tsne_perplexity'] = perplexity
    save_json(summary, tables / 'public_language_statistics.json')
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot([comparison.own_transcript_similarity, comparison.other_transcript_similarity], labels=['Own transcript', 'Other nearby-question transcript'])
    ax.set_ylabel('Question–transcript cosine similarity')
    figure_save(fig, config['output_dir'], 'B0_question_transcript_similarity')
    cross_human(config, sample, humans, lookup, negative)
    human_language(config, sample, humans, lookup, negative, qvec, tvec)
    LOG.info('Public-language analysis complete for %d real questions/transcripts. Human analyses use only completed responses.', len(sample))


def human_language(config, sample, humans, lookup, negative, qvec, tvec):
    """Post-hoc text association diagnostics, never reference-evidence validation."""
    from .results import modality_label
    out = Path(config['output_dir'])
    tables = out / 'tables'
    summary = {'status': 'pending_human_descriptions', 'annotators': {},
               'design': 'Post-hoc single-annotator diagnostics. Transcript targets are input text, not gold evidence. Alternatives fixed by public question similarity.',
               'limitations': 'Human saw questions, choices and transcripts. Text overlap and topic can drive retrieval; visible or non-speech audio observations need not match a transcript. These are not grounding/reward-validity metrics.'}
    all_pairs, ranks = [], []
    positions = {sid: i for i, sid in enumerate(sample.sample_id)}
    for name, group in humans[humans.response_status.eq('completed')].groupby('annotator'):
        group = group.sort_values('sample_id').reset_index(drop=True)
        indices = [positions[sid] for sid in group.sample_id]
        vectors = np.stack([lookup[json.dumps([sid, 'human_' + name])] for sid in group.sample_id])
        pairs = []
        for k, row in enumerate(group.itertuples()):
            i = indices[k]
            alt = negative[i]
            own = float(vectors[k] @ tvec[i])
            other = float(vectors[k] @ tvec[alt])
            chosen = lookup[json.dumps([row.sample_id, 'a'+str('ABCD'.index(row.selected_answer))])]
            pairs.append({'annotator': name, 'sample_id': row.sample_id, 'alternative_id': sample.iloc[alt].sample_id,
                          'evidence_modality': row.evidence_modality, 'own_transcript': own, 'alternative_transcript': other,
                          'difference': own-other, 'own_question': float(vectors[k] @ qvec[i]),
                          'chosen_answer': float(vectors[k] @ chosen)})
            rank = retrieval_rank(vectors[k], tvec, i)
            ranks.append({'annotator': name, 'sample_id': row.sample_id, 'rank': rank, 'candidate_count': len(tvec),
                          'recall_at_1': int(rank <= 1), 'recall_at_5': int(rank <= 5), 'reciprocal_rank': 1/rank})
        pairs = pd.DataFrame(pairs)
        all_pairs.extend(pairs.to_dict('records'))
        rank_frame = pd.DataFrame([r for r in ranks if r['annotator'] == name])
        stats = {'n': len(group), 'candidate_transcripts': len(tvec),
                 'own_minus_alternative': cluster_mean_ci(pairs.difference, group.source_video_id, config['seed'], config['bootstrap_replicates']),
                 'fraction_own_gt_alternative': float(pairs.difference.gt(0).mean()),
                 'means': {field: float(pairs[field].mean()) for field in ['own_transcript','alternative_transcript','own_question','chosen_answer']},
                 'recall_at_1': float(rank_frame.recall_at_1.mean()), 'recall_at_5': float(rank_frame.recall_at_5.mean()),
                 'median_rank': float(rank_frame['rank'].median()), 'mrr': float(rank_frame.reciprocal_rank.mean())}
        if len(group) >= 4:
            lexical = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=3000)
            tfidf = lexical.fit_transform(group.human_evidence)
            k = min(4, len(group)-1)
            labels = KMeans(n_clusters=k, n_init=10, random_state=config['seed']).fit_predict(vectors)
            lexical_labels = KMeans(n_clusters=k, n_init=10, random_state=config['seed']).fit_predict(tfidf)
            stats['clustering'] = {'k': k, 'neural_vs_tfidf_ari': float(adjusted_rand_score(labels, lexical_labels)),
                                   'cosine_silhouette': float(silhouette_score(vectors, labels, metric='cosine')) if 1 < len(set(labels)) < len(labels) else None}
            perplexity = min(10, (len(group)-1)/3)
            xy = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=config['seed'], max_iter=1000).fit_transform(vectors)
            assignments = group[['sample_id', 'human_evidence', 'evidence_modality']].copy()
            assignments = assignments.assign(embedding_cluster=labels, tfidf_cluster=lexical_labels, x=xy[:,0], y=xy[:,1])
            assignments.to_csv(tables/f'human_evidence_structure_{name}.csv', index=False)
            terms = []
            for label in sorted(set(labels)):
                weights = np.asarray(tfidf[labels==label].mean(axis=0)).ravel()
                terms.append({'cluster': int(label), 'n': int((labels==label).sum()),
                              'top_lexical_terms': '; '.join(lexical.get_feature_names_out()[np.argsort(weights)[-10:][::-1]])})
            pd.DataFrame(terms).to_csv(tables/f'human_evidence_cluster_terms_{name}.csv', index=False)
            with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42}):
                fig, ax = plt.subplots(figsize=(7,5))
                for modality in sorted(group.evidence_modality.unique()):
                    mask = group.evidence_modality.eq(modality).to_numpy()
                    ax.scatter(xy[mask,0], xy[mask,1], label=modality_label(modality), alpha=.85)
                ax.set(xlabel='t-SNE 1', ylabel='t-SNE 2', title='Human evidence · exploratory projection')
                ax.legend(title='Reported evidence modality', loc='best', fontsize=9)
                figure_save(fig, out, f'B2_human_evidence_{name}')
            stats['tsne_perplexity'] = perplexity
        summary['status'] = 'exploratory_posthoc'
        summary['annotators'][name] = stats
        with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42}):
            fig, ax = plt.subplots(figsize=(6,4))
            for row in pairs.itertuples():
                ax.plot([0,1], [row.own_transcript,row.alternative_transcript], color='0.7', alpha=.5, linewidth=.7)
            ax.boxplot([pairs.own_transcript, pairs.alternative_transcript], positions=[0,1], labels=['Own transcript','Similar-question transcript'])
            ax.set(ylabel='Human evidence–transcript cosine similarity', title='Input-text association; transcripts are not gold evidence')
            figure_save(fig, out, f'B4_human_transcript_association_{name}')
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(rank_frame['rank'], bins=np.arange(.5, len(tvec)+1.5), rwidth=.85)
            ax.set(xlabel=f'Rank of paired transcript among {len(tvec)} transcripts', ylabel='Human descriptions', title='Transcript retrieval (exploratory input-text diagnostic)')
            figure_save(fig, out, f'B5_human_transcript_ranks_{name}')
    pd.DataFrame(all_pairs).to_csv(tables/'human_transcript_association.csv', index=False)
    pd.DataFrame(ranks).to_csv(tables/'human_transcript_retrieval.csv', index=False)
    save_json(summary, tables/'human_language_statistics.json')
