"""Inspect the verified public schema and obtain only selected one-minute clips."""
import hashlib
import html
import json
import logging
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from urllib.request import urlopen

import pandas as pd

from analysis.idea2.data import digest, save_json

LOG = logging.getLogger(__name__)
SOURCE = 'https://raw.githubusercontent.com/CMU-MultiComp-Lab/social-genome'
FIELDS = {'qid', 'q', 'vid_name', 'ts', 'a0', 'a1', 'a2', 'a3'}


def metadata(config, download=False):
    root = Path(config['raw_dir'])
    root.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name in ['qa_test.json', 'videos.json', 'trims.json']:
        path = root / name
        url = f'{SOURCE}/{config["revision"]}/{name}'
        if download and not path.exists():
            with urlopen(url, timeout=30) as f:
                value = f.read(2_000_001)
            if len(value) > 2_000_000:
                raise ValueError('Unexpectedly large metadata file; inspect release manually.')
            json.loads(value)
            path.write_bytes(value)
        provenance.append({'file': name, 'url': url, 'sha256': digest(path), 'bytes': path.stat().st_size})
    rows = json.loads((root / 'qa_test.json').read_text())
    videos = json.loads((root / 'videos.json').read_text())
    trims = json.loads((root / 'trims.json').read_text())
    if any(set(r) != FIELDS for r in rows):
        raise ValueError('Public question schema changed; inspect before adapting.')
    frame = pd.DataFrame(rows)
    if not frame.qid.is_unique or frame.isna().any().any() or frame.eq('').any().any():
        raise ValueError('Public questions contain duplicate IDs or missing fields.')
    if set(videos) != set(frame.vid_name) or len(videos) != len(set(videos)):
        raise ValueError('Video manifest and question source IDs disagree.')
    out = Path(config['output_dir']) / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    missing = frame[~frame.vid_name.isin(trims)].copy()
    missing[['qid', 'vid_name']].to_csv(out / 'missing_trims.csv', index=False)
    frame.groupby('vid_name').size().rename('questions').to_csv(out / 'questions_per_video.csv')
    mapping = {'qid': 'native question ID', 'q': 'social-inference question (not a norm)',
               'vid_name': 'YouTube source-video ID', 'ts': 'question time range relative to supplied clip',
               **{f'a{i}': f'candidate answer {i}; no correctness implied' for i in range(4)}}
    save_json({'rows': len(frame), 'source_videos': frame.vid_name.nunique(),
               'source_videos_with_trims': len(set(frame.vid_name) & set(trims)),
               'questions_without_trims': len(missing), 'schema_mapping': mapping,
               'gold_answers_available': False, 'reference_evidence_available': False,
               'split': 'public benchmark test inputs; exploratory study, not training',
               'provenance': provenance}, out / 'dataset_inspection.json')
    save_json(provenance, root / 'provenance.json')
    LOG.info('Verified %d questions, %d sources, %d rows lacking trims.', len(frame), frame.vid_name.nunique(), len(missing))
    return frame, trims


def source_order(frame, seed):
    ids = sorted(frame.vid_name.unique())
    random.Random(seed).shuffle(ids)
    return ids


def study_sources(config, frame):
    """Use frozen IDs on replication; unavailable locked sources never get replaced."""
    if not config.get('selection_lock'):
        return source_order(frame, config['seed'])
    lock = pd.read_csv(config['selection_lock'], dtype=str, keep_default_na=False)
    if len(lock) != config['sample_size'] or not lock.source_video_id.is_unique or not lock.qid.is_unique:
        raise ValueError('Selection lock must contain the requested number of unique sources/questions.')
    actual = frame.set_index('qid').vid_name.to_dict()
    if any(actual.get(r.qid) != r.source_video_id for r in lock.itertuples()):
        raise ValueError('Selection lock does not match the pinned public metadata.')
    return lock.source_video_id.tolist()


def seconds(text):
    parts = text.replace(',', '.').split(':')
    return sum(float(x) * 60**i for i, x in enumerate(reversed(parts)))


def trim_vtt(text, start, duration=60, rolling=False):
    """Intersect full-source caption intervals with clip; deduplicate rolling ASR words."""
    result = []
    previous = []
    for block in re.split(r'\n\s*\n', text.replace('\r\n', '\n')):
        lines = block.splitlines()
        pos = next((i for i, line in enumerate(lines) if '-->' in line), None)
        if pos is None:
            continue
        stamps = lines[pos].split('-->')
        begin, end = seconds(stamps[0].strip()), seconds(stamps[1].strip().split()[0])
        if end <= start or begin >= start + duration:
            continue
        content = html.unescape(re.sub(r'<[^>]+>', '', ' '.join(lines[pos+1:]))).strip()
        words = content.split()
        overlap = 0
        if rolling:
            for n in range(1, min(len(previous), len(words)) + 1):
                if previous[-n:] == words[:n]:
                    overlap = n
        fresh = words[overlap:]
        if fresh:
            result.append({'start': round(max(0, begin-start), 3), 'end': round(min(duration, end-start), 3), 'text': ' '.join(fresh)})
        previous = words
    return result


def refresh_captions(config):
    """Offline deterministic caption processing before freezing annotation inputs."""
    media = Path(config['raw_dir']) / 'media'
    for path in sorted(media.glob('*/complete.json')):
        info = json.loads(path.read_text())
        segments = trim_vtt((path.parent / 'source.en.vtt').read_text(), info['source_start'],
                            rolling=info['caption_source'] == 'youtube_automatic_captions')
        if not segments:
            raise ValueError(f'No captions for completed video {info["source_video_id"]}')
        save_json(segments, path.parent / 'transcript_segments.json')
        (path.parent / 'transcript.txt').write_text('\n'.join(f'[{s["start"]:.2f}–{s["end"]:.2f}s] {s["text"]}' for s in segments))
        info['transcript_sha256'] = digest(path.parent / 'transcript.txt')
        info['caption_processing'] = 'v2: preserve uploader repetition; deduplicate rolling automatic captions'
        save_json(info, path)


def command(args, log, timeout=180):
    env = dict(os.environ)
    extra = str(Path('.cache/social_genome_tools').resolve())
    env['PYTHONPATH'] = extra + os.pathsep + env.get('PYTHONPATH', '')
    with Path(log).open('w') as stream:
        result = subprocess.run(args, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        tail = Path(log).read_text(errors='replace')[-1800:]
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}: {tail}')


def fetch_media(config, limit=None):
    frame, trims = metadata(config)
    root = Path(config['raw_dir'])
    logs = Path(config['output_dir']) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    media = root / 'media'
    media.mkdir(exist_ok=True)
    statuses = []
    target = config['sample_size'] if limit is None else limit
    ready = 0
    common = [sys.executable, '-m', 'yt_dlp', '--ignore-config', '--no-cache-dir', '--no-playlist',
              '--socket-timeout', '15', '--retries', '1', '--extractor-retries', '1', '--no-progress']
    for vid in study_sources(config, frame)[:config['max_download_attempts']]:
        record = {'source_video_id': vid, 'status': '', 'reason': ''}
        if vid not in trims:
            record.update(status='excluded', reason='missing_trim')
            statuses.append(record)
            continue
        folder = media / vid
        folder.mkdir(exist_ok=True)
        try:
            if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > config['max_disk_gb'] * 1024**3:
                raise ValueError('Download disk budget exceeded.')
            if not (folder / 'complete.json').exists():
                url = f'https://www.youtube.com/watch?v={vid}'
                info = folder / 'source.info.json'
                captions = folder / 'source.en.vtt'
                if not info.exists() or not captions.exists():
                    command(common + ['--skip-download', '--write-info-json', '--write-subs', '--write-auto-subs',
                                      '--sub-langs', 'en', '--sub-format', 'vtt', '-o', str(folder / 'source.%(ext)s'), url], logs / f'{vid}_metadata.log', 100)
                if not captions.exists():
                    raise RuntimeError('No downloadable English VTT captions.')
                meta = json.loads(info.read_text())
                language = meta.get('language') or ''
                if language and not language.startswith('en'):
                    raise RuntimeError(f'Original language metadata is {language}, not English.')
                segments = trim_vtt(captions.read_text(), float(trims[vid]))
                if not segments:
                    raise RuntimeError('No English captions overlap annotated clip.')
                start = float(trims[vid])
                clip = folder / 'clip.mp4'
                if not clip.exists() or clip.stat().st_size == 0:
                    command(common + ['--load-info-json', str(info), '--download-sections', f'*{start}-{start+60}',
                                      '--force-keyframes-at-cuts', '--format', 'bv[height<=480][vcodec^=avc1]',
                                      '-o', str(folder / 'clip.%(ext)s')], logs / f'{vid}_download.log', 180)
                probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(clip)], text=True))
                duration = float(probe['format']['duration'])
                if not 55 <= duration <= 62 or any(s['codec_type'] == 'audio' for s in probe['streams']):
                    raise RuntimeError(f'Unexpected clip duration/audio streams: {duration}')
                save_json(segments, folder / 'transcript_segments.json')
                (folder / 'transcript.txt').write_text('\n'.join(f'[{s["start"]:.1f}–{s["end"]:.1f}s] {s["text"]}' for s in segments))
                save_json({'source_video_id': vid, 'source_url': url, 'source_start': start, 'duration': duration,
                           'caption_source': 'uploader_subtitles' if 'en' in meta.get('subtitles', {}) else 'youtube_automatic_captions',
                           'source_language_metadata': language or 'unknown; human verification needed',
                           'input': 'silent video + English caption transcript', 'clip_sha256': digest(clip),
                           'transcript_sha256': digest(folder / 'transcript.txt')}, folder / 'complete.json')
            record['status'] = 'ready'
            ready += 1
            LOG.info('Ready %d/%d: %s', ready, target, vid)
        except (RuntimeError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            record.update(status='unavailable', reason=str(exc))
            LOG.warning('Unavailable %s: %s', vid, str(exc)[:180])
        statuses.append(record)
        pd.DataFrame(statuses).to_csv(logs / 'media_access.csv', index=False)
        if ready >= target:
            break
    LOG.info('Media check ended: %d ready, %d attempted.', ready, len(statuses))
    return ready


def fetch_audio(config):
    """Add matching source audio without changing previously frozen silent clips."""
    frame, trims = metadata(config)
    root = Path(config['raw_dir'])
    logs = Path(config['output_dir']) / 'logs'
    statuses = []
    for vid in study_sources(config, frame):
        folder = root / 'media' / vid
        record = {'source_video_id': vid, 'status': '', 'reason': ''}
        try:
            original = json.loads((folder / 'complete.json').read_text())
            complete = folder / 'audio_complete.json'
            clip = folder / 'clip_av.mp4'
            if complete.exists():
                saved = json.loads(complete.read_text())
                if digest(clip) != saved['clip_sha256']:
                    raise ValueError(f'Changed audio/video clip: {vid}')
            else:
                if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > config['max_disk_gb'] * 1024**3:
                    raise ValueError('Download disk budget exceeded.')
                start = float(trims[vid])
                audio = folder / 'audio.m4a'
                if not audio.exists():
                    audio_command = [sys.executable, '-m', 'yt_dlp', '--ignore-config', '--no-cache-dir', '--no-playlist',
                             '--socket-timeout', '15', '--retries', '1', '--no-progress',
                             '--download-sections', f'*{start}-{start+60}', '--force-keyframes-at-cuts',
                             '--no-continue', '-o', str(audio)]
                    url = f'https://www.youtube.com/watch?v={vid}'
                    try:
                        command(audio_command + ['-f', 'ba[ext=m4a]', url], logs / f'{vid}_audio_download.log', 180)
                    except RuntimeError:
                        # Some direct audio URLs fail while the public English HLS track works.
                        command(audio_command + ['-f', 'ba[protocol=m3u8_native][language^=en]', url],
                                logs / f'{vid}_audio_hls.log', 180)
                command(['ffmpeg', '-y', '-v', 'error', '-i', str(folder / 'clip.mp4'), '-i', str(audio),
                         '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-t', '60',
                         '-movflags', '+faststart', str(clip)], logs / f'{vid}_mux.log', 60)
                probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(clip)], text=True))
                streams = {s['codec_type']: s for s in probe['streams']}
                if set(streams) != {'video', 'audio'} or any(not 59 <= float(s['duration']) <= 61 for s in streams.values()):
                    raise RuntimeError('Audio/video stream missing or not approximately 60 seconds.')
                if any(abs(float(s.get('start_time', 0))) > .1 for s in streams.values()):
                    raise RuntimeError('Unexpected audio/video start offset.')
                save_json({**original, 'input': 'video + audio + English caption transcript',
                           'clip_sha256': digest(clip), 'silent_clip_sha256': original['clip_sha256'],
                           'audio_sha256': digest(audio), 'audio_source_start': start,
                           'audio_duration': float(streams['audio']['duration'])}, complete)
            record['status'] = 'ready'
            LOG.info('Audio ready %s', vid)
        except (RuntimeError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            record.update(status='unavailable', reason=str(exc))
            LOG.warning('Audio unavailable %s: %s', vid, str(exc)[:180])
        statuses.append(record)
        pd.DataFrame(statuses).to_csv(logs / 'audio_access.csv', index=False)
    if any(s['status'] != 'ready' for s in statuses):
        raise RuntimeError('Some original audio unavailable; inspect audio_access.csv. No silent fallback supplied.')
