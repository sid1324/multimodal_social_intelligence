"""Build prompts for the three cue channels. One framing (config/channels.yaml); the remark sits after the image and
before the question. cond is a cue condition name from data.load_cues_all(); 'none' has no remark in any channel."""
import os
from pathlib import Path
import yaml
from PIL import Image
from data import CONFIG, P777, permuted_options, load_cues_all, image_path, seeded_int
from caption import add_caption

CH = yaml.safe_load((CONFIG / 'channels.yaml').read_text())
AUDIO = Path(P777) / 'cache/tts'
CUES = load_cues_all()
CHANNELS = ('text', 'speech', 'image')


def voice_for(item_id):
    return ('female', 'male')[seeded_int(CH['voice_seed'], item_id) % 2]


def audio_path(text, voice):
    import hashlib
    h = hashlib.sha256(f"{CH['voices'][voice]}|{text}|{CH['audio']}".encode()).hexdigest()[:20]
    return AUDIO / f'{h}.wav'


def scene(item_id):
    return Image.open(image_path(item_id)).convert('RGB')


def build(v, img, cond, channel):
    """-> list of parts for omni_model. img: the scene strip (PIL)."""
    cue = CUES[cond]
    head = 'The image shows five frames, in order from left to right, from an egocentric video recorded by the camera wearer.'
    lines = '\n'.join(f'{l}. {b}' for l, _, b in permuted_options(v))
    tail = ('What should the camera wearer do next? Choose the most socially appropriate action.'
            f'\n\n{lines}\n\nRespond with only the letter of your choice and no explanation.')
    if not cue:
        return [('image', img), ('text', f'{head} {tail}')]
    r = CH['remark']
    if channel == 'text':
        return [('image', img), ('text', f"{head} {r} {CH['text'].format(cue=cue)} {tail}")]
    if channel == 'speech':
        return [('image', img), ('text', f"{head} {r} {CH['speech']} "), ('audio', str(audio_path(cue, voice_for(v['id'])))), ('text', f' {tail}')]
    if channel == 'image':
        return [('image', add_caption(img, cue)), ('text', f"{head} {r} {CH['image']} {tail}")]
    raise ValueError(channel)


def build_repeat(v, img, cond, channel):
    """Channel perception check: same inputs, but ask for the remark verbatim."""
    parts = build(v, img, cond, channel)
    out = []
    for k, x in parts:
        if k == 'text' and 'What should the camera wearer do next?' in x:
            x = x[:x.index('What should the camera wearer do next?')] + 'Repeat the remark word for word. Respond with only the remark.'
        out.append((k, x))
    return out
