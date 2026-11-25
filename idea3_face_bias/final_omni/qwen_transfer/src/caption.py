"""Caption channel: the cue is drawn on a white strip appended BELOW the frame strip (no scene pixel is covered).
Deterministic: fixed font file, fixed size for every sentence, greedy word wrap."""
import functools, os
from pathlib import Path
import yaml
from PIL import Image, ImageDraw, ImageFont
from data import CONFIG

CFG = yaml.safe_load((CONFIG / 'channels.yaml').read_text())['caption']


@functools.lru_cache(1)
def font():
    import matplotlib
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'), CFG['font_px'])


def wrap(text, width):
    lines, cur = [], ''
    for w in text.split():
        t = (cur + ' ' + w).strip()
        if font().getlength(t) <= width or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + [cur]


def add_caption(img, text):
    pad, lh = CFG['pad_px'], int(CFG['font_px'] * CFG['line_spacing'])
    lines = wrap(text, img.width - 2 * pad)
    h = 2 * pad + lh * len(lines)
    out = Image.new('RGB', (img.width, img.height + h), 'white')
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    for i, l in enumerate(lines):
        d.text((pad, img.height + pad + i * lh), l, fill='black', font=font())
    return out
