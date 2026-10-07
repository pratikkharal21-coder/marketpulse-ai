import io
import logging
import re
import time
import urllib.parse
import urllib.request

import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

from chart import GREEN, RED, WATERMARK_HANDLE

logger = logging.getLogger("marketpulse.illustration")

POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt/"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 576

# The anonymous/keyless tier is throttled to ~1 request/15s and always carries Pollinations'
# own small watermark -- confirmed in testing that passing nologo=true without a registered
# account triggers a 402 (it routes into the paid-account code path, which then has no budget),
# so it's deliberately left off here rather than chased. One retry after the full throttle
# window covers a request that lands just inside someone else's 15s window.
_RETRY_ATTEMPTS = 1
_RETRY_DELAY_SECONDS = 16

# Diffusion image models (used here for the background art, via the free/keyless Pollinations.ai
# endpoint) can't reliably spell words -- any text baked in by the model itself comes back as
# garbled nonsense (confirmed in testing against the real endpoint). So the scene prompt
# deliberately describes only the scene and explicitly asks for none -- every piece of readable
# text in the final image is drawn separately with PIL in compose_image, the same
# self-rendered-text approach chart.py/memeart.py already use everywhere else in this pipeline,
# just applied to a generated background instead of a blank canvas.
SCENE_PROMPT_PREFIX = (
    "Flat-color editorial cartoon illustration, bold simple shapes, finance-infographic style, "
    "no text, no words, no letters, no logos, no watermarks. Generic unnamed invented characters "
    "only, never a real or recognizable person. Scene: "
)

# Cheap, deterministic heuristic for "this scene describes a real, specifically-named
# individual": a consecutive run of 2-3 Title-Case words (e.g. "Dennis Hastert", "Jerome
# Powell"). Shared by both callers (satire_image.py, news_image.py) as a hard runtime backstop
# -- the prompt already instructs "never a real or recognizable person", but that instruction
# alone isn't trustworthy enough on its own (confirmed in testing: a real named individual made
# it into generated text despite an equivalent instruction). Rejecting the whole image rather
# than trying to rewrite the scene text keeps this simple and conservative.
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")


def scene_names_real_individual(scene, reference_text):
    """True if a capitalized multi-word name found in `reference_text` (typically the story's
    own headline) also appears in `scene` -- i.e. the illustration prompt is describing that
    specific real person rather than a generic stand-in."""
    if not scene or not reference_text:
        return False
    reference_names = set(_NAME_RE.findall(reference_text))
    if not reference_names:
        return False
    return any(name in scene for name in reference_names)


_BOLD_FONT_PATH = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))


def _font(size):
    return ImageFont.truetype(_BOLD_FONT_PATH, size)


def fetch_background(scene):
    prompt = SCENE_PROMPT_PREFIX + scene
    url = POLLINATIONS_BASE_URL + urllib.parse.quote(prompt) + f"?width={IMAGE_WIDTH}&height={IMAGE_HEIGHT}"
    req = urllib.request.Request(url, headers={"User-Agent": "MarketPulseAI/1.0 (single-user personal news digest)"})

    last_exc = None
    for attempt in range(1, _RETRY_ATTEMPTS + 2):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        except Exception as exc:
            last_exc = exc
            if attempt <= _RETRY_ATTEMPTS:
                logger.warning(
                    "Illustration background fetch failed (attempt %d/%d), retrying in %ds: %s",
                    attempt, _RETRY_ATTEMPTS + 1, _RETRY_DELAY_SECONDS, exc,
                )
                time.sleep(_RETRY_DELAY_SECONDS)
    raise last_exc


def _draw_outlined_text(draw, xy, text, font, fill, outline_width=3, outline_fill="black", anchor=None):
    x, y = xy
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline_fill, anchor=anchor)
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)


def _wrap(text, font, max_width, max_lines=3):
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if font.getlength(trial) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;: ") + "…"
    return lines


def compose_image(background, caption, stat_text=None, badge_text=None, badge_color=(185, 28, 28)):
    """Draws a bold poster-style caption, an optional colored stat callout, an optional
    bottom-left disclosure badge, and the account watermark on top of `background`. `badge_text`
    is deliberately a required-by-convention (not hardcoded) parameter -- satire_image.py and
    news_image.py each burn in a different disclosure ("SATIRE -- NOT REAL NEWS" vs "AI
    ILLUSTRATION -- NOT A REAL PHOTO") since the two badges mean different things and must not
    be confused with each other. Pass badge_text=None to skip the badge entirely."""
    canvas = background.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    # Dark gradient scrim behind the top caption band so bold white text stays legible no
    # matter what the generated background looks like underneath it -- same idea as a real
    # poster/meme text overlay.
    scrim_height = int(h * 0.34)
    scrim = Image.new("RGBA", (w, scrim_height), (0, 0, 0, 0))
    scrim_draw = ImageDraw.Draw(scrim)
    for i in range(scrim_height):
        alpha = int(190 * (1 - i / scrim_height))
        scrim_draw.line([(0, i), (w, i)], fill=(0, 0, 0, alpha))
    canvas.paste(scrim, (0, 0), scrim)

    caption_font = _font(52)
    lines = _wrap(caption.upper(), caption_font, w - 60, max_lines=3)
    y = 22
    for line in lines:
        _draw_outlined_text(draw, (30, y), line, caption_font, fill="white", outline_width=4)
        y += caption_font.size + 8

    # Reserve the bottom-right corner for the watermark, stacking the stat callout box (if any)
    # above it rather than on top of it -- the two collided before this was made explicit.
    watermark_font = _font(15)
    watermark_y = h - 10

    if stat_text:
        stat_font = _font(42)
        is_negative = stat_text.strip().startswith("-")
        box_color = RED if is_negative else GREEN
        padding = 14
        text_w = stat_font.getlength(stat_text)
        box_w, box_h = text_w + padding * 2, stat_font.size + padding * 2
        box_x, box_y = w - box_w - 24, h - box_h - 34
        draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=box_color)
        draw.text((box_x + padding, box_y + padding - 4), stat_text, font=stat_font, fill="white")
        watermark_y = box_y - 6

    if badge_text:
        badge_font = _font(18)
        badge_width = draw.textlength(badge_text, font=badge_font) + 24
        draw.rectangle([0, h - 32, badge_width, h], fill=badge_color)
        draw.text((12, h - 28), badge_text, font=badge_font, fill="white")

    _draw_outlined_text(
        draw, (w - 10, watermark_y), WATERMARK_HANDLE, watermark_font, fill="#dddddd",
        outline_width=2, anchor="rs",
    )

    return canvas.convert("RGB")


def generate_illustration(scene, caption, stat_text=None, badge_text=None, badge_color=(185, 28, 28), stats_out=None):
    """Shared entry point for both satire_image.py and news_image.py: fetches a free, keyless
    background scene from Pollinations.ai and composes a bold caption/stat/badge overlay on top
    with PIL. Returns PNG bytes, or None if the background fetch or composition fails -- callers
    treat a missing image the same as any other optional visual in this pipeline (see chart.py's
    resolve_visual None-return contract)."""
    try:
        background = fetch_background(scene)
    except Exception as exc:
        logger.warning("Illustration background fetch failed for scene '%s': %s", scene[:80], exc)
        return None

    try:
        final = compose_image(background, caption, stat_text, badge_text=badge_text, badge_color=badge_color)
    except Exception as exc:
        logger.warning("Illustration composition failed for caption '%s': %s", caption[:80], exc)
        return None

    if stats_out is not None:
        stats_out.update({"source": "pollinations+pil", "caption": caption, "stat": stat_text})

    buf = io.BytesIO()
    final.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
