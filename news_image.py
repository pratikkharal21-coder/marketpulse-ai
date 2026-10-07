import logging

from illustration import generate_illustration, scene_names_real_individual

logger = logging.getLogger("marketpulse.news_image")

# Neutral, not alarming like the satire badge -- this is a real-news thread, just illustrated
# rather than charted. Still a mandatory, burned-in disclosure: these threads can get
# auto-posted to X via poster.py, and a photorealistic-looking AI scene of a real event needs a
# clear, conspicuous "this is illustrative, not a photo" label to stay on the right side of
# platform synthetic-media policies, same reasoning as the satire badge.
BADGE_TEXT = "AI ILLUSTRATION — NOT A REAL PHOTO"
BADGE_COLOR = (55, 65, 81)


def generate_news_illustration(scene, caption, stat_text=None, headline=None, stats_out=None):
    """Illustrated visual for a SERIOUS (non-satire) thread -- same free Pollinations.ai
    background art + PIL caption/stat overlay as satire_image.py, but with a neutral
    "AI ILLUSTRATION" disclosure badge instead of "SATIRE". `headline` (the real story's own
    title) is checked against `scene`: real news routinely centers a real named person (a Fed
    chair, a CEO, a head of state), which is completely normal for the THREAD TEXT, but the
    image prompt is separately instructed to depict only a generic stand-in -- if a real name
    from the headline leaks into the scene description anyway, that means the image generator
    would be attempting that person's actual likeness, which this pipeline does not do anywhere
    else either (see memeart.py's and satire_image.py's identical rule). Returns None (no image,
    thread text is unaffected) rather than risk it."""
    if scene_names_real_individual(scene, headline):
        logger.warning(
            "News illustration skipped: scene describes a real individual named in the "
            "headline '%s', despite the system prompt forbidding it", (headline or "")[:80],
        )
        return None

    return generate_illustration(
        scene, caption, stat_text, badge_text=BADGE_TEXT, badge_color=BADGE_COLOR, stats_out=stats_out,
    )
