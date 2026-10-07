import logging

from illustration import generate_illustration, scene_names_real_individual

logger = logging.getLogger("marketpulse.satire_image")

BADGE_TEXT = "SATIRE — NOT REAL NEWS"
BADGE_COLOR = (185, 28, 28)


def generate_satire_image(scene, caption, stat_text=None, headline=None, stats_out=None):
    """Satire-flavored wrapper around illustration.generate_illustration(): always burns in a
    "SATIRE -- NOT REAL NEWS" badge, since this image could be screenshotted/shared out of the
    context of this email. `headline` (the source satire story's own title) is checked against
    `scene` as a hard backstop -- satire.py's own name check only covers the generated THREAD
    text, but `scene` is a separate model output that could independently describe a real named
    person even when the thread doesn't name them. Returns None (no image) rather than risk it."""
    if scene_names_real_individual(scene, headline):
        logger.warning(
            "Satire image skipped: scene describes a real individual named in the headline "
            "'%s', despite the system prompt forbidding it", (headline or "")[:80],
        )
        return None

    return generate_illustration(
        scene, caption, stat_text, badge_text=BADGE_TEXT, badge_color=BADGE_COLOR, stats_out=stats_out,
    )
