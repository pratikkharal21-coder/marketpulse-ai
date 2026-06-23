from datetime import datetime, timezone

from jinja2 import Environment, BaseLoader

TEMPLATE = """
<html>
<body style="font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 640px; margin: 0 auto; color: #1a1a1a;">
  <h2 style="margin-bottom: 4px;">MarketPulse AI — {{ run_time }}</h2>
  <p style="color: #666; margin-top: 0;">
    {{ threads|length }} thread{{ 's' if threads|length != 1 else '' }} +
    {{ deep_dives|length }} deep dive{{ 's' if deep_dives|length != 1 else '' }}
    from {{ story_count }} stories.
  </p>

  {% macro thread_card(item, accent) %}
  <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 14px; margin-bottom: 14px;">
    <div style="font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.05em;">
      {{ item.story_source }} · {{ item.thread|length }} tweets
    </div>
    {% if item.cid %}
    <img src="cid:{{ item.cid }}" alt="chart" style="width: 100%; max-width: 560px; border-radius: 6px; margin: 10px 0;">
    {% endif %}
    {% for tweet in item.thread %}
    <p style="font-size: 15px; line-height: 1.4; margin: 10px 0; padding-left: 10px; border-left: 3px solid {{ accent }};">
      {{ tweet }}
    </p>
    {% endfor %}
    <div style="font-size: 12px; color: #555; margin-top: 8px;">
      Relevance {{ item.relevance }}/10 · Engagement {{ item.expected_engagement }}/10 ·
      Significance {{ item.market_significance }}/10 · Confidence {{ item.confidence }}/10
    </div>
    <div style="font-size: 12px; margin-top: 6px;">
      <a href="{{ item.story_link }}" style="color: #2563eb;">{{ item.story_title }}</a>
    </div>
  </div>
  {% endmacro %}

  <h3 style="margin-top: 20px; margin-bottom: 8px;">Today's Threads</h3>
  {% for thread in threads %}{{ thread_card(thread, '#2563eb') }}{% endfor %}
  {% if not threads %}
  <p>No high-impact stories cleared the threshold this run.</p>
  {% endif %}

  {% if deep_dives %}
  <h3 style="margin-top: 24px; margin-bottom: 8px;">Deep Dive Thread{{ 's' if deep_dives|length != 1 else '' }}</h3>
  {% for deep_dive in deep_dives %}{{ thread_card(deep_dive, '#b45309') }}{% endfor %}
  {% endif %}
</body>
</html>
"""

_env = Environment(loader=BaseLoader())
_template = _env.from_string(TEMPLATE)


def _assign_cids(items, prefix):
    inline_images = {}
    for i, item in enumerate(items):
        chart_image = item.get("chart_image")
        if chart_image:
            cid = f"{prefix}{i}"
            item["cid"] = cid
            inline_images[cid] = chart_image
        else:
            item["cid"] = None
    return inline_images


def render(threads, story_count, deep_dives):
    inline_images = {}
    inline_images.update(_assign_cids(threads, "thread"))
    inline_images.update(_assign_cids(deep_dives, "deepdive"))

    html = _template.render(
        run_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        story_count=story_count,
        threads=threads,
        deep_dives=deep_dives,
    )
    return html, inline_images
