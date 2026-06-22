import json
import logging
import re

from groq import Groq

import config

logger = logging.getLogger("marketpulse.ai")

client = Groq(api_key=config.GROQ_API_KEY)

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def call_for_json(model, system, user_content, max_tokens=2048):
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system + "\n\nRespond with a JSON object."},
            {"role": "user", "content": user_content},
        ],
    )
    text = response.choices[0].message.content
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"No JSON found in model response: {text[:300]}")
    return json.loads(match.group(0))
