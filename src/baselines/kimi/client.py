import json
import os
from typing import Any, Dict, Tuple

import requests

DEFAULT_BASE = "https://api.moonshot.ai/v1"
IN_PRICE_PER_1M = 0.60
OUT_PRICE_PER_1M = 3.00


def estimate_cost_usd(usage: dict) -> float:
    pt = float(usage.get("prompt_tokens", 0))
    ct = float(usage.get("completion_tokens", 0))
    return (pt / 1_000_000) * IN_PRICE_PER_1M + (ct / 1_000_000) * OUT_PRICE_PER_1M


def call_kimi_api(
    prompt: str, model: str = "kimi-k2.5", temperature: float = 1.0
) -> Tuple[Dict, Dict]:
    """
    Calls the Moonshot/Kimi API and returns the parsed JSON content and usage.
    """
    base = os.environ.get("MOONSHOT_API_BASE", DEFAULT_BASE)
    key = os.environ.get("MOONSHOT_API_KEY")

    if not key:
        raise ValueError("MOONSHOT_API_KEY not found in environment variables.")

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": "You must output ONLY JSON. No explanation. Do NOT include markdown.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"].get("content", "{}")
    usage = data.get("usage", {})

    # Simple JSON cleaning in case of markdown blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content), usage
