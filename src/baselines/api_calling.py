import os
import json
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Tuple, List

load_dotenv()

DEFAULT_BASE = "https://api.moonshot.ai/v1"

# Pricing (based on Kimi K2.5 official docs – Moonshot)
IN_PRICE_PER_1M  = 0.60
OUT_PRICE_PER_1M = 3.00


def estimate_cost_usd(usage: dict) -> float:
    pt = float(usage.get("prompt_tokens", 0))
    ct = float(usage.get("completion_tokens", 0))
    return (pt / 1_000_000) * IN_PRICE_PER_1M + (ct / 1_000_000) * OUT_PRICE_PER_1M


def build_prompt(instance: dict, allow_empty: bool) -> str:
    # Phase 1: to avoid empty output, default allow_empty=False
    empty_rule = "0 to 3" if allow_empty else "1 to 3 (do NOT return empty list)"
    return (
        "Return ONLY valid JSON. No reasoning. No explanation. No extra keys.\n"
        f"You MUST return between {empty_rule} entrances.\n"
        "All entrances should lie ON the building footprint boundary if possible.\n"
        "Schema exactly:\n"
        "{\"entrances\":[{\"lat\":0.0,\"lon\":0.0,\"confidence\":0.0}]}\n"
        "Rules:\n"
        f"- entrances length: {empty_rule}\n"
        "- confidence in [0,1]\n"
        "Input footprint_wkt:\n"
        f"{instance['footprint_wkt']}\n"
    )


def call_moonshot(prompt: str, model: str, temperature: float, debug: bool = False) -> Tuple[str, dict, dict]:
    base = os.environ.get("MOONSHOT_API_BASE", DEFAULT_BASE)
    key = os.environ["MOONSHOT_API_KEY"]

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You must output ONLY JSON. No explanation. "
                    "Do NOT include markdown. Do NOT include extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    r = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()

    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    usage = data.get("usage", {})
    meta = {
        "id": data.get("id"),
        "model": data.get("model"),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "has_reasoning_content": "reasoning_content" in msg,
        "reasoning_len_chars": len(msg.get("reasoning_content", "")) if "reasoning_content" in msg else 0,
        "content_len_chars": len(content),
    }

    if debug:
        print("=== RAW RESPONSE (meta) ===")
        print({"id": meta["id"], "model": meta["model"], "usage": usage, "finish_reason": meta["finish_reason"]})
        print({"content_len_chars": meta["content_len_chars"], "has_reasoning_content": meta["has_reasoning_content"],
               "reasoning_len_chars": meta["reasoning_len_chars"]})

    return content, usage, meta


def parse_pred_entrances(content: str) -> List[Dict[str, float]]:
    obj = json.loads(content)
    entrances = obj.get("entrances", [])
    if not isinstance(entrances, list):
        return []

    cleaned = []
    for e in entrances:
        if not isinstance(e, dict):
            continue
        try:
            lat = float(e["lat"])
            lon = float(e["lon"])
            conf = float(e.get("confidence", 0.0))
        except Exception:
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        conf = max(0.0, min(1.0, conf))
        cleaned.append({"lat": lat, "lon": lon, "confidence": conf})
    return cleaned


def format_money(x: float) -> str:
    return f"${x:.6f}"


def print_compare_table(rows: List[Dict[str, Any]]) -> None:
    # Simple aligned text table (no extra deps)
    headers = ["model", "prompt_toks", "completion_toks", "total_toks", "est_cost", "entrances", "reasoning_chars"]
    colw = {h: len(h) for h in headers}
    for r in rows:
        colw["model"] = max(colw["model"], len(str(r["model"])))
        for h in headers[1:]:
            colw[h] = max(colw[h], len(str(r[h])))

    def line(vals):
        return "  ".join(str(vals[h]).ljust(colw[h]) for h in headers)

    print("\n=== Model Cost/Usage Comparison ===")
    print(line({h: h for h in headers}))
    print(line({h: "-" * colw[h] for h in headers}))
    for r in rows:
        print(line(r))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a single building JSON file")
    parser.add_argument("--out", default="outputs/preds_api.jsonl", help="Output JSONL path")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--retries", type=int, default=2)

    parser.add_argument("--model", default="kimi-k2.5", help="Model id (e.g., kimi-latest, moonshot-v1-auto)")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature (model may restrict it)")
    parser.add_argument("--allow_empty", action="store_true", help="Allow entrances=[] (default: forbid empty)")

    parser.add_argument(
        "--compare_models",
        default="",
        help='Comma-separated list of models to compare (e.g., "kimi-k2.5,kimi-latest,moonshot-v1-auto")',
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        instance = json.load(f)

    prompt = build_prompt(instance, allow_empty=args.allow_empty)

    # Compare mode: run multiple models and print a table (for pricing/usage comparison)
    if args.compare_models.strip():
        models = [m.strip() for m in args.compare_models.split(",") if m.strip()]
        rows = []
        for m in models:
            # temperature default: kimi-k2.5 requires 1; others often accept 0
            temp = args.temperature
            if temp is None:
                temp = 1.0 if m == "kimi-k2.5" else 0.0

            # Call with retries for JSON parse robustness
            last_content, last_usage, last_meta = "", {}, {}
            pred_entrances = []
            for attempt in range(args.retries):
                try:
                    content, usage, meta = call_moonshot(
                        prompt if attempt == 0 else (prompt + "\nREMINDER: JSON only."),
                        model=m,
                        temperature=temp,
                        debug=args.debug,
                    )
                    last_content, last_usage, last_meta = content, usage, meta
                    pred_entrances = parse_pred_entrances(content)
                    # If empty is forbidden but model returned empty, keep it as a signal (don't loop forever)
                    break
                except Exception:
                    continue

            cost = estimate_cost_usd(last_usage)
            rows.append({
                "model": m,
                "prompt_toks": last_usage.get("prompt_tokens", 0),
                "completion_toks": last_usage.get("completion_tokens", 0),
                "total_toks": last_usage.get("total_tokens", 0),
                "est_cost": format_money(cost),
                "entrances": len(pred_entrances),
                "reasoning_chars": last_meta.get("reasoning_len_chars", 0),
            })

        print_compare_table(rows)
        return

    # Single-model run (default)
    model = args.model
    temp = args.temperature
    if temp is None:
        temp = 1.0 if model == "kimi-k2.5" else 0.0

    last_content = ""
    last_usage = {}
    last_meta = {}
    pred_entrances = []

    for attempt in range(args.retries):
        content, usage, meta = call_moonshot(
            prompt if attempt == 0 else (prompt + "\nREMINDER: JSON only."),
            model=model,
            temperature=temp,
            debug=args.debug,
        )
        last_content, last_usage, last_meta = content, usage, meta
        try:
            pred_entrances = parse_pred_entrances(content)
            break
        except Exception:
            pred_entrances = []
            continue

    cost = estimate_cost_usd(last_usage)

    output = {
        "building_id": instance.get("building_id"),
        "model": model,
        "pred_entrances": pred_entrances,
        "usage": last_usage,
        "estimated_cost_usd": round(cost, 8),
        "meta": last_meta,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print("✅ Prediction saved to", args.out)
    print("Token usage:", last_usage)
    print("Estimated cost (USD):", round(cost, 6))

    if args.debug:
        print("\n=== MODEL CONTENT (raw) ===")
        print(last_content)


if __name__ == "__main__":
    main()