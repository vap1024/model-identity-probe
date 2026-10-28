"""Core of model-identity-probe: offline fingerprint analysis.

Standard library only. No network access. This module never calls a
model: it scores *recorded* probe answers that you collected yourself
(with any client you like) and turns them into signals and a verdict.

Why several signals: a model label behind a third-party gateway is a
claim, not a fact. Any single test can lie -- a system prompt can teach
a model a false name, a gateway can pad tokens, a reasoning model can
look broken simply because its answer was truncated. A substitution is
only claimed here when at least two *independent* signal families
disagree with the label at the same time.

Record schema (one JSON object per probed endpoint):

    {
      "endpoint": "endpoint-alpha",          # your own label, invented
      "claimed_model": "claimed-flash-1",    # the label being sold
      "claimed_family": "deepseek",          # family the label implies
      "identity": {
        "direct":   "I am ...",              # verbatim short answers
        "repeat":   "I am ...",
        "pressure": "I am ..."
      },
      "burn_prompt_tokens": 7,               # prompt_tokens for "Hi"
      "tokenizer": {"pangram": 32, "cjk": 41, "code": 49},
      "ladder": {"L1": true, "L2": true, ...},
      "cutoff": [{"ref": "2024-03", "answer": "YES"}, ...]
    }

Reference tokenizer signatures for known families, if you have them,
are passed in separately -- they are measurements, not part of a record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Public model-creator families a self-identification answer can name.
# These are product families, not accounts or providers.
FAMILY_KEYWORDS = {
    "openai": ("openai", "gpt-", "gpt ", "chatgpt"),
    "anthropic": ("anthropic", "claude"),
    "google": ("google", "gemini"),
    "meta": ("meta ai", "llama"),
    "xai": ("xai", "grok"),
    "deepseek": ("deepseek",),
    "mistral": ("mistral", "mixtral"),
    "alibaba": ("alibaba", "qwen"),
}

# A "Hi" prompt is a handful of tokens in every known tokenizer. Far
# above that, the gateway is injecting hidden context (padding) or the
# usage numbers themselves are not trustworthy. Heuristic bands only.
BURN_NORMAL_MAX = 20
BURN_ELEVATED_MAX = 100


def detect_family(text: str | None) -> str | None:
    """Family named in a free-text self-identification, or None."""
    if not text:
        return None
    low = text.lower()
    for family, keys in FAMILY_KEYWORDS.items():
        if any(k in low for k in keys):
            return family
    return None


def tokenizer_signature(record: dict) -> tuple[int, ...] | None:
    """(pangram, cjk, code) prompt-token counts, or None if incomplete."""
    tok = record.get("tokenizer") or {}
    try:
        return (int(tok["pangram"]), int(tok["cjk"]), int(tok["code"]))
    except (KeyError, TypeError, ValueError):
        return None


def tokenizer_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Total absolute distance between two tokenizer signatures."""
    return sum(abs(x - y) for x, y in zip(a, b))


def closest_family(signature, references: dict[str, tuple[int, ...]]):
    """(family, distance) of the closest reference signature."""
    if signature is None or not references:
        return None, None
    ranked = sorted(
        ((fam, tokenizer_distance(signature, ref)) for fam, ref in references.items()),
        key=lambda item: item[1],
    )
    return ranked[0]


def burn_verdict(prompt_tokens) -> str:
    """Classify the hidden-token burn of a minimal "Hi" call."""
    if prompt_tokens is None:
        return "UNKNOWN"
    if prompt_tokens <= BURN_NORMAL_MAX:
        return "NORMAL"
    if prompt_tokens <= BURN_ELEVATED_MAX:
        return "ELEVATED"
    return "INFLATED"


def ladder_tier(ladder: dict | None) -> int:
    """Highest consecutive passed level, counting from L1."""
    tier = 0
    for n in range(1, 7):
        if (ladder or {}).get(f"L{n}"):
            tier = n
        else:
            break
    return tier


def cutoff_estimate(cutoff: list[dict] | None) -> dict:
    """Estimate a knowledge cutoff from YES/NO answers to dated events.

    Events are chronological. The estimate is the latest event the model
    knows; a YES after a NO is flagged as inconsistent instead of being
    silently smoothed over.
    """
    known, inconsistent, last_yes = [], False, None
    seen_no = False
    for item in cutoff or []:
        ans = str(item.get("answer", "")).upper()
        if ans == "YES":
            if seen_no:
                inconsistent = True
            last_yes = item.get("ref")
            known.append(item.get("ref"))
        elif ans == "NO":
            seen_no = True
    return {"last_known_ref": last_yes, "known_count": len(known),
            "inconsistent": inconsistent}


def identity_signals(record: dict) -> dict:
    """What the identity block says, including under pressure.

    The pressure probe pre-loads a false identity in the system prompt.
    A model that adopts it has a weak (or overridden) identity; a model
    that keeps its own name scores one consistency point. Neither alone
    proves anything about the label.
    """
    ident = record.get("identity") or {}
    direct = detect_family(ident.get("direct"))
    repeat = detect_family(ident.get("repeat"))
    pressure = detect_family(ident.get("pressure"))
    return {
        "direct_family": direct,
        "consistent_repeat": bool(direct) and direct == repeat,
        "resists_pressure": bool(direct) and pressure == direct,
    }


def assess(record: dict, references: dict[str, tuple[int, ...]] | None = None) -> dict:
    """Combine all blocks into signals and a verdict for one record.

    Verdicts:
      MATCH       -- no independent signal contradicts the claimed family.
      MISMATCH    -- at least two independent signals contradict it.
      INCONCLUSIVE-- exactly one signal contradicts it, or data is too
                     thin to judge. One signal is a question, not a proof.
    """
    references = references or {}
    claimed = (record.get("claimed_family") or "").lower() or None
    signals: list[dict] = []

    ids = identity_signals(record)
    if ids["direct_family"] and claimed and ids["direct_family"] != claimed:
        signals.append({"signal": "identity_family",
                        "against_claim": True,
                        "detail": f"self-identifies as {ids['direct_family']}, "
                                  f"label claims {claimed}"})
    elif ids["direct_family"] and claimed:
        signals.append({"signal": "identity_family", "against_claim": False,
                        "detail": f"self-identifies as {ids['direct_family']}"})

    sig = tokenizer_signature(record)
    closest, dist = closest_family(sig, references)
    if closest and claimed and closest != claimed:
        signals.append({"signal": "tokenizer_family", "against_claim": True,
                        "detail": f"tokenizer signature is closest to {closest} "
                                  f"(distance {dist}), label claims {claimed}"})
    elif closest and claimed:
        signals.append({"signal": "tokenizer_family", "against_claim": False,
                        "detail": f"tokenizer signature is closest to {closest} "
                                  f"(distance {dist})"})

    burn = burn_verdict(record.get("burn_prompt_tokens"))
    if burn in ("ELEVATED", "INFLATED"):
        signals.append({"signal": "token_burn", "against_claim": False,
                        "detail": f"hidden-token burn is {burn} "
                                  f"({record.get('burn_prompt_tokens')} prompt tokens "
                                  f"for a minimal greeting) -- a gateway-padding "
                                  f"warning, not an identity signal"})

    against = [s for s in signals if s["against_claim"]]
    if claimed is None or not signals:
        verdict = "INCONCLUSIVE"
    elif len(against) >= 2:
        verdict = "MISMATCH"
    elif len(against) == 1:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "MATCH"

    return {
        "endpoint": record.get("endpoint", "?"),
        "claimed_model": record.get("claimed_model", "?"),
        "claimed_family": claimed,
        "verdict": verdict,
        "tier": ladder_tier(record.get("ladder")),
        "cutoff": cutoff_estimate(record.get("cutoff")),
        "burn": burn,
        "identity": ids,
        "signals": signals,
    }


def load_results(path: str | Path) -> dict:
    """Load a results file: {"references": {...}, "records": [...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "records" not in data:
        raise ValueError("results file must have a 'records' list")
    for rec in data["records"]:
        for field in ("endpoint", "claimed_model", "claimed_family"):
            if field not in rec:
                raise ValueError(f"record is missing field: {field}")
    return data


def parse_references(raw: dict | None) -> dict[str, tuple[int, ...]]:
    refs = {}
    for family, values in (raw or {}).items():
        refs[family.lower()] = tuple(int(v) for v in values)
    return refs


def render_report(data: dict) -> str:
    """Markdown summary of every record in a results file."""
    refs = parse_references(data.get("references"))
    lines = ["# Model identity probe -- report", "",
             "Synthetic or self-collected data only; verdicts need two "
             "independent signals (see README).", "",
             "| Endpoint | Claimed | Verdict | Tier | Cutoff (last known) | Burn |",
             "|---|---|---|---|---|---|"]
    details = []
    for rec in data["records"]:
        a = assess(rec, refs)
        cut = a["cutoff"]["last_known_ref"] or "unknown"
        if a["cutoff"]["inconsistent"]:
            cut += " (inconsistent)"
        lines.append(f"| {a['endpoint']} | {a['claimed_model']} | **{a['verdict']}** "
                     f"| {a['tier']}/6 | {cut} | {a['burn']} |")
        details.append(f"\n## {a['endpoint']} -- {a['claimed_model']}")
        for s in a["signals"]:
            mark = "against claim" if s["against_claim"] else "supports / neutral"
            details.append(f"- [{mark}] {s['signal']}: {s['detail']}")
        if not a["signals"]:
            details.append("- no usable signals in this record")
    return "\n".join(lines + details) + "\n"
