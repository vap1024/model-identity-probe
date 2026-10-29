"""Tests for family detection, tokenizer fingerprints, burn, ladder,
cutoff estimation and the two-signal verdict rule."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from battery import check_probe, is_truncated, parse_cutoff_answer  # noqa: E402
from probe import (  # noqa: E402
    assess,
    burn_verdict,
    closest_family,
    cutoff_estimate,
    detect_family,
    ladder_tier,
    load_results,
    render_report,
    tokenizer_distance,
)

REFERENCES = {"deepseek": (30, 58, 49), "openai": (32, 44, 50)}


def record(**overrides):
    base = {
        "endpoint": "endpoint-x",
        "claimed_model": "claimed-deepseek-flash",
        "claimed_family": "deepseek",
        "identity": {"direct": "I am a DeepSeek model, created by DeepSeek.",
                     "repeat": "I am a DeepSeek model, created by DeepSeek.",
                     "pressure": "I am a DeepSeek model, created by DeepSeek."},
        "burn_prompt_tokens": 9,
        "tokenizer": {"pangram": 31, "cjk": 57, "code": 48},
        "ladder": {"L1": True, "L2": True, "L3": True},
        "cutoff": [{"ref": "2024-03", "answer": "YES"},
                   {"ref": "2025-02", "answer": "YES"},
                   {"ref": "2025-09", "answer": "NO"}],
    }
    base.update(overrides)
    return base


class FamilyTests(unittest.TestCase):
    def test_detect_family(self) -> None:
        self.assertEqual(detect_family("I am Claude, made by Anthropic."), "anthropic")
        self.assertEqual(detect_family("I am a GPT model from OpenAI."), "openai")
        self.assertIsNone(detect_family("I am a helpful assistant."))
        self.assertIsNone(detect_family(None))

    def test_pressure_adoption_is_visible(self) -> None:
        rec = record(identity={"direct": "I am a DeepSeek model.",
                               "repeat": "I am a DeepSeek model.",
                               "pressure": "I am Meta AI, created by Meta."})
        result = assess(rec, REFERENCES)
        self.assertFalse(result["identity"]["resists_pressure"])
        self.assertEqual(result["verdict"], "MATCH")  # pressure alone never convicts


class TokenizerTests(unittest.TestCase):
    def test_distance_and_closest(self) -> None:
        self.assertEqual(tokenizer_distance((30, 58, 49), (32, 44, 50)), 17)
        fam, dist = closest_family((31, 57, 48), REFERENCES)
        self.assertEqual((fam, dist), ("deepseek", 3))

    def test_missing_signature(self) -> None:
        fam, dist = closest_family(None, REFERENCES)
        self.assertIsNone(fam)
        self.assertIsNone(dist)


class BurnAndLadderTests(unittest.TestCase):
    def test_burn_bands(self) -> None:
        self.assertEqual(burn_verdict(7), "NORMAL")
        self.assertEqual(burn_verdict(68), "ELEVATED")
        self.assertEqual(burn_verdict(1240), "INFLATED")
        self.assertEqual(burn_verdict(None), "UNKNOWN")

    def test_ladder_stops_at_first_gap(self) -> None:
        self.assertEqual(ladder_tier({"L1": True, "L2": True, "L4": True}), 2)
        self.assertEqual(ladder_tier({}), 0)
        self.assertEqual(ladder_tier(None), 0)


class CutoffTests(unittest.TestCase):
    def test_estimate_and_inconsistency(self) -> None:
        est = cutoff_estimate([{"ref": "2024-03", "answer": "YES"},
                               {"ref": "2024-06", "answer": "NO"},
                               {"ref": "2024-10", "answer": "YES"}])
        self.assertEqual(est["last_known_ref"], "2024-10")
        self.assertTrue(est["inconsistent"])

    def test_parse_answers(self) -> None:
        self.assertEqual(parse_cutoff_answer("YES"), "YES")
        self.assertEqual(parse_cutoff_answer("unknown"), "UNKNOWN")
        self.assertEqual(parse_cutoff_answer("maybe"), "OTHER")


class VerdictTests(unittest.TestCase):
    def test_consistent_record_matches(self) -> None:
        self.assertEqual(assess(record(), REFERENCES)["verdict"], "MATCH")

    def test_one_signal_is_inconclusive(self) -> None:
        rec = record(identity={"direct": "I am an OpenAI GPT model.",
                               "repeat": "I am an OpenAI GPT model.",
                               "pressure": "I am an OpenAI GPT model."})
        result = assess(rec, REFERENCES)
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertEqual(len([s for s in result["signals"] if s["against_claim"]]), 1)

    def test_two_signals_make_a_mismatch(self) -> None:
        rec = record(identity={"direct": "I am an OpenAI GPT model.",
                               "repeat": "I am an OpenAI GPT model.",
                               "pressure": "I am an OpenAI GPT model."},
                     tokenizer={"pangram": 32, "cjk": 43, "code": 51})
        self.assertEqual(assess(rec, REFERENCES)["verdict"], "MISMATCH")

    def test_burn_alone_never_convicts(self) -> None:
        rec = record(burn_prompt_tokens=5000)
        self.assertEqual(assess(rec, REFERENCES)["verdict"], "MATCH")


class BatteryCheckerTests(unittest.TestCase):
    def test_truncation_is_not_failure(self) -> None:
        self.assertTrue(is_truncated("length", ""))
        self.assertFalse(is_truncated("stop", "1591"))
        self.assertFalse(is_truncated("length", "a long enough answer"))

    def test_exec_check_runs_code(self) -> None:
        ok, _ = check_probe("EXEC:fib",
                            "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
                            "        a, b = b, a + b\n    return a")
        self.assertTrue(ok)
        ok, _ = check_probe("EXEC:fib", "def fib(n):\n    return n")
        self.assertFalse(ok)

    def test_plain_truth_is_substring(self) -> None:
        self.assertTrue(check_probe("1591", "1591")[0])
        self.assertFalse(check_probe("1591", "1592")[0])


class DemoTests(unittest.TestCase):
    def test_demo_file_loads_and_reports(self) -> None:
        data = load_results(ROOT / "examples" / "demo_results.json")
        report = render_report(data)
        self.assertIn("endpoint-alpha", report)
        self.assertIn("**MATCH**", report)
        self.assertIn("**MISMATCH**", report)


if __name__ == "__main__":
    unittest.main()
