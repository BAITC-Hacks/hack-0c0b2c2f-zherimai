"""Evaluate a small reviewed query set; online mode must be enabled explicitly."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aml.ai_assistant import AIAnalystAssistant, GID_PATTERN


def evaluate_case(case, result):
    failures = []
    expected_mode = case.get("expected_mode", "online_semantic_plan_local_execution")
    if result.get("mode") != expected_mode:
        failures.append("mode")
    if case.get("expected_fallback_reason") != result.get("fallback_reason"):
        failures.append("fallback_reason")
    if case.get("expected_local_tool") and result.get("tool") != case["expected_local_tool"]:
        failures.append("local_tool")
    if case.get("expect_api_call") is False and (result.get("quota") or result.get("api_usage")):
        failures.append("unexpected_api_reservation")
    if case.get("expect_api_call") is True and not result.get("quota"):
        failures.append("missing_api_reservation")
    plan = result.get("plan") or {}
    if plan.get("action") != case["expected_action"]:
        failures.append("action")
    gids = list(dict.fromkeys(GID_PATTERN.findall(case["question"])))
    aliases = {f"node_{i}": gid for i, gid in enumerate(gids, 1)}
    actual_gids = [aliases.get(alias) for alias in plan.get("node_refs", [])]
    expected_gids = case.get("expected_gids_order", [])
    if plan.get("action") == "paths":
        if actual_gids != expected_gids:
            failures.append("ordered_gids")
    elif set(actual_gids) != set(expected_gids):
        failures.append("gids")
    if plan.get("cluster_id") != case.get("expected_cluster_id"):
        failures.append("cluster_id")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("docs/ai-eval-cases.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--report", type=Path, default=Path("out-check/ai-eval.json"))
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = spec["cases"] if isinstance(spec, dict) else spec
    if not 1 <= len(cases) <= 20:
        parser.error("Expected 1..20 reviewed cases")
    if args.online and not (os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL")):
        parser.error("Set OPENAI_API_KEY and OPENAI_MODEL locally before --online")
    assistant = AIAnalystAssistant(args.data, args.out)
    rows, tokens = [], Counter()
    for case in cases:
        started = time.perf_counter()
        result = assistant.answer(case["question"], online=args.online)
        failures = evaluate_case(case, result)
        usage = result.get("api_usage") or {}
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            tokens[name] += usage.get(name, 0)
        tokens["cached_input_tokens"] += usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        rows.append({"id": case["id"], "passed": not failures, "failures": failures,
                     "elapsed_seconds": round(time.perf_counter()-started, 3),
                     "mode": result.get("mode"), "tool": result.get("tool"),
                     "plan": result.get("plan"), "fallback_reason": result.get("fallback_reason"),
                     "api_usage": result.get("api_usage"), "quota": result.get("quota")})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "model": os.environ.get("OPENAI_MODEL") if args.online else None,
              "online_enabled": args.online, "n_cases": len(rows),
              "n_passed": sum(row["passed"] for row in rows), "reported_tokens": dict(tokens),
              "limits": ["Small hand-reviewed development set, not a held-out benchmark",
                         "Tests routing and arguments, not AML role accuracy",
                         "Token totals include reported usage only, not all account/Codex costs",
                         "No automatic retries; offline run does not measure model quality"], "results": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"passed": report["n_passed"], "cases": len(rows), "tokens": dict(tokens)}))
    return 0 if report["n_passed"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
