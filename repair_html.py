from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import agents
from agents import AgentRepairA11y
from gemini_client import gemini
from openai_client import gpt


BACKENDS: dict[str, Callable] = {
    "openai": gpt,
    "gemini": gemini,
}


def log(message: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair HTML accessibility issues using one report JSON.")
    parser.add_argument("--html", "-i", default="output/prediction.html", help="Input HTML path")
    parser.add_argument(
        "--report",
        "-r",
        default="output/axe_report.json",
        help="Path to one accessibility report JSON (axe/lighthouse/pa11y)",
    )
    parser.add_argument("--out-html", default="output/prediction_fixed.html", help="Output repaired HTML path")
    parser.add_argument(
        "--out-analysis",
        default="output/axe_repair_analysis.json",
        help="Output analysis JSON path",
    )
    parser.add_argument(
        "--backend",
        "-b",
        default="openai",
        choices=sorted(BACKENDS.keys()),
        help="LLM backend",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    return parser


def _load_optional_json(path: Path):
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def detect_report_type(report_json) -> str:
    if isinstance(report_json, list):
        return "pa11y"

    if isinstance(report_json, dict):
        if "violations" in report_json or "incomplete" in report_json:
            return "axe"
        if "lighthouseVersion" in report_json or "categories" in report_json or "audits" in report_json:
            return "lighthouse"

    return "unknown"


def main() -> int:
    args = build_parser().parse_args()

    html_path = Path(args.html).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    out_html = Path(args.out_html).expanduser().resolve()
    out_analysis = Path(args.out_analysis).expanduser().resolve()

    log(f"Input HTML: {html_path}")
    log(f"Input report: {report_path}")

    if not html_path.exists() or not html_path.is_file():
        print(f"Input HTML not found: {html_path}")
        return 1
    if not report_path.exists() or not report_path.is_file():
        print(f"Report not found: {report_path}")
        return 1

    log("Loading input files...")
    html_text = html_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    report_json = _load_optional_json(report_path)
    if report_json is None:
        print(f"Report JSON is invalid: {report_path}")
        return 1

    report_type = detect_report_type(report_json)
    log(f"Detected report type: {report_type}")

    log(f"Selecting backend: {args.backend}")
    agents.BACKBONE = BACKENDS[args.backend]
    repair_agent = AgentRepairA11y()

    log("Running accessibility repair agent...")
    try:
        parsed = repair_agent.infer(html_text, report_text, parse=True, temperature=args.temperature)
    except Exception as exc:  # noqa: BLE001
        print(f"Agent failed: {exc}")
        return 1

    repaired_html = parsed.get("repaired_html") if isinstance(parsed, dict) else None
    if not repaired_html or not isinstance(repaired_html, str):
        log("Agent output missing repaired_html. Writing debug analysis file.")
        out_analysis.parent.mkdir(parents=True, exist_ok=True)
        out_analysis.write_text(
            json.dumps(
                {
                    "error": "missing repaired_html",
                    "backend": args.backend,
                    "parsed_output": parsed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 1

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_analysis.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing repaired HTML: {out_html}")
    out_html.write_text(repaired_html, encoding="utf-8")

    analysis_obj = {
        "backend": args.backend,
        "input_html": str(html_path),
        "report_type": report_type,
        "input_report": str(report_path),
        "output_html": str(out_html),
        "analysis": parsed.get("analysis", {}) if isinstance(parsed, dict) else {},
        "changes": parsed.get("changes", []) if isinstance(parsed, dict) else [],
    }

    log(f"Writing analysis JSON: {out_analysis}")
    out_analysis.write_text(json.dumps(analysis_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    log("Accessibility repair completed.")
    print(f"Repaired HTML: {out_html}")
    print(f"Analysis: {out_analysis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
