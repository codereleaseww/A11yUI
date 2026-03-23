from __future__ import annotations

import argparse
import json
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import agents
from agents import AgentMergeA11yReports
from playwright.sync_api import sync_playwright


AXE_CDN_URLS = [
    "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js",
    "https://unpkg.com/axe-core@4.10.2/axe.min.js",
]


@dataclass
class ToolResult:
    name: str
    ok: bool
    message: str
    report_path: Path | None = None
    summary_path: Path | None = None


def log(message: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run axe-core, Lighthouse, and Pa11y audits for an HTML file.")
    parser.add_argument("--html", "-i", type=str, default="output/prediction.html", help="Path to HTML file")
    parser.add_argument("--out-dir", "-o", type=str, default="output", help="Directory for reports")
    parser.add_argument(
        "--ruleset",
        type=str,
        default="wcag2a",
        choices=["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa", "best-practice"],
        help="WCAG guideline ruleset",
    )
    parser.add_argument("--timeout-ms", type=int, default=60000, help="Playwright timeout in ms")
    parser.add_argument(
        "--tool",
        type=str,
        default="axe",
        choices=["axe", "lighthouse", "pa11y", "all"],
        help="accessibility tool to run",
    )
    parser.add_argument(
        "--merge-backend",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="LLM backend used only when --tool all (report merge step)",
    )
    parser.add_argument(
        "--merge-temperature",
        type=float,
        default=0.0,
        help="LLM temperature used only when --tool all",
    )
    parser.add_argument(
        "--merged-out",
        type=str,
        default="merged_accessibility_report.json",
        help="Merged report filename under --out-dir when --tool all",
    )
    return parser


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args):  # noqa: A003
        return


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_static_server(root_dir: Path):
    port = _pick_free_port()

    def factory(*args, **kwargs):
        return QuietHandler(*args, directory=str(root_dir), **kwargs)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), factory)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _inject_axe(page) -> None:
    last_error = None
    for url in AXE_CDN_URLS:
        try:
            page.add_script_tag(url=url)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Failed to load axe-core from CDN URLs: {last_error}")


def run_axe_audit(url: str, ruleset: str, timeout_ms: int, out_dir: Path) -> ToolResult:
    report_path = out_dir / "axe_report.json"
    summary_path = out_dir / "axe_summary.txt"

    try:
        log("axe-core: launching browser and loading page...")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(url, wait_until="networkidle")
            log("axe-core: page loaded, injecting axe script...")
            _inject_axe(page)
            log(f"axe-core: running audit with ruleset '{ruleset}'...")

            results = page.evaluate(
                """
                async ({ ruleset }) => {
                    return await axe.run(document, {
                        runOnly: {
                            type: 'tag',
                            values: [ruleset]
                        }
                    });
                }
                """,
                {"ruleset": ruleset},
            )
            browser.close()

        filtered_results = {
            "violations": results.get("violations", []),
            "incomplete": results.get("incomplete", []),
        }

        report_path.write_text(json.dumps(filtered_results, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_text = summarize_axe(filtered_results)
        summary_path.write_text(summary_text, encoding="utf-8")
        log(f"axe-core: completed. report={report_path} summary={summary_path}")
        return ToolResult("axe-core", True, "completed", report_path, summary_path)
    except Exception as exc:  # noqa: BLE001
        log(f"axe-core: failed: {exc}")
        return ToolResult("axe-core", False, f"failed: {exc}")


def summarize_axe(results: dict) -> str:
    violations = results.get("violations", [])
    incomplete = results.get("incomplete", [])

    lines = [
        f"Violations: {len(violations)}",
        f"Incomplete: {len(incomplete)}",
        "",
    ]

    if not violations:
        lines.append("No accessibility violations found by axe-core.")
        return "\n".join(lines)

    lines.append("Top violations:")
    for idx, item in enumerate(violations, start=1):
        impact = item.get("impact") or "unknown"
        node_count = len(item.get("nodes", []))
        lines.append(f"{idx}. {item.get('id')} | impact={impact} | nodes={node_count}")
        lines.append(f"   {item.get('help')}")
        lines.append(f"   {item.get('helpUrl')}")

    return "\n".join(lines)


def run_lighthouse(url: str, out_dir: Path) -> ToolResult:
    report_path = out_dir / "lighthouse_report.json"
    summary_path = out_dir / "lighthouse_summary.txt"

    cmd = [
        "npx",
        "--yes",
        "lighthouse",
        url,
        "--quiet",
        "--output=json",
        f"--output-path={report_path}",
        "--chrome-flags=--headless=new --no-sandbox",
        "--only-categories=accessibility",
    ]

    try:
        log("lighthouse: starting npx lighthouse...")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        log("lighthouse: npx not found.")
        return ToolResult("lighthouse", False, "npx not found. Install Node.js and npm.")

    if proc.returncode != 0 or not report_path.exists():
        msg = proc.stderr.strip() or proc.stdout.strip() or "lighthouse failed"
        log(f"lighthouse: failed: {msg}")
        return ToolResult("lighthouse", False, msg)

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))

        # Keep only accessibility-relevant audits.
        audits = data.get("audits", {})
        if isinstance(audits, dict):
            filtered_audits = {
                k: v
                for k, v in audits.items()
                if isinstance(v, dict)
                and v.get("scoreDisplayMode") != "notApplicable"
                and v.get("score") != 1
                and v.get("score") is not None
            }
            data["audits"] = filtered_audits
        report_path.write_text(json.dumps(data["audits"], ensure_ascii=False, indent=2), encoding="utf-8")

        score = data.get("categories", {}).get("accessibility", {}).get("score")
        if isinstance(score, (int, float)):
            pct = round(float(score) * 100, 2)
            summary = f"Accessibility score: {pct}/100"
        else:
            summary = "Accessibility score unavailable in report."
        summary_path.write_text(summary + "\n", encoding="utf-8")
        log(f"lighthouse: completed. score summary written to {summary_path}")
    except Exception as exc:  # noqa: BLE001
        summary = f"Report generated, but summary parse failed: {exc}"
        summary_path.write_text(summary + "\n", encoding="utf-8")
        log(f"lighthouse: report generated but summary parse failed: {exc}")

    return ToolResult("lighthouse", True, "completed", report_path, summary_path)


def run_pa11y(url: str, out_dir: Path) -> ToolResult:
    report_path = out_dir / "pa11y_report.json"
    summary_path = out_dir / "pa11y_summary.txt"

    cmd = ["npx", "--yes", "pa11y", url, "--reporter", "json"]

    try:
        log("pa11y: starting npx pa11y...")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        log("pa11y: npx not found.")
        return ToolResult("pa11y", False, "npx not found. Install Node.js and npm.")

    if proc.returncode not in (0, 2):  # pa11y may return 2 when issues found
        msg = proc.stderr.strip() or proc.stdout.strip() or "pa11y failed"
        log(f"pa11y: failed: {msg}")
        return ToolResult("pa11y", False, msg)

    stdout = proc.stdout.strip() or "[]"
    try:
        issues = json.loads(stdout)
    except json.JSONDecodeError:
        log("pa11y: failed to parse JSON output.")
        return ToolResult("pa11y", False, f"Unexpected pa11y output: {stdout[:300]}")

    report_path.write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = f"Issues: {len(issues)}"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    log(f"pa11y: completed. report={report_path} summary={summary_path}")
    return ToolResult("pa11y", True, "completed", report_path, summary_path)


def _load_optional_json(path: Path):
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _backend_from_name(name: str):
    if name == "openai":
        from openai_client import gpt

        return gpt
    if name == "gemini":
        from gemini_client import gemini

        return gemini
    raise ValueError(f"Unsupported backend: {name}")


def run_merge_reports(out_dir: Path, backend_name: str, temperature: float, merged_filename: str) -> ToolResult:
    report_path = out_dir / merged_filename
    summary_path = out_dir / "merged_accessibility_summary.txt"

    axe_report = _load_optional_json(out_dir / "axe_report.json")
    lighthouse_report = _load_optional_json(out_dir / "lighthouse_report.json")
    pa11y_report = _load_optional_json(out_dir / "pa11y_report.json")

    if axe_report is None and lighthouse_report is None and pa11y_report is None:
        return ToolResult("merge", False, "No valid input reports found to merge.")

    payload = {
        "axe_report": axe_report,
        "lighthouse_report": lighthouse_report,
        "pa11y_report": pa11y_report,
    }

    try:
        agents.BACKBONE = _backend_from_name(backend_name)
        agent = AgentMergeA11yReports()
        log(f"merge: running AgentMergeA11yReports with backend={backend_name}...")
        merged = agent.infer([json.dumps(payload, ensure_ascii=False)], parse=True, temperature=temperature)
    except Exception as exc:  # noqa: BLE001
        return ToolResult("merge", False, f"merge failed: {exc}")

    if not isinstance(merged, dict) or not merged:
        return ToolResult("merge", False, "merge failed: empty/invalid agent output")

    report_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    total = len(merged.get("merged_issues", [])) if isinstance(merged.get("merged_issues", []), list) else 0
    summary = f"Merged issues: {total}\\nOutput: {report_path}\\n"
    summary_path.write_text(summary, encoding="utf-8")
    log(f"merge: completed. report={report_path} summary={summary_path}")
    return ToolResult("merge", True, "completed", report_path, summary_path)


def main() -> int:
    args = build_parser().parse_args()

    html_path = Path(args.html).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not html_path.exists() or not html_path.is_file():
        print(f"HTML file not found: {html_path}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Input HTML: {html_path}")
    log(f"Output directory: {out_dir}")

    server_root = html_path.parent
    log(f"Starting local static server from: {server_root}")
    httpd, port = start_static_server(server_root)
    url = f"http://127.0.0.1:{port}/{html_path.name}"
    log(f"Local URL: {url}")

    results = []
    if args.tool == "axe":
        results.append(run_axe_audit(url, args.ruleset, args.timeout_ms, out_dir))
    elif args.tool == "lighthouse":
        results.append(run_lighthouse(url, out_dir))
    elif args.tool == "pa11y":
        results.append(run_pa11y(url, out_dir))
    elif args.tool == "all":
        results.append(run_axe_audit(url, args.ruleset, args.timeout_ms, out_dir))
        results.append(run_lighthouse(url, out_dir))
        results.append(run_pa11y(url, out_dir))
        results.append(run_merge_reports(out_dir, args.merge_backend, args.merge_temperature, args.merged_out))

    log("Stopping local static server...")
    httpd.shutdown()

    print(f"Audited URL: {url}")
    for result in results:
        status = "OK" if result.ok else "FAILED"
        print(f"- {result.name}: {status} ({result.message})")

    return 0 


if __name__ == "__main__":
    raise SystemExit(main())
