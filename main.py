from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

import agents
from agents import AgentRepairA11y
from gui_codegen import GUICodegenConfig, SingleScreenshotGUICodegen, build_run_directory
from run_audit import run_axe_audit, run_lighthouse, run_merge_reports, run_pa11y, start_static_server
from utils.log import init_logger, logger
from utils.utils import crop_image
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end UI codegen + accessibility audit + repair pipeline.")
    parser.add_argument("--image", "-i", required=True, help="Path to input screenshot")
    parser.add_argument("--out_dir", "-o", default="./output", help="Output directory")
    parser.add_argument("--backbone", "-b", default="openai", choices=["openai", "gemini"], help="Model backend")
    parser.add_argument("--max_blocks", type=int, default=25, help="Max blocks from blocker")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation/repair temperature")
    parser.add_argument(
        "--audit_tool",
        default="all",
        choices=["axe", "lighthouse", "pa11y", "all"],
        help="Accessibility audit tool: one tool or all three + merge",
    )
    parser.add_argument("--ruleset", default="wcag2a", choices=["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa", "best-practice"])
    parser.add_argument("--timeout_ms", type=int, default=60000)
    parser.add_argument(
        "--audit_merge_temperature",
        type=float,
        default=0.0,
        help="Temperature for merge agent when --audit_tool all",
    )
    return parser.parse_args()


def detect_report_type(report_json) -> str:
    if isinstance(report_json, list):
        return "pa11y"
    if isinstance(report_json, dict):
        if "violations" in report_json or "incomplete" in report_json:
            return "axe"
        if "lighthouseVersion" in report_json or "categories" in report_json or "audits" in report_json:
            return "lighthouse"
    return "unknown"


def audit_html_file(
    html_path: Path,
    out_dir: Path,
    tool: str,
    ruleset: str,
    timeout_ms: int,
    merge_backend: str,
    merge_temperature: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    httpd, port = start_static_server(html_path.parent)
    url = f"http://127.0.0.1:{port}/{html_path.name}"
    try:
        if tool == "axe":
            result = run_axe_audit(url, ruleset, timeout_ms, out_dir)
            report_path = out_dir / "axe_report.json"
        elif tool == "lighthouse":
            result = run_lighthouse(url, out_dir)
            report_path = out_dir / "lighthouse_report.json"
        elif tool == "pa11y":
            result = run_pa11y(url, out_dir)
            report_path = out_dir / "pa11y_report.json"
        else:
            axe_res = run_axe_audit(url, ruleset, timeout_ms, out_dir)
            lighthouse_res = run_lighthouse(url, out_dir)
            pa11y_res = run_pa11y(url, out_dir)
            merge_res = run_merge_reports(
                out_dir=out_dir,
                backend_name=merge_backend,
                temperature=merge_temperature,
                merged_filename="merged_accessibility_report.json",
            )

            # Prefer merged report for repair when available.
            if merge_res.ok and merge_res.report_path and merge_res.report_path.exists():
                result = merge_res
                report_path = merge_res.report_path
            elif axe_res.ok and (out_dir / "axe_report.json").exists():
                # Fallback: use axe report if merge fails.
                result = axe_res
                report_path = out_dir / "axe_report.json"
            elif lighthouse_res.ok and (out_dir / "lighthouse_report.json").exists():
                result = lighthouse_res
                report_path = out_dir / "lighthouse_report.json"
            else:
                result = pa11y_res
                report_path = out_dir / "pa11y_report.json"
    finally:
        httpd.shutdown()

    return result, report_path


def repair_html_with_report(
    repair_agent: AgentRepairA11y,
    html_path: Path,
    report_path: Path,
    out_html: Path,
    out_analysis: Path,
    temperature: float,
):
    html_text = html_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    report_json = json.loads(report_text)

    parsed = repair_agent.infer(html_text, report_text, parse=True, temperature=temperature)
    repaired_html = parsed.get("repaired_html") if isinstance(parsed, dict) else None

    if not repaired_html:
        out_analysis.parent.mkdir(parents=True, exist_ok=True)
        out_analysis.write_text(
            json.dumps(
                {
                    "error": "missing repaired_html",
                    "report_type": detect_report_type(report_json),
                    "parsed_output": parsed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return False

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_analysis.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(repaired_html, encoding="utf-8")

    out_analysis.write_text(
        json.dumps(
            {
                "input_html": str(html_path),
                "input_report": str(report_path),
                "report_type": detect_report_type(report_json),
                "output_html": str(out_html),
                "analysis": parsed.get("analysis", {}) if isinstance(parsed, dict) else {},
                "changes": parsed.get("changes", []) if isinstance(parsed, dict) else [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return True


def main() -> None:
    args = parse_args()

    image = Image.open(args.image).convert("RGB")
    output_root = Path(args.out_dir)
    run_dir = build_run_directory(output_root, image)
    run_dir.mkdir(parents=True, exist_ok=True)

    init_logger(logger, logfile=str(run_dir / "log.txt"))
    logger.info("Args: %s", args)

    generator = SingleScreenshotGUICodegen(
        GUICodegenConfig(
            backbone=args.backbone,
            max_blocks_limit=args.max_blocks,
            sample_temperature=args.temperature,
        )
    )
    # BACKBONE is configured inside SingleScreenshotGUICodegen.
    repair_agent = AgentRepairA11y()

    logger.info("Step 1/6: Split screenshot into modules")
    plans, blocks_preview = generator.split_modules(image)
    (run_dir / "modules").mkdir(exist_ok=True, parents=True)
    blocks_preview.save(str(run_dir / "blocks.png"))

    logger.info("Step 2/6: Generate module codes")
    module_codes = generator.generate_module_codes(image, plans)
    (run_dir / "modules_original.json").write_text(
        json.dumps(module_codes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    repaired_modules = []

    logger.info("Step 3/6: Audit + repair each module")
    for idx, module in enumerate(module_codes, start=1):
        time.sleep(60)
        module_dir = run_dir / "modules" / f"module_{idx:02d}"
        module_dir.mkdir(parents=True, exist_ok=True)

        module_img = crop_image(image, module["module_position"])
        module_img_path = module_dir / "module_input.png"
        module_html_path = module_dir / "module_original.html"
        module_repaired_html_path = module_dir / "module_repaired.html"
        module_analysis_path = module_dir / "repair_analysis.json"

        module_img.save(str(module_img_path))
        module_html_path.write_text(module["module_code"], encoding="utf-8")

        audit_result, report_path = audit_html_file(
            html_path=module_html_path,
            out_dir=module_dir / "audit_before",
            tool=args.audit_tool,
            ruleset=args.ruleset,
            timeout_ms=args.timeout_ms,
            merge_backend=args.backbone,
            merge_temperature=args.audit_merge_temperature,
        )

        repaired_code = module["module_code"]
        if audit_result.ok and report_path.exists():
            try:
                ok = repair_html_with_report(
                    repair_agent,
                    html_path=module_html_path,
                    report_path=report_path,
                    out_html=module_repaired_html_path,
                    out_analysis=module_analysis_path,
                    temperature=args.temperature,
                )
                if ok:
                    repaired_code = module_repaired_html_path.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.error("Module %s repair failed: %s", idx, exc)

        repaired_modules.append(
            {
                "module_position": module["module_position"],
                "module_code": repaired_code,
            }
        )

    (run_dir / "modules_repaired.json").write_text(
        json.dumps(repaired_modules, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("Step 4/6: Assemble repaired modules")
    final_html_before, final_img_before = generator.assemble_agent(image, repaired_modules)
    final_before_html_path = run_dir / "prediction_before_final_repair.html"
    final_before_img_path = run_dir / "prediction_before_final_repair.png"
    final_before_html_path.write_text(final_html_before, encoding="utf-8")
    final_img_before.save(str(final_before_img_path))

    logger.info("Step 5/6: Final audit")
    final_audit_result, final_report_path = audit_html_file(
        html_path=final_before_html_path,
        out_dir=run_dir / "final_audit_before",
        tool=args.audit_tool,
        ruleset=args.ruleset,
        timeout_ms=args.timeout_ms,
        merge_backend=args.backbone,
        merge_temperature=args.audit_merge_temperature,
    )

    logger.info("Step 6/6: Final repair")
    final_html_path = run_dir / "prediction.html"
    final_analysis_path = run_dir / "final_repair_analysis.json"

    time.sleep(90)
    if final_audit_result.ok and final_report_path.exists():
        repaired_ok = repair_html_with_report(
            repair_agent,
            html_path=final_before_html_path,
            report_path=final_report_path,
            out_html=final_html_path,
            out_analysis=final_analysis_path,
            temperature=args.temperature,
        )
        if not repaired_ok:
            final_html_path.write_text(final_html_before, encoding="utf-8")
    else:
        final_html_path.write_text(final_html_before, encoding="utf-8")

    # Render final HTML for output parity.
    from utils.html2shot_sync import html2shot

    final_img = html2shot(final_html_path.read_text(encoding="utf-8"))
    final_img.save(str(run_dir / "prediction.png"))

    # Optional post-repair audit snapshot.
    audit_html_file(
        html_path=final_html_path,
        out_dir=run_dir / "final_audit_after",
        tool=args.audit_tool,
        ruleset=args.ruleset,
        timeout_ms=args.timeout_ms,
        merge_backend=args.backbone,
        merge_temperature=args.audit_merge_temperature,
    )

    image.save(str(run_dir / "input.png"))

    logger.info("Done. prediction_image=%s prediction_html=%s", run_dir / "prediction.png", final_html_path)
    print(f"Prediction image: {run_dir / 'prediction.png'}")
    print(f"Prediction html: {final_html_path}")


if __name__ == "__main__":
    main()
