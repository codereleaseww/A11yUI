from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup as bs
from PIL import Image

import agents
from agents import AgentAssemble, AgentGenerate
from utils.html2shot_sync import html2shot
from smart_blocker import blocker
from utils.log import logger
from utils.utils import crop_image, remove_code_markers


def _openai_backend(prompt, texts_imgs=None, temperature=0.0, seed=0, n=1):
    from openai_client import gpt

    return gpt(prompt, texts_imgs, temperature, seed, n)


def _gemini_backend(prompt, texts_imgs=None, temperature=0.0, seed=0, n=1):
    from gemini_client import gemini

    return gemini(prompt, texts_imgs, temperature, seed, n)


BACKBONES = {"openai": _openai_backend, "gemini": _gemini_backend}


@dataclass
class GUICodegenConfig:
    backbone: str = "openai"
    max_blocks_limit: int = 25
    sample_temperature: float = 0.0


class SingleScreenshotGUICodegen:
    def __init__(self, config: GUICodegenConfig):
        if config.backbone not in BACKBONES:
            raise ValueError(f"Unsupported backbone '{config.backbone}'. Supported: {list(BACKBONES.keys())}")

        self.config = config
        agents.BACKBONE = BACKBONES[config.backbone]

        self.agent_generate = AgentGenerate()
        self.agent_assemble = AgentAssemble()

    def split_modules(self, image: Image.Image) -> tuple[list[list[float]], Image.Image]:
        plans, blocks_preview = blocker(image)
        if len(plans) > self.config.max_blocks_limit:
            raise ValueError(f"Too many blocks detected: {len(plans)} > {self.config.max_blocks_limit}")
        return plans, blocks_preview

    def generate_module_codes(self, image: Image.Image, plans: list[list[float]]) -> list[dict[str, Any]]:
        module_codes: list[dict[str, Any]] = []
        generator = self.agent_generate

        for idx, plan in enumerate(plans, start=1):
            module_image = crop_image(image, plan)
            try:
                raw_code = generator.infer([module_image], n=1, temperature=self.config.sample_temperature)
                module_code = remove_code_markers(raw_code)
                module_codes.append({"module_position": plan, "module_code": module_code})
            except Exception as exc:
                logger.warning("Skip block %s because generation failed: %s", idx, exc)

        if not module_codes:
            raise RuntimeError("No module code was generated.")

        return module_codes

    def assemble_absolute(self, image: Image.Image, code_plans: list[dict[str, Any]]) -> tuple[str, Image.Image]:
        framework_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
</head>
<body style="margin:0; padding:0; position:relative;">
</body>
</html>
"""

        soup = bs(framework_html, "html.parser")
        body = soup.find("body")
        if body is None:
            raise RuntimeError("Failed to create HTML body for assembly.")

        for node in code_plans:
            bbox = node["module_position"]
            module_soup = bs(node["module_code"], "html.parser")
            body_tag = module_soup.find("body")
            module_markup = body_tag.decode_contents() if body_tag else str(module_soup)

            left = round(bbox[0] * image.width)
            top = round(bbox[1] * image.height)
            width = round((bbox[2] - bbox[0]) * image.width)
            height = round((bbox[3] - bbox[1]) * image.height)

            module_abs = (
                f'<div style="position:absolute; overflow:hidden; left:{left}px; top:{top}px; '
                f'width:{width}px; height:{height}px;">{module_markup}</div>'
            )
            body.append(bs(module_abs, "html.parser"))

        html = soup.prettify()
        rendered = html2shot(html_content=html)
        return html, rendered

    def assemble_agent(
        self,
        image: Image.Image,
        code_plans: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> tuple[str, Image.Image]:
        """
        Assemble modules into a full-page HTML with AgentAssemble.
        Returns (html, rendered_image).
        """
        if temperature is None:
            temperature = self.config.sample_temperature

        payload = [json.dumps(code_plans, ensure_ascii=False), image]
        html = self.agent_assemble.infer(payload, temperature=temperature)
        if not html:
            raise RuntimeError("AgentAssemble returned empty HTML.")

        rendered = html2shot(html_content=html)
        return html, rendered

    def run(self, design_image: Image.Image, run_dir: Path) -> dict[str, Path]:
        run_dir.mkdir(parents=True, exist_ok=True)

        plans, blocks_preview = self.split_modules(design_image)
        blocks_preview_path = run_dir / "blocks.png"
        blocks_preview.save(str(blocks_preview_path))
        codes = self.generate_module_codes(design_image, plans)
        pred_html, pred_img = self.assemble_absolute(design_image, codes)

        input_path = run_dir / "input.png"
        pred_img_path = run_dir / "prediction.png"
        pred_html_path = run_dir / "prediction.html"
        modules_path = run_dir / "modules.json"

        design_image.save(str(input_path))
        pred_img.save(str(pred_img_path))
        pred_html_path.write_text(pred_html, encoding="utf-8")
        modules_path.write_text(json.dumps(codes, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "input": input_path,
            "blocks": blocks_preview_path,
            "prediction_image": pred_img_path,
            "prediction_html": pred_html_path,
            "modules": modules_path,
        }


def build_run_directory(output_root: Path, image: Image.Image) -> Path:
    del image
    return output_root
