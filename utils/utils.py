from __future__ import annotations

import base64
import hashlib
import io
import re
from typing import Iterable

from PIL import Image


def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image2md5(image: Image.Image) -> str:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return hashlib.md5(image_bytes.getvalue()).hexdigest()


def remove_code_markers(code: str) -> str:
    return re.sub(r"^```html\s*|\s*```$", "", code, flags=re.MULTILINE)


def crop_image(image: Image.Image, bbox):
    width, height = image.size
    left = bbox[0] * width
    top = bbox[1] * height
    right = bbox[2] * width
    bottom = bbox[3] * height
    return image.crop((left, top, right, bottom))


def assemble_node_list(node_list: Iterable[dict]) -> str:
    parts = []
    for node in node_list:
        parts.append(
            f"""
## {node['name']}
### bbox
({', '.join([str(x) for x in node['bbox']])})
### html
{node['html']}
"""
        )
    return "".join(parts)
