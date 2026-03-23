from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw

import argparse
import json
from pathlib import Path

BBox = List[int]
PropBBox = List[float]
TreeNode = Dict[str, object]

# Global cache of discovered split boxes while traversing the split tree.
BBOX_CACHE: List[BBox] = []


def blocker(image: Image.Image):
    """Split a screenshot into block candidates and return normalized boxes + preview image."""
    BBOX_CACHE.clear()

    root: TreeNode = {
        "image": image,
        "bbox": [0, 0, image.width, image.height],
        "ox": 0,
        "oy": 0,
    }

    ocr_results, _ = ocr_with_easyocr(image, merge_threshold=20)
    text_boxes = [item[0] for item in ocr_results]

    treeSplit(root, mceil=5, mblock=50, max_deep=3, skip=10, bboxs=text_boxes)
    _merge_leaf_nodes(root)

    # drawSplitTree records split boxes into BBOX_CACHE.
    drawSplitTree(image.copy(), root)

    bboxes = remove_contained_bboxes(BBOX_CACHE)
    if not bboxes:
        bboxes = [[0, 0, image.width, image.height]]

    crops = crop_image_by_bboxes(image, bboxes)
    bboxes = filter_non_blank_bboxes(crop_images=crops, bboxes=bboxes)
    if not bboxes:
        bboxes = [[0, 0, image.width, image.height]]

    preview = drawwhole(image=image, bboxes=bboxes)
    return length2propotion(image2=image, bboxes=bboxes), preview


def _merge_leaf_nodes(root: TreeNode, max_iter: int = 100, min_area: int = 300 * 300, min_edge: int = 300):
    iter_count = 0
    while True:
        node = tryFind(root)
        if not node:
            break

        node["tried"] = True
        should_continue = True

        while should_continue:
            children = node["children"]
            should_continue = False

            for idx, current in enumerate(children):
                if idx >= len(children) - 1:
                    break

                nxt = children[idx + 1]
                if current.get("tried", False) or nxt.get("tried", False):
                    continue

                bbox = current["bbox"]
                next_bbox = nxt["bbox"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                next_area = (next_bbox[2] - next_bbox[0]) * (next_bbox[3] - next_bbox[1])

                merge_for_area = (area < min_area or next_area < min_area)
                merge_for_pure = (current["pure"] and not nxt["children"]) or (nxt["pure"] and not current["children"])
                same_width_small = (bbox[2] - bbox[0] == next_bbox[2] - next_bbox[0]) and (
                    bbox[3] - bbox[1] < min_edge and next_bbox[3] - next_bbox[1] < min_edge
                )
                same_height_small = (bbox[3] - bbox[1] == next_bbox[3] - next_bbox[1]) and (
                    bbox[2] - bbox[0] < min_edge and next_bbox[2] - next_bbox[0] < min_edge
                )

                if merge_for_area or merge_for_pure or same_width_small or same_height_small:
                    mergeBros(node, idx, idx + 1)
                    should_continue = True
                    break

        if len(node["children"]) == 1:
            node["children"] = []
            node["tried"] = False

        iter_count += 1
        if iter_count >= max_iter:
            break


def drawOnImage(image: Image.Image, bbox: Sequence[int], ox: int = 0, oy: int = 0, copy: bool = False, padding: int = 2):
    if copy:
        image = image.copy()

    draw = ImageDraw.Draw(image)
    padded_bbox = [
        int(bbox[0] + ox + padding),
        int(bbox[1] + oy + padding),
        int(bbox[2] + ox - padding),
        int(bbox[3] + oy - padding),
    ]
    BBOX_CACHE.append(padded_bbox)
    draw.rectangle(padded_bbox, outline="red", width=2)
    return image


def splitImage(image: Image.Image, mceil: int = 10):
    width, height = image.size
    dw = max(1, round(width / mceil))
    dh = max(1, round(height / mceil))

    x_step = width / dw
    y_step = height / dh

    points = []
    last_color = None
    all_same = True

    for j in range(2, dh):
        row = []
        for i in range(2, dw):
            x = int(i * x_step) - 1
            y = int(j * y_step) - 1
            color = image.getpixel((x, y))
            if all_same and last_color is not None and last_color != color:
                all_same = False
            last_color = color
            row.append({"x": x, "y": y, "color": color})
        if row:
            points.append(row)

    return points, all_same


def breakImage(
    image: Image.Image,
    points,
    axis: str = "x",
    mblock: int = 50,
    skip: int = 5,
    bboxs: Optional[list] = None,
    ox: int = 0,
    oy: int = 0,
):
    if not points or not points[0]:
        return []

    bboxs = bboxs or []
    xs = len(points[0])
    ys = len(points)
    lines = []

    if axis == "x":
        for i in range(xs):
            flag = True
            prev_color = None

            for j in range(ys):
                if j < skip or j > ys - skip - 1:
                    continue
                p = points[j][i]
                color = p["color"]
                x = p["x"] + ox
                over_text = any(x >= b[0][0] and x <= b[2][0] for b in bboxs)
                if over_text:
                    flag = False
                    break
                if prev_color is None:
                    prev_color = color
                elif color != prev_color:
                    flag = False
                    break

            if flag:
                x = points[0][i]["x"]
                enough_gap = (len(lines) and x - lines[-1][0]["x"] > mblock) or (not lines and x > mblock)
                if image.width - x > mblock and enough_gap:
                    lines.append([{"x": x, "y": 0}, {"x": x, "y": image.height}])

        if lines:
            lines.append([{"x": image.width, "y": 0}, {"x": image.width, "y": image.height}])
    else:
        for i in range(ys):
            flag = True
            prev_color = None

            for j in range(xs):
                if j < skip or j > xs - skip - 1:
                    continue
                p = points[i][j]
                color = p["color"]
                y = p["y"] + oy
                over_text = any(y >= b[0][1] and y <= b[2][1] for b in bboxs)
                if over_text:
                    flag = False
                    break
                if prev_color is None:
                    prev_color = color
                elif color != prev_color:
                    flag = False
                    break

            if flag:
                y = points[i][0]["y"]
                enough_gap = (len(lines) and y - lines[-1][1]["y"] > mblock) or (not lines and y > mblock)
                if image.height - y > mblock and enough_gap:
                    lines.append([{"x": 0, "y": y}, {"x": image.width, "y": y}])

        if lines:
            lines.append([{"x": 0, "y": image.height}, {"x": image.width, "y": image.height}])

    return lines


def applyBreak(image: Image.Image, brs, ox: int = 0, oy: int = 0):
    last_x, last_y = 0, 0
    children = []
    for br in brs:
        bbox = (last_x, last_y, br[1]["x"], br[1]["y"])
        crop = image.crop(bbox)
        last_x = br[0]["x"]
        last_y = br[0]["y"]
        children.append({"image": crop, "bbox": bbox, "ox": ox + bbox[0], "oy": oy + bbox[1]})
    return children


def mergeCnodes(node, begin: int, end: int):
    node["children"][end]["bbox"] = [
        node["children"][begin]["bbox"][0],
        node["children"][begin]["bbox"][1],
        node["children"][end]["bbox"][2],
        node["children"][end]["bbox"][3],
    ]
    node["children"] = node["children"][:begin] + node["children"][end:]
    if len(node["children"]) == 1:
        node["children"] = []


def treeSplit(node, mceil: int = 10, mblock: int = 50, deep: int = 1, max_deep: int = 10, skip: int = 5, bboxs: Optional[list] = None):
    bboxs = bboxs or []
    if deep > max_deep:
        node["children"] = []
        return

    image = node["image"]
    points, all_same = splitImage(image, mceil)
    if all_same or not points:
        node["children"] = []
        return

    breaks = breakImage(image, points, axis="y", mblock=mblock, skip=skip, bboxs=bboxs, ox=node["ox"], oy=node["oy"])
    if not breaks:
        breaks = breakImage(image, points, axis="x", mblock=mblock, skip=skip, bboxs=bboxs, ox=node["ox"], oy=node["oy"])

    if not breaks:
        node["children"] = []
        return

    children = applyBreak(image, breaks, node["ox"], node["oy"])
    node["children"] = children

    for child in node["children"]:
        _, pure = splitImage(child["image"], mceil)
        child["pure"] = pure
        if child["image"].width > mceil and child["image"].height > mceil:
            treeSplit(child, mceil, mblock, deep=deep + 1, max_deep=max_deep, skip=skip, bboxs=bboxs)


def drawSplitTree(image: Image.Image, node, ox: int = 0, oy: int = 0, deep: int = 1):
    for child in node["children"]:
        drawOnImage(image, child["bbox"], ox, oy, padding=deep**2)
        drawSplitTree(image, child, ox=ox + child["bbox"][0], oy=oy + child["bbox"][1], deep=deep + 1)


def tryFind(node):
    children = node.get("children", [])
    if isinstance(children, list) and children:
        all_children_are_leaves = all(("children" not in child) or len(child["children"]) == 0 for child in children)
        if all_children_are_leaves and not node.get("tried", False):
            return node

        for child in children:
            result = tryFind(child)
            if result:
                return result

        all_children_relaxed = all(
            ("children" not in child)
            or len(child["children"]) == 0
            or child.get("tried", False)
            for child in children
        )
        if all_children_relaxed and not node.get("tried", False):
            return node

        for child in children:
            result = tryFind(child)
            if result:
                return result

    return None


def tryMerge(node):
    node["tried"] = True
    flag = True
    while flag:
        last_pure_idx = -1
        flag = False
        for idx, child in enumerate(node["children"]):
            if child["pure"]:
                if last_pure_idx == -1:
                    last_pure_idx = idx
                elif idx == len(node["children"]) - 1:
                    mergeCnodes(node, last_pure_idx, idx)
                    flag = True
                    break
            elif last_pure_idx != -1:
                if idx - last_pure_idx > 1:
                    mergeCnodes(node, last_pure_idx, idx - 1)
                    flag = True
                    break
                last_pure_idx = -1


def mergeBros(fnode, idx1: int, idx2: int):
    node = fnode["children"][idx1]
    next_node = fnode["children"][idx2]
    node["bbox"] = [node["bbox"][0], node["bbox"][1], next_node["bbox"][2], next_node["bbox"][3]]
    node["pure"] = node["pure"] and next_node["pure"]
    fnode["children"] = fnode["children"][:idx2] + fnode["children"][idx2 + 1 :]


def merge_bboxs(results, threshold: int = 50):
    while True:
        merged = False
        for idx, item in enumerate(results):
            bbox, text, _ = item
            left, top, right, bottom = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]
            for idx2, item2 in enumerate(results):
                if idx2 == idx:
                    continue
                bbox2, text2, _ = item2
                left2, top2, right2, bottom2 = bbox2[0][0], bbox2[0][1], bbox2[2][0], bbox2[2][1]
                overlaps = not (
                    left2 > right + threshold
                    or right2 < left - threshold
                    or bottom2 < top - threshold
                    or top2 > bottom + threshold
                )
                if overlaps:
                    left3, top3 = min(left, left2), min(top, top2)
                    right3, bottom3 = max(right, right2), max(bottom, bottom2)
                    results[idx] = (
                        [[left3, top3], [right3, top3], [right3, bottom3], [left3, bottom3]],
                        f"{text}\n{text2}",
                        None,
                    )
                    results = results[:idx2] + results[idx2 + 1 :]
                    merged = True
                    break
            if merged:
                break
        if not merged:
            return results


def ocr_with_easyocr(pil_image: Image.Image, lang_list: Optional[List[str]] = None, merge_threshold: int = 50):
    lang_list = lang_list or ["en"]
    reader = easyocr.Reader(lang_list, gpu=False)

    image_np = np.array(pil_image)
    results = reader.readtext(image_np)
    results = merge_bboxs(results, threshold=merge_threshold)

    for bbox, _, _ in results:
        top_left = tuple(map(int, bbox[0]))
        top_right = tuple(map(int, bbox[1]))
        bottom_right = tuple(map(int, bbox[2]))
        bottom_left = tuple(map(int, bbox[3]))
        cv2.polylines(
            image_np,
            [np.array([top_left, top_right, bottom_right, bottom_left])],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

    result_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    return results, result_image


def length2propotion(image2: Image.Image, bboxes: List[BBox]):
    for bbox in bboxes:
        bbox[0] = round(bbox[0] / image2.width, 3)
        bbox[1] = round(bbox[1] / image2.height, 3)
        bbox[2] = round(bbox[2] / image2.width, 3)
        bbox[3] = round(bbox[3] / image2.height, 3)
    return bboxes


def propotion2length(image: Image.Image, bboxes: List[PropBBox]):
    abs_boxes = []
    for bbox in bboxes:
        x1 = int(bbox[0] * image.width)
        y1 = int(bbox[1] * image.height)
        x2 = int(bbox[2] * image.width)
        y2 = int(bbox[3] * image.height)
        abs_boxes.append([x1, y1, x2, y2])
    return abs_boxes


def crop_image_by_bboxes(image: Image.Image, bboxes: List[BBox]):
    return [image.crop((x1, y1, x2, y2)) for x1, y1, x2, y2 in bboxes]


def is_blank_image(image: Image.Image):
    rgb = image.convert("RGB")
    return all(pixel == (255, 255, 255) for pixel in rgb.getdata())


def filter_non_blank_bboxes(crop_images: List[Image.Image], bboxes: List[BBox]):
    return [bbox for bbox, crop in zip(bboxes, crop_images) if not is_blank_image(crop)]


def is_contained(bbox1: Sequence[int], bbox2: Sequence[int]):
    x1, y1, x2, y2 = bbox1
    x1p, y1p, x2p, y2p = bbox2
    return x1 <= x1p and y1 <= y1p and x2 >= x2p and y2 >= y2p


def remove_contained_bboxes(bboxes: List[BBox]):
    result = []
    for i, bbox1 in enumerate(bboxes):
        contained = False
        for j, bbox2 in enumerate(bboxes):
            if i != j and is_contained(bbox1, bbox2):
                contained = True
                break
        if not contained:
            result.append(bbox1)
    return result


def drawwhole(image: Image.Image, bboxes: List[BBox]):
    image2 = image.copy()
    draw = ImageDraw.Draw(image2)
    for x1, y1, x2, y2 in bboxes:
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
    return image2



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test smart_blocker on one GUI image.")
    parser.add_argument("--image", "-i", required=True, help="Input GUI image path")
    parser.add_argument(
        "--out-image",
        "-o",
        default="output/blocks_test.png",
        help="Output image path for blocker preview",
    )
    parser.add_argument(
        "--out-json",
        default="output/blocks_test.json",
        help="Output JSON path for normalized block coordinates",
    )
    return parser


def main() -> int:
    

    args = build_parser().parse_args()

    image_path = Path(args.image).expanduser().resolve()
    out_image_path = Path(args.out_image).expanduser().resolve()
    out_json_path = Path(args.out_json).expanduser().resolve()

    if not image_path.exists() or not image_path.is_file():
        print(f"Input image not found: {image_path}")
        return 1

    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    plans, preview = blocker(image)

    preview.save(str(out_image_path))
    out_json_path.write_text(json.dumps(plans, ensure_ascii=False, indent=2), encoding="utf-8")

    print("smart_blocker test completed.")
    print(f"Input image: {image_path}")
    print(f"Output preview image: {out_image_path}")
    print(f"Output blocks JSON: {out_json_path}")
    print(f"Detected blocks: {len(plans)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())