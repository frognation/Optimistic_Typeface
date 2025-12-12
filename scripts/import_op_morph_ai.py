#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pytesseract
from lxml import etree
from PIL import Image, ImageOps
from svgpathtools import CubicBezier, Line, Path as SvgPath, QuadraticBezier, parse_path

from fontTools.pens.boundsPen import BoundsPen
from ufoLib2 import Font


ALT_LETTERS = {"A", "B", "E", "F", "K", "M", "P", "Q", "R", "W"}


@dataclass(frozen=True)
class DetectedShape:
    index: int
    d: str
    bbox: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax (SVG viewBox coords)
    center: tuple[float, float]


@dataclass(frozen=True)
class OcrResult:
    index: int
    ocr: str
    conf: float


@dataclass(frozen=True)
class GlyphAssignment:
    index: int
    letter: str
    variant: str  # "base" | "alt"
    glyph_name: str
    conf: float
    bbox: tuple[float, float, float, float]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ensure_outputs(ai_path: Path, svg_path: Path, png_path: Path, scale_to: int) -> None:
    # AI here is PDF-compatible.
    if not svg_path.exists():
        run(["pdftocairo", "-svg", str(ai_path), str(svg_path.with_suffix(""))])
        # pdftocairo writes without extension sometimes; normalize
        if not svg_path.exists() and svg_path.with_suffix("").exists():
            svg_path.with_suffix("").rename(svg_path)
    if not png_path.exists():
        run([
            "pdftocairo",
            "-png",
            "-singlefile",
            "-scale-to",
            str(scale_to),
            str(ai_path),
            str(png_path.with_suffix("")),
        ])
        if not png_path.exists() and png_path.with_suffix("").exists():
            png_path.with_suffix("").rename(png_path)


def parse_svg_paths(svg_path: Path) -> tuple[list[DetectedShape], float, float]:
    tree = etree.parse(str(svg_path))
    root = tree.getroot()

    view_box = root.get("viewBox")
    if not view_box:
        raise RuntimeError("SVG missing viewBox")
    vb = [float(x) for x in view_box.split()]
    vb_w, vb_h = vb[2], vb[3]

    ns = {"svg": "http://www.w3.org/2000/svg"}
    path_nodes = root.findall(".//svg:path", namespaces=ns)

    shapes: list[DetectedShape] = []
    for i, node in enumerate(path_nodes):
        d = node.get("d")
        if not d:
            continue
        sp = parse_path(d)
        xmin, xmax, ymin, ymax = sp.bbox()
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        shapes.append(DetectedShape(index=i, d=d, bbox=(xmin, xmax, ymin, ymax), center=(cx, cy)))

    return shapes, vb_w, vb_h


def _crop_for_bbox(page: Image.Image, bbox: tuple[float, float, float, float], sx: float, sy: float) -> Image.Image:
    xmin, xmax, ymin, ymax = bbox
    px0 = int(math.floor(xmin * sx))
    px1 = int(math.ceil(xmax * sx))
    py0 = int(math.floor(ymin * sy))
    py1 = int(math.ceil(ymax * sy))

    w = max(1, px1 - px0)
    h = max(1, py1 - py0)
    pad = int(max(8, 0.10 * max(w, h)))

    x0 = max(0, px0 - pad)
    y0 = max(0, py0 - pad)
    x1 = min(page.width, px1 + pad)
    y1 = min(page.height, py1 + pad)

    crop = page.crop((x0, y0, x1, y1))
    crop = crop.convert("L")
    crop = ImageOps.autocontrast(crop)

    # binary-ish to help tesseract
    arr = np.array(crop)
    thr = np.percentile(arr, 70)
    bw = (arr < thr).astype(np.uint8) * 255
    return Image.fromarray(bw, mode="L")


def ocr_letter(img: Image.Image) -> tuple[str, float]:
    config = "--oem 1 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
    best = ("", -1.0)
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        txt = (txt or "").strip().upper()
        try:
            c = float(conf)
        except Exception:
            c = -1.0
        if len(txt) == 1 and txt.isalpha() and c > best[1]:
            best = (txt, c)

    if best[0]:
        return best[0], best[1]

    # fallback a bit more permissive
    config2 = "--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    txt = (pytesseract.image_to_string(img, config=config2) or "").strip().upper()
    if len(txt) >= 1:
        ch = next((c for c in txt if c.isalpha()), "")
        return ch, -1.0
    return "", -1.0


def build_assignments(
    shapes: list[DetectedShape],
    png_path: Path,
    vb_w: float,
    vb_h: float,
    out_dir: Path,
) -> list[GlyphAssignment]:
    out_dir.mkdir(parents=True, exist_ok=True)

    page = Image.open(png_path)
    sx = page.width / vb_w
    sy = page.height / vb_h

    # 1) OCR everything (best-effort) and save crops for human inspection.
    ocr_results: dict[int, OcrResult] = {}
    for s in shapes:
        crop = _crop_for_bbox(page, s.bbox, sx, sy)
        letter, conf = ocr_letter(crop)
        ocr_results[s.index] = OcrResult(index=s.index, ocr=letter, conf=conf)
        tag = letter if letter else "unknown"
        crop.save(out_dir / f"shape_{s.index:04d}_{tag}.png")

    # 2) Primary mapping: Op_Morph layout is stable (see screenshot):
    # Row 1: A..Q (17)
    # Row 2: A,B,E,F,K,M,P,Q alternates (8)
    # Row 3: R..Z (9)
    # Row 4: R,W alternates (2)
    # We derive rows from y-centers and assign left-to-right.
    heights = [s.bbox[3] - s.bbox[2] for s in shapes]
    median_h = float(np.median(np.array(heights))) if heights else 0.0
    row_break = max(1.0, 0.60 * median_h)

    shapes_sorted_y = sorted(shapes, key=lambda s: s.center[1])
    rows: list[list[DetectedShape]] = []
    for s in shapes_sorted_y:
        if not rows:
            rows.append([s])
            continue
        if abs(s.center[1] - rows[-1][-1].center[1]) <= row_break:
            rows[-1].append(s)
        else:
            rows.append([s])

    # Normalize each row left-to-right and sort rows top-to-bottom.
    rows = [sorted(r, key=lambda s: s.center[0]) for r in rows]
    rows = sorted(rows, key=lambda r: float(np.mean([s.center[1] for s in r])))

    row_templates: list[list[tuple[str, str]]] = [
        [(ch, "base") for ch in list("ABCDEFGHIJKLMNOPQ")],
        [(ch, "alt") for ch in ["A", "B", "E", "F", "K", "M", "P", "Q"]],
        [(ch, "base") for ch in list("RSTUVWXYZ")],
        [(ch, "alt") for ch in ["R", "W"]],
    ]

    # Find best matching row->template assignment by row lengths.
    expected_lengths = [len(t) for t in row_templates]
    actual_lengths = [len(r) for r in rows]
    if actual_lengths != expected_lengths:
        raise RuntimeError(
            f"Unexpected row layout. Expected rows {expected_lengths} but got {actual_lengths}. "
            "If the AI layout changed, export per-artboard or update templates."
        )

    assignments: list[GlyphAssignment] = []
    for row, tmpl in zip(rows, row_templates, strict=True):
        for shape, (letter, variant) in zip(row, tmpl, strict=True):
            gname = f"{letter}.alt" if variant == "alt" else letter
            conf = ocr_results.get(shape.index, OcrResult(shape.index, "", -1.0)).conf
            assignments.append(
                GlyphAssignment(
                    index=shape.index,
                    letter=letter,
                    variant=variant,
                    glyph_name=gname,
                    conf=conf,
                    bbox=shape.bbox,
                )
            )

    mapping = {
        "svg_viewbox": [0, 0, vb_w, vb_h],
        "png_size": [page.width, page.height],
        "assignments": [asdict(a) for a in sorted(assignments, key=lambda a: (a.letter, a.variant))],
        "ocr": {str(i): asdict(r) for i, r in ocr_results.items()},
    }
    (out_dir / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    return assignments


def _glyph_bounds(font: Font, glyph_name: str) -> Optional[tuple[float, float, float, float]]:
    if glyph_name not in font:
        return None
    g = font[glyph_name]
    pen = BoundsPen(font)
    g.draw(pen)
    return pen.bounds


def _svg_to_font_xy(x: float, y: float, vb_h: float) -> tuple[float, float]:
    # SVG viewBox has y-down. UFO has y-up.
    return x, vb_h - y


def _draw_svgpath_to_glyph(
    glyph,
    sp: SvgPath,
    vb_h: float,
    dx: float,
    dy: float,
) -> None:
    pp = glyph.getPointPen()

    # Split into subpaths so we can create separate contours.
    subpaths = sp.continuous_subpaths()
    for sub in subpaths:
        if len(sub) == 0:
            continue

        # Ensure closed contour
        if not sub.isclosed():
            # add a line segment to close
            try:
                sub.append(Line(sub[-1].end, sub[0].start))
            except Exception:
                pass

        start = sub[0].start
        sx, sy = _svg_to_font_xy(start.real, start.imag, vb_h)
        sx += dx
        sy += dy

        pp.beginPath()
        pp.addPoint((sx, sy), segmentType="move", smooth=False, name=None)

        for seg in sub:
            if isinstance(seg, Line):
                ex, ey = _svg_to_font_xy(seg.end.real, seg.end.imag, vb_h)
                pp.addPoint((ex + dx, ey + dy), segmentType="line", smooth=False, name=None)
            elif isinstance(seg, CubicBezier):
                c1x, c1y = _svg_to_font_xy(seg.control1.real, seg.control1.imag, vb_h)
                c2x, c2y = _svg_to_font_xy(seg.control2.real, seg.control2.imag, vb_h)
                ex, ey = _svg_to_font_xy(seg.end.real, seg.end.imag, vb_h)
                pp.addPoint((c1x + dx, c1y + dy), segmentType=None, smooth=False, name=None)
                pp.addPoint((c2x + dx, c2y + dy), segmentType=None, smooth=False, name=None)
                pp.addPoint((ex + dx, ey + dy), segmentType="curve", smooth=False, name=None)
            elif isinstance(seg, QuadraticBezier):
                # Approximate quadratic as cubic (standard conversion)
                x0, y0 = seg.start.real, seg.start.imag
                x1, y1 = seg.control.real, seg.control.imag
                x2, y2 = seg.end.real, seg.end.imag
                c1 = complex(x0 + 2 / 3 * (x1 - x0), y0 + 2 / 3 * (y1 - y0))
                c2 = complex(x2 + 2 / 3 * (x1 - x2), y2 + 2 / 3 * (y1 - y2))
                c1x, c1y = _svg_to_font_xy(c1.real, c1.imag, vb_h)
                c2x, c2y = _svg_to_font_xy(c2.real, c2.imag, vb_h)
                ex, ey = _svg_to_font_xy(x2, y2, vb_h)
                pp.addPoint((c1x + dx, c1y + dy), segmentType=None, smooth=False, name=None)
                pp.addPoint((c2x + dx, c2y + dy), segmentType=None, smooth=False, name=None)
                pp.addPoint((ex + dx, ey + dy), segmentType="curve", smooth=False, name=None)
            else:
                # Unsupported segment types (Arc etc) — svgpathtools should have already expanded most.
                raise RuntimeError(f"Unsupported SVG segment type: {type(seg)}")

        pp.endPath()


def apply_to_ufo(
    ufo_dir: Path,
    shapes: list[DetectedShape],
    assignments: list[GlyphAssignment],
    vb_h: float,
    svg_path: Path,
    write_features: bool,
) -> None:
    font = Font.open(str(ufo_dir))

    shape_by_index = {s.index: s for s in shapes}

    for a in assignments:
        if a.index not in shape_by_index:
            continue
        shape = shape_by_index[a.index]

        if a.glyph_name in font:
            glyph = font[a.glyph_name]
        else:
            glyph = font.newGlyph(a.glyph_name)
            # If this is an alt, keep width from base.
            base_name = a.letter
            if base_name in font:
                glyph.width = font[base_name].width

        # Keep existing width as-is.
        old_width = glyph.width
        old_bounds = _glyph_bounds(font, a.glyph_name)
        if old_bounds is None and a.glyph_name.endswith(".alt"):
            old_bounds = _glyph_bounds(font, a.letter)
        if old_bounds is None:
            old_bounds = (0, 0, 0, 0)

        sp = parse_path(shape.d)

        # Compute bbox in font coords for this shape
        xmin, xmax, ymin, ymax = sp.bbox()
        # flip y
        fxmin, fymax = xmin, vb_h - ymin
        fxmax, fymin = xmax, vb_h - ymax
        new_minx, new_miny = fxmin, fymin

        old_minx, old_miny = old_bounds[0], old_bounds[1]
        dx = old_minx - new_minx
        dy = old_miny - new_miny

        glyph.clearContours()
        glyph.clearComponents()

        _draw_svgpath_to_glyph(glyph, sp, vb_h=vb_h, dx=dx, dy=dy)

        glyph.width = old_width

    if write_features:
        # Add a minimal stylistic set (ss01) for the requested uppercase alternates.
        lines = [
            "# Auto-generated by scripts/import_op_morph_ai.py",
            "feature ss01 {",
        ]
        for letter in sorted(ALT_LETTERS):
            alt = f"{letter}.alt"
            if alt in font and letter in font:
                lines.append(f"  sub {letter} by {alt};")
        lines += ["} ss01;", ""]
        (ufo_dir / "features.fea").write_text("\n".join(lines), encoding="utf-8")

    font.save(str(ufo_dir), overwrite=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", default="Vector/Op_Morph.ai")
    ap.add_argument("--ufo", default="OptimisticAI_Medium.ufo")
    ap.add_argument("--out", default="Vector/Op_Morph_exports")
    ap.add_argument("--scale-to", type=int, default=4000)
    ap.add_argument("--write-features", action="store_true")
    ap.add_argument("--apply", action="store_true", help="If set, writes outlines into UFO")
    args = ap.parse_args()

    repo_root = Path.cwd()
    ai_path = (repo_root / args.ai).resolve()
    svg_path = ai_path.with_suffix(".svg")
    png_path = ai_path.with_suffix(".png")

    if not ai_path.exists():
        raise SystemExit(f"AI file not found: {ai_path}")

    ensure_outputs(ai_path, svg_path, png_path, scale_to=args.scale_to)

    shapes, vb_w, vb_h = parse_svg_paths(svg_path)

    out_dir = (repo_root / args.out).resolve()
    assignments = build_assignments(shapes, png_path, vb_w, vb_h, out_dir)

    if args.apply:
        ufo_dir = (repo_root / args.ufo).resolve()
        apply_to_ufo(
            ufo_dir=ufo_dir,
            shapes=shapes,
            assignments=assignments,
            vb_h=vb_h,
            svg_path=svg_path,
            write_features=args.write_features,
        )


if __name__ == "__main__":
    main()
