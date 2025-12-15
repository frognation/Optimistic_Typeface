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
from fontTools.agl import toUnicode as agl_to_unicode
from ufoLib2 import Font


# Historically only a subset had alternates, but we now derive alternates from
# the font contents when writing features.
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
    indices: list[int]
    letter: str
    variant: str  # "base" | "alt"
    glyph_name: str
    conf: float
    bbox: tuple[float, float, float, float]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ensure_outputs(ai_path: Path, svg_path: Path, png_path: Path, scale_to: int, force: bool) -> None:
    # AI here is PDF-compatible.
    ai_mtime = ai_path.stat().st_mtime
    svg_stale = force or (not svg_path.exists()) or (svg_path.stat().st_mtime < ai_mtime)
    png_stale = force or (not png_path.exists()) or (png_path.stat().st_mtime < ai_mtime)

    if svg_stale:
        prefix = svg_path.with_suffix("")

        # Clean previous outputs; pdftocairo may emit prefix.svg or prefix-1.svg.
        if svg_path.exists():
            svg_path.unlink()
        if prefix.exists():
            prefix.unlink()
        for p in svg_path.parent.glob(prefix.name + "*.svg"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

        run(["pdftocairo", "-svg", str(ai_path), str(prefix)])

        # Normalize output name.
        if not svg_path.exists():
            cand0 = prefix
            cand1 = svg_path.parent / f"{prefix.name}-1.svg"
            cand2 = svg_path.parent / f"{prefix.name}.svg"
            if cand0.exists():
                cand0.rename(svg_path)
            elif cand1.exists():
                cand1.rename(svg_path)
            elif cand2.exists():
                cand2.rename(svg_path)
            else:
                raise RuntimeError("pdftocairo did not produce an SVG output")
    if png_stale:
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

    # 2) Layout mapping.
    # We derive rows from y-centers and assign left-to-right.
    # Historically this file had a fixed template (A..Q, some alts, R..Z, some alts).
    # As the AI evolves (e.g. adding new alternates under their base glyphs), we
    # keep base row mapping stable and attach any extra shapes to their nearest
    # base glyph below/above by x-proximity.
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

    def bbox_union(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        ax0, ax1, ay0, ay1 = a
        bx0, bx1, by0, by1 = b
        return (min(ax0, bx0), max(ax1, bx1), min(ay0, by0), max(ay1, by1))

    def bbox_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax0, ax1, ay0, ay1 = a
        bx0, bx1, by0, by1 = b
        ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
        iy0, iy1 = max(ay0, by0), min(ay1, by1)
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        inter = iw * ih
        aa = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
        ba = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
        denom = min(aa, ba)
        if denom <= 0:
            return 0.0
        return inter / denom

    def row_mean_y(r: list[DetectedShape]) -> float:
        return float(np.mean([s.center[1] for s in r])) if r else 0.0

    # Identify base rows.
    # Top base row should contain at least 17 glyphs (A..Q).
    top_candidates = [r for r in rows if len(r) >= 17]
    if not top_candidates:
        raise RuntimeError("Could not find top base row (expected >=17 shapes).")
    top_row = min(top_candidates, key=row_mean_y)

    # Bottom base row should contain at least 9 glyphs (R..Z).
    bottom_candidates = [r for r in rows if 9 <= len(r) <= 12]
    if not bottom_candidates:
        # fallback: any >=9, choose the lowest
        bottom_candidates = [r for r in rows if len(r) >= 9]
    if not bottom_candidates:
        raise RuntimeError("Could not find bottom base row (expected >=9 shapes).")
    bottom_row = max(bottom_candidates, key=row_mean_y)

    # Map base glyphs left-to-right.
    base_map: dict[str, DetectedShape] = {}
    used_indices: set[int] = set()

    for shape, ch in zip(top_row[:17], list("ABCDEFGHIJKLMNOPQ"), strict=True):
        base_map[ch] = shape
        used_indices.add(shape.index)
    for shape, ch in zip(bottom_row[:9], list("RSTUVWXYZ"), strict=True):
        base_map[ch] = shape
        used_indices.add(shape.index)

    assignments: list[GlyphAssignment] = []
    for ch, shape in base_map.items():
        conf = ocr_results.get(shape.index, OcrResult(shape.index, "", -1.0)).conf
        assignments.append(
            GlyphAssignment(
                indices=[shape.index],
                letter=ch,
                variant="base",
                glyph_name=ch,
                conf=conf,
                bbox=shape.bbox,
            )
        )

    # Attach remaining shapes to nearest base glyph by x-proximity (prefer below).
    remaining = [s for s in shapes if s.index not in used_indices]

    def best_base_for(shape: DetectedShape) -> Optional[str]:
        best: tuple[float, Optional[str]] = (float("inf"), None)
        for ch, base_shape in base_map.items():
            dx = abs(shape.center[0] - base_shape.center[0])
            dy = shape.center[1] - base_shape.center[1]  # positive means below (SVG y-down)
            # Strongly penalize shapes above their base glyph.
            if dy < -0.25 * row_break:
                cost = dx + 10_000.0 + abs(dy)
            else:
                cost = dx + 0.25 * abs(dy)
            if cost < best[0]:
                best = (cost, ch)
        return best[1]

    per_base_shapes: dict[str, list[DetectedShape]] = {}
    for s in remaining:
        ch = best_base_for(s)
        if not ch:
            continue
        per_base_shapes.setdefault(ch, []).append(s)

    for ch, shapes_for_base in per_base_shapes.items():
        # Some Illustrator exports may split a single glyph into multiple <path> elements.
        # Cluster overlapping paths so they get imported into one glyph.
        clusters: list[dict[str, object]] = []
        for s in sorted(shapes_for_base, key=lambda s: (s.center[1], s.center[0])):
            placed = False
            for c in clusters:
                cbbox = c["bbox"]  # type: ignore[assignment]
                if bbox_overlap_ratio(cbbox, s.bbox) >= 0.60:  # type: ignore[arg-type]
                    c["shapes"].append(s)  # type: ignore[index]
                    c["bbox"] = bbox_union(cbbox, s.bbox)  # type: ignore[arg-type]
                    placed = True
                    break
            if not placed:
                clusters.append({"shapes": [s], "bbox": s.bbox})

        base_y = base_map[ch].center[1]

        def cluster_sort_key(c: dict[str, object]) -> tuple[float, float]:
            ss: list[DetectedShape] = c["shapes"]  # type: ignore[assignment]
            cy = float(np.mean([x.center[1] for x in ss]))
            return (abs(cy - base_y), cy)

        clusters.sort(key=cluster_sort_key)

        for i, c in enumerate(clusters, start=1):
            ss: list[DetectedShape] = c["shapes"]  # type: ignore[assignment]
            gname = f"{ch}.alt" if i == 1 else f"{ch}.alt{i}"
            # Use the best confidence among components.
            conf = max((ocr_results.get(s.index, OcrResult(s.index, "", -1.0)).conf for s in ss), default=-1.0)
            bbox = c["bbox"]  # type: ignore[assignment]
            assignments.append(
                GlyphAssignment(
                    indices=[s.index for s in ss],
                    letter=ch,
                    variant="alt" if i == 1 else f"alt{i}",
                    glyph_name=gname,
                    conf=conf,
                    bbox=bbox,  # type: ignore[arg-type]
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
    scale: float,
) -> None:
    pen = glyph.getPen()

    # Split into subpaths so we can create separate contours.
    subpaths = sp.continuous_subpaths()
    for sub in subpaths:
        if len(sub) == 0:
            continue

        # Ensure closed contour
        is_closed = sub.isclosed()
        if not is_closed:
            # add a line segment to close
            try:
                sub.append(Line(sub[-1].end, sub[0].start))
                is_closed = True
            except Exception:
                is_closed = False

        start = sub[0].start
        sx, sy = _svg_to_font_xy(start.real, start.imag, vb_h)
        sx = sx * scale + dx
        sy = sy * scale + dy

        pen.moveTo((sx, sy))

        for seg in sub:
            if isinstance(seg, Line):
                ex, ey = _svg_to_font_xy(seg.end.real, seg.end.imag, vb_h)
                pen.lineTo((ex * scale + dx, ey * scale + dy))
            elif isinstance(seg, CubicBezier):
                c1x, c1y = _svg_to_font_xy(seg.control1.real, seg.control1.imag, vb_h)
                c2x, c2y = _svg_to_font_xy(seg.control2.real, seg.control2.imag, vb_h)
                ex, ey = _svg_to_font_xy(seg.end.real, seg.end.imag, vb_h)
                pen.curveTo(
                    (c1x * scale + dx, c1y * scale + dy),
                    (c2x * scale + dx, c2y * scale + dy),
                    (ex * scale + dx, ey * scale + dy),
                )
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
                pen.curveTo(
                    (c1x * scale + dx, c1y * scale + dy),
                    (c2x * scale + dx, c2y * scale + dy),
                    (ex * scale + dx, ey * scale + dy),
                )
            else:
                # Unsupported segment types (Arc etc) — svgpathtools should have already expanded most.
                raise RuntimeError(f"Unsupported SVG segment type: {type(seg)}")

        if is_closed:
            pen.closePath()
        else:
            pen.endPath()


def apply_to_ufo(
    ufo_dir: Path,
    shapes: list[DetectedShape],
    assignments: list[GlyphAssignment],
    vb_h: float,
    svg_path: Path,
    write_features: bool,
    source_upem: int,
    write_unicodes: bool,
) -> None:
    font = Font.open(str(ufo_dir))

    target_upem = getattr(font.info, "unitsPerEm", None) or 1000
    if source_upem <= 0:
        raise ValueError("source_upem must be > 0")
    scale = float(target_upem) / float(source_upem)

    shape_by_index = {s.index: s for s in shapes}

    for a in assignments:
        if not a.indices:
            continue
        if any(i not in shape_by_index for i in a.indices):
            continue

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
        if old_bounds is None and a.glyph_name.startswith(f"{a.letter}.alt"):
            old_bounds = _glyph_bounds(font, a.letter)
        if old_bounds is None:
            old_bounds = (0, 0, 0, 0)

        # Compute union bbox from all component paths.
        sps = [parse_path(shape_by_index[i].d) for i in a.indices]
        xmin = min(sp.bbox()[0] for sp in sps)
        xmax = max(sp.bbox()[1] for sp in sps)
        ymin = min(sp.bbox()[2] for sp in sps)
        ymax = max(sp.bbox()[3] for sp in sps)

        # flip y
        fxmin, fymax = xmin, vb_h - ymin
        fxmax, fymin = xmax, vb_h - ymax
        new_minx, new_miny = fxmin * scale, fymin * scale

        old_minx, old_miny = old_bounds[0], old_bounds[1]
        dx = old_minx - new_minx
        dy = old_miny - new_miny

        glyph.clearContours()
        glyph.clearComponents()

        for sp in sps:
            _draw_svgpath_to_glyph(glyph, sp, vb_h=vb_h, dx=dx, dy=dy, scale=scale)

        glyph.width = old_width

    if write_features:
        def collect_variants(base: str) -> list[str]:
            variants: list[str] = []
            v1 = f"{base}.alt"
            if v1 in font:
                variants.append(v1)
            n = 2
            while True:
                vn = f"{base}.alt{n}"
                if vn not in font:
                    break
                variants.append(vn)
                n += 1
            return variants

        bases = sorted(
            {
                name
                for name in font.keys()
                if len(name) == 1 and name.isalpha() and name.isupper()
            }
        )
        per_base_all = {b: collect_variants(b) for b in bases}
        per_base_all = {b: v for b, v in per_base_all.items() if v}

        # Keep stylistic sets scoped to the original design intent,
        # but let `salt` expose all alternates.
        per_base_sets = {b: per_base_all[b] for b in sorted(ALT_LETTERS) if b in per_base_all}

        lines: list[str] = ["# Auto-generated by scripts/import_op_morph_ai.py", ""]

        # Stylistic Sets: ss01 uses .alt, ss02 uses .alt2, ss03 uses .alt3, ...
        max_alts = max((len(v) for v in per_base_sets.values()), default=0)
        for i in range(1, max_alts + 1):
            tag = f"ss{i:02d}"
            subs: list[str] = []
            for base, variants in per_base_sets.items():
                if len(variants) >= i:
                    subs.append(f"  sub {base} by {variants[i-1]};")
            if subs:
                lines.append(f"feature {tag} {{")
                lines.extend(subs)
                lines.append(f"}} {tag};")
                lines.append("")

        # Stylistic alternates list: lets apps expose alternates selection UI.
        salt_lines: list[str] = []
        for base, variants in per_base_all.items():
            if variants:
                salt_lines.append(f"  sub {base} from [{' '.join(variants)}];")
        if salt_lines:
            lines.append("feature salt {")
            lines.extend(salt_lines)
            lines.append("} salt;")
            lines.append("")

        font.features.text = "\n".join(lines)

    if write_unicodes:
        _assign_unicodes(font)

    font.save(str(ufo_dir), overwrite=True)


def _assign_unicodes(font: Font) -> None:
    """Best-effort restore Unicode values for base glyphs.

    FontGoggles (and OTF export) rely on a cmap. If glyphs have no Unicode
    values in the UFO, typing shows .notdef. We reconstruct Unicodes from
    standard glyph names (AGL + uniXXXX/uXXXXX).

    Notes:
    - We skip alternates (e.g. .alt/.alt2), feature glyphs (.dnom, .numr, etc),
      and non-encoded glyphs.
    - We only assign single-codepoint mappings.
    """

    def parse_uni_name(name: str) -> list[int]:
        if name.startswith("uni") and len(name) >= 7:
            hex_part = name[3:]
            if len(hex_part) % 4 != 0:
                return []
            cps = []
            for i in range(0, len(hex_part), 4):
                chunk = hex_part[i : i + 4]
                try:
                    cps.append(int(chunk, 16))
                except ValueError:
                    return []
            return cps
        if name.startswith("u") and 5 <= len(name) <= 7:
            try:
                return [int(name[1:], 16)]
            except ValueError:
                return []
        return []

    # Anything with a dot suffix is almost always non-encoded.
    skip_suffixes = (
        ".alt",
        ".dnom",
        ".numr",
        ".tnum",
        ".pnum",
        ".onum",
        ".case",
        ".sc",
        ".locl",
    )

    for name in list(font.keys()):
        g = font[name]
        if getattr(g, "unicodes", None):
            continue
        if name.startswith("."):
            continue
        if "." in name:
            # keep alternates/unencoded glyphs unencoded
            continue
        if name.endswith(skip_suffixes):
            continue

        cps = parse_uni_name(name)
        if not cps:
            u = agl_to_unicode(name)
            if u:
                cps = [ord(ch) for ch in u]

        cps = [cp for cp in cps if 0 <= cp <= 0x10FFFF]
        if len(cps) == 1:
            g.unicodes = cps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", default="Vector/Op_Morph.ai")
    ap.add_argument("--ufo", default="OptimisticAI_Morph.ufo")
    ap.add_argument("--out", default="Vector/Op_Morph_exports")
    ap.add_argument("--scale-to", type=int, default=4000)
    ap.add_argument("--source-upem", type=int, default=1000, help="UPEM used when drawing vectors (e.g. 1000)")
    ap.add_argument(
        "--force-render",
        action="store_true",
        help="Re-render SVG/PNG from the AI even if outputs already exist",
    )
    ap.add_argument("--write-features", action="store_true")
    ap.add_argument(
        "--write-unicodes",
        action="store_true",
        help="Restore Unicode mappings in the UFO (fixes .notdef when typing in FontGoggles)",
    )
    ap.add_argument("--apply", action="store_true", help="If set, writes outlines into UFO")
    args = ap.parse_args()

    repo_root = Path.cwd()
    ai_path = (repo_root / args.ai).resolve()
    svg_path = ai_path.with_suffix(".svg")
    png_path = ai_path.with_suffix(".png")

    if not ai_path.exists():
        raise SystemExit(f"AI file not found: {ai_path}")

    ensure_outputs(ai_path, svg_path, png_path, scale_to=args.scale_to, force=args.force_render)

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
            source_upem=args.source_upem,
            write_unicodes=args.write_unicodes,
        )


if __name__ == "__main__":
    main()
