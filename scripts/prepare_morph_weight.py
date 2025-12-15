#!/usr/bin/env python3
"""Prepare additional weight UFOs for the Optimistic AI Morph family.

What this script does (high level):
- Set consistent naming fields (family/style/PostScript).
- Optionally copy features.fea from a reference UFO (feature sharing).
- Optionally create/update a designspace that includes multiple UFO sources.

This script is intentionally conservative: it only touches naming + features +
optionally designspace metadata.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from fontTools.designspaceLib import AxisDescriptor, DesignSpaceDocument, SourceDescriptor
from ufoLib2 import Font


def _derive_ps(family: str, style: str) -> str:
    # Common convention: remove spaces, use '-' between family and style.
    fam = "".join(family.split())
    sty = "".join(style.split())
    return f"{fam}-{sty}" if sty else fam


def _set_names(font: Font, family: str, style: str, ps_name: str | None) -> None:
    font.info.familyName = family
    font.info.styleName = style

    # Typographic naming
    font.info.openTypeNamePreferredFamilyName = family
    font.info.openTypeNamePreferredSubfamilyName = style

    # PostScript
    font.info.postscriptFontName = ps_name or _derive_ps(family, style)

    # Style map: keep family stable. styleMapStyleName is intentionally left as-is
    # (commonly 'regular') unless you decide to do style-linking.
    font.info.styleMapFamilyName = family


def _copy_features(src_ufo: Path, dst_font: Font) -> None:
    src_font = Font.open(str(src_ufo))
    dst_font.features.text = src_font.features.text or ""


def _load_or_create_designspace(path: Path) -> DesignSpaceDocument:
    if path.exists():
        return DesignSpaceDocument.fromfile(str(path))
    return DesignSpaceDocument()


def _ensure_weight_axis(ds: DesignSpaceDocument) -> None:
    # Ensure a Weight axis exists. We keep the axis range broad; sources define actual points.
    for axis in ds.axes:
        if axis.tag == "wght":
            return

    axis = AxisDescriptor()
    axis.name = "Weight"
    axis.tag = "wght"
    axis.minimum = 100
    axis.default = 400
    axis.maximum = 900
    ds.addAxis(axis)


def _source_id(ufo_path: Path) -> str:
    # A stable identifier based on filename.
    return ufo_path.stem


def _add_or_update_source(
    ds: DesignSpaceDocument,
    ufo_path: Path,
    weight_value: int,
    family: str,
    style: str,
    designspace_dir: Path,
) -> None:
    # Update existing source if present.
    sid = _source_id(ufo_path)
    rel_path = os.path.relpath(ufo_path.resolve(), designspace_dir.resolve()).replace(os.sep, "/")
    for s in ds.sources:
        if getattr(s, "name", None) == sid or Path(s.path).name == ufo_path.name:
            s.path = rel_path
            s.familyName = family
            s.styleName = style
            s.location = {"Weight": float(weight_value)}
            return

    src = SourceDescriptor()
    src.name = sid
    src.path = rel_path
    src.familyName = family
    src.styleName = style
    src.location = {"Weight": float(weight_value)}
    ds.addSource(src)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a new weight UFO for Optimistic AI Morph")
    ap.add_argument("--in-ufo", required=True, help="Input UFO directory (existing)")
    ap.add_argument(
        "--out-ufo",
        help="Output UFO directory. If omitted, edits the input UFO in place.",
    )

    ap.add_argument("--family", default="Optimistic AI", help="Family name")
    ap.add_argument("--style", required=True, help='Style name, e.g. "Light Morph" or "Bold Morph"')
    ap.add_argument("--ps", help="PostScript name; if omitted, derived from family/style")

    ap.add_argument(
        "--copy-features-from",
        default="OptimisticAI_Morph.ufo",
        help="Reference UFO to copy features.fea from (feature sharing). Use empty string to skip.",
    )

    ap.add_argument(
        "--designspace",
        default="OptimisticAI_Morph.designspace",
        help="Designspace path to create/update. Use empty string to skip.",
    )

    ap.add_argument(
        "--weight",
        type=int,
        help="Weight value for designspace (wght). If omitted, uses openTypeOS2WeightClass from the UFO.",
    )

    args = ap.parse_args()

    in_ufo = Path(args.in_ufo)
    if not in_ufo.exists():
        raise SystemExit(f"Input UFO not found: {in_ufo}")

    out_ufo = Path(args.out_ufo) if args.out_ufo else in_ufo
    if out_ufo != in_ufo:
        if out_ufo.exists():
            raise SystemExit(f"Output UFO already exists: {out_ufo}")
        shutil.copytree(in_ufo, out_ufo)

    font = Font.open(str(out_ufo))
    _set_names(font, family=args.family, style=args.style, ps_name=args.ps)

    if args.copy_features_from:
        ref = Path(args.copy_features_from)
        if not ref.exists():
            raise SystemExit(f"Reference UFO for features not found: {ref}")
        _copy_features(ref, font)

    font.save(str(out_ufo), overwrite=True)

    if args.designspace:
        ds_path = Path(args.designspace)
        ds = _load_or_create_designspace(ds_path)
        _ensure_weight_axis(ds)
        ds_dir = ds_path.parent

        # Determine weight.
        font2 = Font.open(str(out_ufo))
        w = args.weight
        if w is None:
            w = getattr(font2.info, "openTypeOS2WeightClass", None) or 400

        _add_or_update_source(ds, out_ufo, int(w), args.family, args.style, ds_dir)

        # If the reference Morph UFO exists, include it too (helpful when starting a designspace).
        ref_ufo = Path("OptimisticAI_Morph.ufo")
        if ref_ufo.exists() and ref_ufo.resolve() != out_ufo.resolve():
            ref_font = Font.open(str(ref_ufo))
            ref_style = getattr(ref_font.info, "styleName", None) or "Morph"
            ref_weight = getattr(ref_font.info, "openTypeOS2WeightClass", None) or 400
            _add_or_update_source(ds, ref_ufo, int(ref_weight), args.family, ref_style, ds_dir)

        ds.write(ds_path.as_posix())


if __name__ == "__main__":
    main()
