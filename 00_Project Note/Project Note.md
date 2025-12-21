# Project Note – Optimistic AI Morph (KO/EN)

> 목적: 나중에 다시 돌아오거나 다른 사람이 이 레포를 봤을 때, 지금까지의 작업 내용/의사결정/재현 방법을 빠르게 이해할 수 있도록 정리.

---

## 1) 프로젝트 개요 / Overview

**KO**
- Illustrator(벡터)로 그려진 글립 도안을 UFO(Optimistic AI 기반)에 주입해 `Morph` 스타일을 구성.
- 기존 UFO의 폭/커닝/메트릭은 유지하고, 아웃라인만 교체.
- OpenType alternates(`ss01`, `ss02`, `salt`)를 생성.

**EN**
- Import Illustrator vector glyph drawings into a UFO (based on Optimistic AI) to create the `Morph` style.
- Preserve existing widths/kerning/metrics; replace outlines only.
- Generate OpenType alternates features (`ss01`, `ss02`, `salt`).

---

## 2) 현재 산출물 / Current outputs

**KO**
- UFO: [OptimisticAI_Morph.ufo](OptimisticAI_Morph.ufo)
- Features: [OptimisticAI_Morph.ufo/features.fea](OptimisticAI_Morph.ufo/features.fea)
- 입력 벡터(최신): [Vector/Op_Morph_Flat edge.ai](Vector/Op_Morph_Flat%20edge.ai)
- 빌드된 OTF(검증용): [build_otf/OptimisticAI_Morph.otf](build_otf/OptimisticAI_Morph.otf)

**EN**
- UFO: [OptimisticAI_Morph.ufo](OptimisticAI_Morph.ufo)
- Features: [OptimisticAI_Morph.ufo/features.fea](OptimisticAI_Morph.ufo/features.fea)
- Source vectors (latest): [Vector/Op_Morph_Flat edge.ai](Vector/Op_Morph_Flat%20edge.ai)
- Built OTF (for validation): [build_otf/OptimisticAI_Morph.otf](build_otf/OptimisticAI_Morph.otf)

---

## 3) 중요한 의사결정 / Key decisions

### 3.1 UPEM 처리 / UPEM handling

**KO**
- UFO의 `unitsPerEm=2000`을 유지.
- Illustrator 벡터가 1000 기준일 때, **주입되는 outline만 2배 스케일**해서 맞춤.

**EN**
- Keep UFO `unitsPerEm=2000`.
- If Illustrator vectors are drawn at 1000 UPEM, **scale imported outlines only** by 2×.

### 3.2 alternates 노출 방식 / Alternate exposure strategy

**KO**
- `ss01`: 기존 대상(A,B,E,F,K,M,P,Q,R,W)에 대한 1st alt(`.alt`) 토글
- `ss02`: 해당 글자에 2nd alt가 있을 때(`.alt2`) 토글 (현재는 M만)
- `salt`: 앱에서 “Stylistic Alternates” UI로 고르게 하기 위해, 존재하는 alternates 목록을 전부 나열
- O/V는 `salt`로만 노출하기로 결정 (ssXX에는 포함하지 않음)

**EN**
- `ss01`: toggles the 1st alternate (`.alt`) for the original set (A,B,E,F,K,M,P,Q,R,W)
- `ss02`: toggles the 2nd alternate (`.alt2`) when present (currently only M)
- `salt`: lists all alternates so apps can expose an alternates picker UI
- O/V are exposed via `salt` only (not included in ssXX)

### 3.3 AI 레이아웃 변경 대응 / Robust mapping when AI layout changes

**KO**
- 초기에는 “행/열 템플릿” 기반으로 매핑했으나, AI에서 추가 도형이 생기면 깨질 수 있음.
- 현재는 **베이스 글립(A–Q, R–Z) 행을 찾은 후**, 남는 도형을 **x축 근접도**로 가장 가까운 베이스 글립에 붙여서 `.alt`, `.alt2`…로 매핑.
- Illustrator/PDF→SVG 변환 시 하나의 글립이 여러 `<path>`로 쪼개지는 케이스를 고려해, **겹치는 path들은 한 글립으로 클러스터링**.

**EN**
- Initially mapping was a rigid row/column template; that breaks when new alternates are added.
- Now we detect base rows (A–Q, R–Z), then attach remaining shapes to the nearest base glyph by x proximity to produce `.alt`, `.alt2`, ...
- Illustrator/PDF→SVG may split a single glyph into multiple `<path>` elements; overlapping paths are clustered into one glyph.

---

## 4) FontGoggles에서 .notdef만 보이던 이유 / Why FontGoggles showed .notdef

**KO**
- UFO 글립에 `unicode` 매핑이 없으면 입력 문자→글립으로 매핑할 수 없어, 전부 `.notdef`로 떨어짐.
- 해결: 스크립트에 `--write-unicodes` 옵션을 추가해서 AGL/`uniXXXX` 이름 기반으로 유니코드를 복구.

**EN**
- If UFO glyphs have no Unicode values, there is no cmap mapping; typing falls back to `.notdef`.
- Fix: add `--write-unicodes` to restore unicodes from AGL/`uniXXXX` glyph names.

---

## 5) 재현 방법 / How to reproduce

### 5.1 벡터 주입 + features + unicode 복구

**KO**
- 최신 AI로부터 주입:
  - `./.venv/bin/python scripts/import_op_morph_ai.py --ai "Vector/Op_Morph_Flat edge.ai" --force-render --apply --write-features --write-unicodes --source-upem 1000`

**EN**
- Import from latest AI:
  - `./.venv/bin/python scripts/import_op_morph_ai.py --ai "Vector/Op_Morph_Flat edge.ai" --force-render --apply --write-features --write-unicodes --source-upem 1000`

### 5.2 OTF 빌드 (검증용)

**KO**
- `qcurve`가 섞여 있어 기본 overlap 제거 단계가 실패할 수 있음 → 검증용은 overlap 제거 없이:
  - `./.venv/bin/python -m fontmake -u OptimisticAI_Morph.ufo -o otf --keep-overlaps --output-dir build_otf`

**EN**
- Due to mixed segment types (e.g. `qcurve`), overlap removal may fail. For validation build without overlap removal:
  - `./.venv/bin/python -m fontmake -u OptimisticAI_Morph.ufo -o otf --keep-overlaps --output-dir build_otf`

---

## 6) 다음 웨이트 추가 준비 (스크립트) / Preparing for additional weights (script)

**KO**
- 다음 웨이트(Light/Bold 등) UFO를 만들면, 아래 스크립트로 한 번에:
  1) 네이밍 규칙에 맞게 `family/style/PS` 설정
  2) 기존 Morph UFO의 `features.fea`를 복사(피쳐 공유)
  3) 여러 소스를 묶는 `.designspace` 파일 생성/업데이트

**EN**
- When you create a new weight UFO (Light/Bold/etc.), use the helper script to:
  1) set naming fields (`family/style/PS`) consistently
  2) copy `features.fea` from the reference Morph UFO (feature sharing)
  3) create/update a `.designspace` that references multiple sources

스크립트: [scripts/prepare_morph_weight.py](scripts/prepare_morph_weight.py)

예시 / Example:
- `./.venv/bin/python scripts/prepare_morph_weight.py --in-ufo path/to/OptimisticAI_Light.ufo --out-ufo OptimisticAI_Morph-Light.ufo --style "Light Morph" --copy-features-from OptimisticAI_Morph.ufo --designspace OptimisticAI_Morph.designspace`

---

## 7) 네이밍 규칙 문서 / Naming rules doc

- [NAMING_RULES.md](NAMING_RULES.md)
