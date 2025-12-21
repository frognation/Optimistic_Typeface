# Optimistic AI Morph – Naming Rules (KO/EN)

## 1) 결론 / TL;DR

**KO**
- `Morph`는 새로운 **패밀리(family)** 로 분리하기보다, 기존 `Optimistic AI` 패밀리 안에서 **스타일(서브패밀리)** 로 운용하는 것을 권장합니다.
- 웨이트(라이트/볼드 등)를 추가할 경우, **패밀리명은 고정**하고 스타일명을 `Light Morph`, `Bold Morph`처럼 **웨이트 + Morph 조합**으로 확장하는 방식이 가장 확장성이 좋습니다.

**EN**
- Prefer keeping `Morph` as a **style/subfamily** inside the existing `Optimistic AI` family, rather than creating a separate family named `Optimistic AI Morph`.
- When adding more weights (Light/Bold/etc.), keep the **family name constant** and expand styles as `Light Morph`, `Bold Morph`, etc. (Weight + Morph).

---

## 2) 추천 네이밍 규칙 / Recommended naming scheme

### Family / Style

**KO (권장)**
- Family: `Optimistic AI`
- Style(Subfamily):
  - `Morph` (단일 웨이트만 있을 때)
  - `Light Morph`, `Regular Morph`, `Medium Morph`, `Bold Morph` … (여러 웨이트 확장)

**EN (Recommended)**
- Family: `Optimistic AI`
- Style/Subfamily:
  - `Morph` (when only one weight exists)
  - `Light Morph`, `Regular Morph`, `Medium Morph`, `Bold Morph` … (when multiple weights exist)

### PostScript name

**KO**
- 공백 없이: `OptimisticAI-Morph`, `OptimisticAI-LightMorph`, `OptimisticAI-BoldMorph` …

**EN**
- No spaces: `OptimisticAI-Morph`, `OptimisticAI-LightMorph`, `OptimisticAI-BoldMorph` …

---

## 3) UFO에서 어떤 필드를 어떻게 맞추나 / Which UFO fields to set

**KO**
아래 필드들은 [OptimisticAI_Morph.ufo/fontinfo.plist](OptimisticAI_Morph.ufo/fontinfo.plist) 에 존재합니다.
- `familyName`: `Optimistic AI`
- `styleName`: 예) `Morph`, `Light Morph`
- `openTypeNamePreferredSubfamilyName`: 예) `Morph`, `Light Morph`
- `postscriptFontName`: 예) `OptimisticAI-Morph`, `OptimisticAI-LightMorph`
- `styleMapFamilyName`: `Optimistic AI`
- `styleMapStyleName`: 기본은 `regular` 유지 (스타일 링크를 강하게 원하면 별도 설계 필요)

**EN**
These fields are in [OptimisticAI_Morph.ufo/fontinfo.plist](OptimisticAI_Morph.ufo/fontinfo.plist).
- `familyName`: `Optimistic AI`
- `styleName`: e.g. `Morph`, `Light Morph`
- `openTypeNamePreferredSubfamilyName`: e.g. `Morph`, `Light Morph`
- `postscriptFontName`: e.g. `OptimisticAI-Morph`, `OptimisticAI-LightMorph`
- `styleMapFamilyName`: `Optimistic AI`
- `styleMapStyleName`: keep `regular` by default (style-linking needs a separate decision)

---

## 4) 왜 이 방식이 확장에 유리한가 / Why this scales better

**KO**
- 웨이트가 늘어날수록 “패밀리로 분리”는 관리가 복잡해지고(메뉴 구조/스타일 링크/변수축 설계), 유지보수 비용이 커집니다.
- 패밀리를 `Optimistic AI`로 유지하면, 폰트 메뉴에서 한 패밀리 안에 웨이트/스타일이 자연스럽게 확장됩니다.

**EN**
- Splitting into a separate family (`Optimistic AI Morph`) can make long-term maintenance harder (menu grouping, style linking, variable-axis planning).
- Keeping a single family (`Optimistic AI`) scales cleanly as you add more styles/weights.

---

## 5) 다음 웨이트 추가 시 작업 흐름 / Workflow when adding a new weight

**KO**
- 새 웨이트 UFO를 준비한 뒤, 아래 스크립트를 사용해:
  - 네이밍 필드 일괄 세팅
  - `features.fea`(salt/ssXX 등) 공유
  - `.designspace` 생성/업데이트

**EN**
- After you have a new weight UFO, run the helper script to:
  - set naming fields consistently
  - share `features.fea` (salt/ssXX, etc.)
  - create/update a `.designspace`

관련 스크립트: [scripts/prepare_morph_weight.py](scripts/prepare_morph_weight.py)
