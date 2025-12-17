# t 컨텍스츄얼 시퀀스 메모

## 목표
- 기본 `t` 하나 입력 시 기본 `t` 유지.
- `tt` 입력 시 두 번째 글자를 `t.c2`로 치환.
- `ttt` 입력 시 `t`, `t.c1`, `t.c2` 순서로 출력.
- `t` 4개 이상 입력 시 `t`, `t.c1` 반복(길이-2), 마지막은 `t.c2`.
- 위 규칙을 다른 자모에도 확장 예정.

## OpenType `calt` 구성
```fea
@t_base = [t];
@t_mid = [t.caltMid];   # 반복 구간
@t_end = [t.caltEnd];   # 마감용

lookup t_pairs {
  sub t' t by t.caltEnd;
} t_pairs;

lookup t_triples {
  sub t' t' t by t t.caltMid t.caltEnd;
} t_triples;

lookup t_runs {
  lookupflag IgnoreMarks;
  # 4개 이상 연속 시 가운데는 mid, 마지막은 end
  sub t' t' t' t' by t t.caltMid t.caltMid t.caltEnd;
  sub t' t' t' t' @t_base by t t.caltMid t.caltMid t.caltMid t.caltEnd;
  # 필요 시 더 긴 시퀀스를 추가
} t_runs;
```
> Glyphs에서는 `File > Font Info > Features > calt`에 위와 같이 클래스와 substitution을 정의하면 자동으로 LookupType 6 (Chain Contextual Substitution)을 생성합니다.

## 작명 규칙
- 기본 형태: `baseName.suffix` (예: `t.caltMid`, `t.caltEnd`).
- 기능을 드러내는 suffix 권장: `.calt1`, `.calt2`, `.alt`, `.ss01` 등.
- 여러 글자에 공통 규칙을 적용하려면 suffix를 통일해 스크립트가 패턴 매칭 가능하도록 유지.

## 여러 글자 확장 팁
- 예: `@runTargets = [t l f ...];` / `@runTargetsMid = [t.caltMid l.caltMid ...];` 형태로 짝을 만들어 재사용.
- `lookupflag IgnoreMarks;`로 마크 간섭을 무시하여 연속성 유지.
- 필요 시 `tt` 외 길이 케이스를 `calt` 외 `liga`, `rlig` 등 다른 피처로도 분산 가능.

## 기타 메모
- t.c1, t.c2 같은 임시 이름 대신 `.caltMid`, `.caltEnd`처럼 용도를 표현하는 이름을 쓰면 유지보수 및 스크립트 제작이 용이.
- 동일 규칙을 다른 Glyph에도 적용할 때, 스크립트는 suffix로 대상을 찾고 복제하도록 작성하면 효율적.
```
python
for glyph in targets:
    font.generateContextualAlternates(glyph, suffix="caltMid", repeatSuffix="caltEnd")
```
- 마크가 포함된 스크립트(예: 한글 초성/중성)에서는 IgnoreMarks 외에 필요한 경우 `lookupflag MarkAttachmentType` 사용 고려.
