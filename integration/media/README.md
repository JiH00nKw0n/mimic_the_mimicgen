# 데모 영상 — SART · CP-Gen: source ↔ synthetic

두 방법론의 **원본 데모(source)** 와 **합성 데이터(synthetic)** 를 나란히 둔 것.
`integration/synthgen` 이 각 방법의 어느 부분을 재현하는지도 함께 표기했다.

> 재생: 로컬 마크다운 프리뷰(VS Code 등)에서는 아래 `<video>` 가 바로 재생된다.
> GitHub 웹에서는 파일 링크를 클릭하면 열린다. 파일을 직접 열려면: `open <경로>`.
> 여기 담은 건 **대표 컷**이고, 전체 세트·프로젝트 페이지 영상은
> `robot_data workspace — augmentation_methods/*/videos/` 에 있다.

---

## 1. CP-Gen  (object-centric + geometry 변환 → cuRobo 모션플랜 → 성공필터)

우리 파이프라인 대응: `synthgen/cpgen_transform.py` (transform) + `synthgen/pipeline.py`

### Source — 원본 데모 (증강 전)
`transform_mode` 가 재투영할 씨앗 궤적. 여기서 물체 pose·geometry 를 샘플해 대량 생성한다.

<video src="cpgen/source/cpgen_live_source.mp4" controls width="480"></video>

- [cpgen/source/cpgen_live_source.mp4](cpgen/source/cpgen_live_source.mp4) · 12s · 실제 소스 데모

### Synthetic — 합성 데이터 (생성된 궤적)
아래 3개가 우리가 노리는 **contact-rich 조립** 계열. 삽입 구간 tolerance 를 유지한 채
pose(+geometry)만 바뀐 변형들이다.

| peg-in-hole (SquareWide) | 3-piece assembly | threading |
|---|---|---|
| <video src="cpgen/synthetic/cpgen_gen_SquareWide_peg-insertion.mp4" controls width="260"></video> | <video src="cpgen/synthetic/cpgen_gen_ThreePieceAssembly.mp4" controls width="260"></video> | <video src="cpgen/synthetic/cpgen_gen_Threading.mp4" controls width="260"></video> |
| [파일](cpgen/synthetic/cpgen_gen_SquareWide_peg-insertion.mp4) · 6s | [파일](cpgen/synthetic/cpgen_gen_ThreePieceAssembly.mp4) · 6s | [파일](cpgen/synthetic/cpgen_gen_Threading.mp4) · 6s |

**볼 포인트:** 같은 스킬이 다양한 물체 배치/크기에서 재생성되지만, 삽입 순간의 정렬은 유지됨
(= `cpgen_transform.py` 가 insert 구간에 좁은 범위 + scale 고정을 거는 이유).

---

## 2. SART / RoboManipAug  (정밀도 구 안에서 국소 self-augmentation)

우리 파이프라인 대응: `synthgen/sart_augmentor.py` (insert 스킬 국소 증강)

### Source — 사람 teleop 데모 1개
Insert(peg-in-hole) 태스크의 단일 시연. SART 는 이 하나에서 시작한다.

<video src="sart/source/robomanipaug_source_teleop_insert.mp4" controls width="480"></video>

- [sart/source/robomanipaug_source_teleop_insert.mp4](sart/source/robomanipaug_source_teleop_insert.mp4) · 16s · 원본 teleop

### Synthetic — 경계 안 self-augmented 궤적들
삽입 waypoint 주변 구(sphere)에서 샘플해 수렴시킨 다양한 궤적. 원본 1개가 다수로 늘어난다.

<video src="sart/synthetic/robomanipaug_augmented_insert.mp4" controls width="480"></video>

- [sart/synthetic/robomanipaug_augmented_insert.mp4](sart/synthetic/robomanipaug_augmented_insert.mp4) · 98s · 국소 증강 결과

**볼 포인트:** 삽입점으로 수렴하는 여러 접근 궤적. 우리 재구현에서는 이 "안전 구"를
cuRobo 충돌체크(point cloud)로 대체해 더 강하게 만든다(`sart_augmentor.py` + `CuroboCollision`).

---

## 요약 매핑

| 방법 | source 영상 | synthetic 영상 | synthgen 모듈 |
|---|---|---|---|
| CP-Gen | live_source | SquareWide / ThreePieceAssembly / Threading | `cpgen_transform.py` + `pipeline.py` |
| SART | teleop_insert | augmented_insert | `sart_augmentor.py` |

출처: CP-Gen https://cp-gen.github.io · SART https://sites.google.com/view/sart-il
</content>
