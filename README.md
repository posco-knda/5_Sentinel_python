# Sentinel : 냉간압연 설비 이상탐지 & 잔존수명 예측

**5조 · 팀명 "감시자들"** — POSCO 연계 "스마트 정비를 위한 설비 데이터 분석" 수업 통합 프로젝트 (주제 G · 자율주제)

## 프로젝트 소개

냉간압연기(TCM, 5단 텐덤 압연기)의 공정 데이터로 전동기·베어링·워크롤 이상을 탐지하고, 워크롤의 잔존수명(RUL)을 예측한 뒤, 이를 정비 비용 시뮬레이션(AS-IS vs TO-BE)으로 연결하는 프로젝트입니다.

## 데이터

- 출처: Zenodo 공개 예지보전 벤치마크 데이터셋 「TCM: Benchmark Datasets for Predictive Maintenance in Steel Manufacturing」
- DOI: [10.5281/zenodo.11469702](https://doi.org/10.5281/zenodo.11469702) (2026 Frontiers in Artificial Intelligence 게재, CC-BY-4.0)
- `tcm5_dataset_1.csv` ~ `tcm5_dataset_6.csv` 6개 파일을 위 DOI 페이지에서 받아 `data/` 폴더에 넣어주세요 (용량 문제로 git에는 커밋하지 않습니다).

## 실행 순서

```
notebooks/01_eda.ipynb          데이터 구조 파악
notebooks/02_timeseries.ipynb   시계열 패턴 분석, 특성 생성
notebooks/03_model.ipynb        이상탐지·RUL 모델링
notebooks/04_eval.ipynb         평가·오류 사례 분석
```

## 폴더 구조

```
README.md
data/            원본 CSV (직접 다운로드, git 미포함)
notebooks/       01_eda / 02_timeseries / 03_model / 04_eval
src/             preprocess.py / features.py / model.py
figures/         분석 결과 그래프
reports/         문제정의서 / 결과보고서 / 발표자료
```

## 팀원 (공식 5역할)

| 역할 | 담당 |
|---|---|
| 팀장(PM) | 문용 |
| 데이터 담당 | 형희 |
| 시계열·특성 담당 | 민우 |
| 모델링 담당 | 영준 |
| 해석·문서 담당 | 규인 |

## 관련 저장소

- 대시보드(시연용, 제출물 아님): `sentinel-dashboard`
