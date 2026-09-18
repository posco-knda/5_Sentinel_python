# Sentinel : 냉간압연 설비 이상탐지 & 잔존수명 예측

**5조 · 팀명 "감시자들"** — POSCO 연계 "스마트 정비를 위한 설비 데이터 분석" 수업 통합 프로젝트 (주제 G · 자율주제)

## 프로젝트 소개

냉간압연기(TCM, 5단 텐덤 압연기)의 공정 데이터로 전동기·베어링·워크롤 이상을 탐지하고, 워크롤의 잔존수명(RUL)을 예측한 뒤, 이를 정비 비용 시뮬레이션(AS-IS vs TO-BE)으로 연결하는 프로젝트입니다.

## 데이터

- 출처: Zenodo 공개 예지보전 벤치마크 데이터셋 「TCM: Benchmark Datasets for Predictive Maintenance in Steel Manufacturing」
- DOI: [10.5281/zenodo.11469702](https://doi.org/10.5281/zenodo.11469702) (2026 Frontiers in Artificial Intelligence 게재, CC-BY-4.0)
- `tcm5_dataset_1.csv` ~ `tcm5_dataset_6.csv` 6개 파일을 위 DOI 페이지에서 받아 `data/` 폴더에 넣어주세요 (용량 문제로 git에는 커밋하지 않습니다).

## 환경 설정

Python 3.10 이상을 권장합니다.

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

라이브러리를 새로 추가했다면(`pip install XXX`), 다른 사람도 같은 버전을 쓸 수 있게 `requirements.txt`에 한 줄 추가해서 같이 커밋해주세요.

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
requirements.txt 분석 라이브러리 목록 (pip install -r requirements.txt)
data/            원본 CSV (직접 다운로드, git 미포함) — data/README.md 참고
notebooks/       01_eda / 02_timeseries / 03_model / 04_eval (각자 새로 만들어서 작업)
src/             preprocess.py / features.py / model.py (각자 새로 만들어서 작업)
figures/         분석 결과 그래프
reports/         문제정의서 / 결과보고서 / 발표자료
```

노트북·스크립트 파일은 아직 비어있는 채로 올려두지 않았습니다 — 위 "실행 순서"에 있는 파일명 그대로, 각자 담당 파트에서 새로 만들어서 작업하면 됩니다.

## 팀원 (공식 5역할)

**영준 · 팀장(PM)**
- 담당: 일정 관리, 강사님과 소통, 주제 승인서(G) 제출, 발표 총괄
- 이메일:
- 깃허브:
- 하고 싶은 말:

**형희 · 데이터 담당**
- 담당: 6개 CSV 구조 파악, 전처리, 워크롤 마일리지 리셋(수명주기) 탐지
- 이메일:
- 깃허브:
- 하고 싶은 말:

**규인 · 시계열·특성 담당**
- 담당: 공정변수 추이 시각화, 이상탐지·RUL용 특성 생성
- 이메일: [choi390900@gmail.com](mailto:choi390900@gmail.com)
- 깃허브: choi390900-commits
- 하고 싶은 말: 10월 20일 ~ 10월 22일 예비군

**민우 · 모델링 담당**
- 담당: 이상탐지·잔존수명(RUL)·비용 시뮬레이션 모델링, 대시보드 연동
- 이메일:
- 깃허브:
- 하고 싶은 말:

**문용 · 해석·문서 담당**
- 담당: 모델 평가, 오탐/미탐 사례 분석, 결과보고서·README 작성
- 이메일: [show4336@nate.com](mailto:show4336@nate.com)
- 깃허브: show4336-collab
- 하고 싶은 말:

## 관련 저장소

- 대시보드(시연용, 제출물 아님): [`5_Sentinel_dashboard`](https://github.com/posco-knda/5_Sentinel_dashboard)
