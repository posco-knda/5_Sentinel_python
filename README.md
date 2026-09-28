# Sentinel : 터보팬 엔진 고장 임박 예측

**5조 · 팀명 "감시자들"** — POSCO 연계 "스마트 정비를 위한 설비 데이터 분석" 수업 통합 프로젝트 (주제 F · 터보팬 엔진 고장 임박 예측 )

## 프로젝트 소개

NASA C-MAPSS 터보팬 엔진의 운전 조건 및 센서 데이터를 활용하여 엔진별 열화 패턴을 분석하고, **잔여 유효 수명(RUL, Remaining Useful Life)**을 기반으로 고장 임박 여부를 예측하는 프로젝트입니다.

## 데이터

- 출처: NASA PCoE — C-MAPSS Turbofan Engine Degradation Simulation Dataset
- 데이터: CMAPSSData.zip (FD001 ~ FD004 포함, 본 프로젝트는 FD001 사용)  https://zenodo.org/records/15346912
- train_FD001.txt · test_FD001.txt · RUL_FD001.txt 파일을 data/ 폴더에 넣어주세요 (용량 문제로 git에는 커밋하지 않습니다).

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
data/            C-MAPSS 원본 데이터 (직접 다운로드, git 미포함) — data/README.md 참고
notebooks/       01_eda / 02_rul / 03_timeseries / 04_model / 05_eval (각자 새로 만들어서 작업)
src/             preprocess.py / features.py / model.py / evaluation.py (각자 새로 만들어서 작업)
figures/         분석 결과 그래프
reports/         문제정의서 / 결과보고서 / 발표자료
```

노트북·스크립트 파일은 아직 비어있는 채로 올려두지 않았습니다 — 위 "실행 순서"에 있는 파일명 그대로, 각자 담당 파트에서 새로 만들어서 작업하면 됩니다.

## 팀원 (공식 5역할)

**영준 · 팀장(PM)**
- 담당: 일정 관리, 강사님과 소통, 주제 승인서(F) 제출, 발표 총괄
- 이메일:
- 깃허브:
- 하고 싶은 말:

**형희 · 데이터 담당**
- 담당: C-MAPSS 데이터 구조 파악, 전처리, 엔진별 운전 이력 정리, RUL 계산 및 위험 라벨 생성
- 이메일:
- 깃허브:
- 하고 싶은 말:

**규인 · 시계열·특성 담당**
- 담당: 센서별 시계열 변화 시각화, 불필요 센서 선별, 이동평균·변화량·변화율 등 특성 생성
- 이메일: [choi390900@gmail.com](mailto:choi390900@gmail.com)
- 깃허브: choi390900-commits
- 하고 싶은 말: 10월 20일 ~ 10월 22일 예비군

**민우 · 모델링 담당**
- 담당: 고장 임박 이진 분류, RUL 회귀 모델링, 위험 임계값별 성능 비교 및 모델 최적화
- 이메일:
- 깃허브:
- 하고 싶은 말:

**문용 · 해석·문서 담당**
- 담당: 모델 평가, 오탐/미탐 사례 분석, 분류·RUL 회귀 결과 비교, 결과보고서, README 작성
- 이메일: [show4336@nate.com](mailto:show4336@nate.com)
- 깃허브: show4336-collab
- 하고 싶은 말:

## 관련 저장소

- 대시보드(시연용, 제출물 아님): [`5_Sentinel_dashboard`](https://github.com/posco-knda/5_Sentinel_dashboard)
