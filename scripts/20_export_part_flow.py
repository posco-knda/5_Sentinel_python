"""
20. 부품별 센서 이탈도 내보내기 — 대시보드 '고장 진단 플로우'용

"어떤 부품 쪽 센서가 얼마나 달라졌고, 모델이 그걸 언제 알아챘는가"를 보여 주기 위한 자료입니다.
공식 test 엔진 248대 각각의 마지막 관측 시점에서, 센서 15개가 그 엔진의 초기 상태에서 얼마나 벗어났는지 계산하고
센서가 측정하는 위치(Saxena 외 2008, Table 2)에 따라 6개 부품 묶음으로 모읍니다.

계산 (17번 '엔진별 초기 대비 변화량'과 같은 기준: 처음 30사이클 평균 = 그 엔진의 초기 상태)
  - 변화량 = 최근 15사이클 평균 - 처음 30사이클 평균
  - 이탈도 z = 변화량 / σ  (σ = 엔진들의 '처음 30사이클 안의 표준편차' 중앙값 = 정상 상태에서 흔들리는 폭)
    → "평소 흔들림의 몇 배만큼 변했는가". 부호는 올랐으면 +, 내려갔으면 -.
  - 부품 이탈도 = 그 부품에 속한 센서들의 |z| 평균 (크기만 — 센서가 많은 묶음이 최댓값에서 유리해지는 편향을 피함)
  - 관측이 45사이클 미만인 엔진은 초기·최근 구간이 겹치지 않도록 줄여서 계산하고 short=true로 표시 (참고용)

주의 — 이것은 고장 원인 판정이 아닙니다.
  - C-MAPSS FD004에는 엔진별 고장 모드(팬 열화 / 고압압축기 열화) 정답이 없어서, "가장 크게 변한 센서가 속한 부품"까지만
    말할 수 있습니다. 그 부품이 실제 고장 원인이라고 검증한 값이 아닙니다 ("관련 부품 추정").
  - 센서 → 부품 묶음은 센서 이름이 가리키는 측정 위치를 근거로 한 팀의 해석입니다.
  - σ는 test 엔진의 초기 구간(정답 RUL을 쓰지 않음)에서 구한 기술 통계이며 모델 학습·평가에는 쓰이지 않습니다.

출력: outputs/<데이터셋>/dashboard_data/part_flow_meta.ts, part_flow.ts
      → 대시보드 레포의 src/data/partFlowMeta.ts, partFlow.ts 로 복사 (직접 고치지 말고 이 스크립트를 다시 돌릴 것)
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from common import OUT_DASH, SENSOR_MEANING, display_unit, display_values, RISK_THRESHOLDS

BASELINE_CYCLES = 30  # 17번과 같음
RECENT_CYCLES = 15
TREND_POINTS = 36      # 화면용 추이는 이 개수로 줄여서 저장

# 센서가 측정하는 위치 기준 부품 묶음 (Saxena 외 2008, Table 2의 센서 이름 근거 — 팀의 해석)
PARTS = [
    ("fan", "팬·바이패스", ["s8", "s13", "s15", "s6"]),
    ("lpc", "저압압축기(LPC)", ["s2"]),
    ("hpc", "고압압축기(HPC)", ["s3", "s7", "s11"]),
    ("core", "코어·연소·블리드", ["s9", "s14", "s12", "s17"]),
    ("hpt", "고압터빈(HPT)", ["s20"]),
    ("lpt", "저압터빈(LPT)", ["s4", "s21"]),
]
PART_OF = {s: pid for pid, _, ss in PARTS for s in ss}

summary = pd.DataFrame(json.load(open(f"{OUT_DASH}/engines_summary.json", encoding="utf-8"))).set_index("unit")
sensors = [s for _, _, ss in PARTS for s in ss]

# 1. 엔진 데이터 읽기
series = {}
for u in summary.index:
    ts = json.load(open(f"{OUT_DASH}/engine_timeseries/{u}.json", encoding="utf-8"))
    missing = [s for s in sensors if s not in ts["sensors"]]
    if missing:
        raise SystemExit(f"엔진 {u}: 센서 {missing} 가 engine_timeseries 에 없습니다 (07번을 다시 돌려 보세요)")
    series[int(u)] = {"c": np.asarray(ts["cycles"], dtype=int), **{s: np.asarray(ts["sensors"][s], dtype=float) for s in sensors}}

# 2. σ: 처음 BASELINE_CYCLES 안에서 센서가 흔들리는 폭의 중앙값 (관측이 충분한 엔진만)
sigma = {}
for s in sensors:
    stds = [np.std(d[s][:BASELINE_CYCLES], ddof=1) for d in series.values() if len(d["c"]) >= BASELINE_CYCLES]
    sigma[s] = float(np.median(stds))
    if sigma[s] <= 0:
        raise SystemExit(f"센서 {s} 의 σ가 0 이하입니다")

# 3. 엔진별 이탈도
flow = {}
for u, d in series.items():
    n = len(d["c"])
    k0 = min(BASELINE_CYCLES, n // 2)
    k1 = min(RECENT_CYCLES, n - k0)
    short = n < BASELINE_CYCLES + RECENT_CYCLES
    z = {s: float((d[s][-k1:].mean() - d[s][:k0].mean()) / sigma[s]) for s in sensors}
    part = {pid: float(np.mean([abs(z[s]) for s in ss])) for pid, _, ss in PARTS}
    top = sorted(sensors, key=lambda s: -abs(z[s]))[:3]
    idx = np.unique(np.linspace(0, n - 1, min(TREND_POINTS, n)).round().astype(int))
    trend = [{"s": s, "c": [int(d["c"][i]) for i in idx], "v": [round(float(display_values(s, d[s][i])), 2) for i in idx],
              "base": round(float(display_values(s, d[s][:k0].mean())), 2)} for s in top]
    flow[u] = {"z": {s: round(z[s], 1) for s in sensors}, "part": {p: round(v, 1) for p, v in part.items()},
               "top": top, "trend": trend, "short": bool(short)}

# 4. 등급별 부품 이탈 프로필 + 위험 엔진에서 가장 크게 변한 부품 분포
level_of = summary["risk_level"].map({"RED": "red", "YELLOW": "warn", "GREEN": "ok"})
profile = {}
for lv in ("red", "warn", "ok"):
    units = [u for u in flow if level_of[u] == lv and not flow[u]["short"]]
    profile[lv] = {pid: round(float(np.mean([flow[u]["part"][pid] for u in units])), 1) for pid, _, _ in PARTS}
    profile[lv]["n"] = len(units)
red_units = [u for u in flow if level_of[u] == "red" and not flow[u]["short"]]
top_part = {}
for u in red_units:
    p = max(flow[u]["part"], key=lambda k: flow[u]["part"][k])
    top_part[p] = top_part.get(p, 0) + 1

meta = {
    "baselineCycles": BASELINE_CYCLES, "recentCycles": RECENT_CYCLES,
    "parts": [{"id": pid, "label": label, "sensors": [{"id": s, "name": SENSOR_MEANING[s], "unit": display_unit(s)} for s in ss]}
              for pid, label, ss in PARTS],
    "sigma": {s: round(sigma[s], 3) for s in sensors},
    "redTopPart": top_part, "redEngines": len(red_units), "profile": profile,
    "red": RISK_THRESHOLDS["red_at_or_below"], "yellow": RISK_THRESHOLDS["yellow_below"],
}
J = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
head = f"""// ⚠️ 자동 생성 파일 — 분석 레포의 scripts/20_export_part_flow.py 가 만든다. 직접 고치지 말고 파이썬을 다시 돌려서 받아올 것.
// 공식 test 엔진의 센서 15개가 초기 상태(처음 {BASELINE_CYCLES}사이클)에서 얼마나 벗어났는지(이탈도 z)와 부품별 요약 (생성: {stamp}).
// 고장 원인 판정이 아니라 "가장 크게 변한 센서가 속한 부품(추정)"이다 — 분석 레포 scripts/20_export_part_flow.py 상단 설명 참고.

"""
meta_code = head + f"""export interface PartMeta {{
  id: string
  label: string
  sensors: {{ id: string; name: string; unit: string }}[]
}}

export const partMeta: {{
  baselineCycles: number
  recentCycles: number
  parts: PartMeta[]
  sigma: Record<string, number>
  redTopPart: Record<string, number>
  redEngines: number
  profile: Record<'red' | 'warn' | 'ok', Record<string, number>>
  red: number
  yellow: number
}} = {J(meta)}
"""
with open(f"{OUT_DASH}/part_flow_meta.ts", "w", encoding="utf-8") as f:
    f.write(meta_code)

flow_code = head + f"""export interface PartFlow {{
  /** 센서별 이탈도 z (평소 흔들림의 몇 배 변했는가, 부호 = 방향) */
  z: Record<string, number>
  /** 부품별 이탈도 = 그 부품 센서들의 |z| 평균 (크기만) */
  part: Record<string, number>
  /** |z| 상위 센서 3개 */
  top: string[]
  /** 상위 센서의 추이(최대 {TREND_POINTS}점)와 초기 평균 */
  trend: {{ s: string; c: number[]; v: number[]; base: number }}[]
  /** 관측이 {BASELINE_CYCLES + RECENT_CYCLES}사이클 미만이라 초기·최근 구간을 줄여 계산한 엔진 (참고용) */
  short: boolean
}}

export const partFlow: Record<number, PartFlow> = {J(flow)}
"""
with open(f"{OUT_DASH}/part_flow.ts", "w", encoding="utf-8") as f:
    f.write(flow_code)

print(f"엔진 {len(flow)}대 계산 (관측 짧은 엔진 {sum(v['short'] for v in flow.values())}대)")
print("위험 엔진에서 가장 크게 변한 부품:", {k: v for k, v in sorted(top_part.items(), key=lambda kv: -kv[1])}, f"/ {len(red_units)}대")
print("부품 이탈 프로필(평균 |z|):", {lv: {k: v for k, v in p.items() if k != 'n'} for lv, p in profile.items()})
