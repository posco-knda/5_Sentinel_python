"""
21. 엔진별 예측 구간·생존곡선 내보내기 — 대시보드 '엔진 상세'가 어느 엔진이든 열리도록

08번은 대시보드에 보여 줄 엔진 9대에 대해서만 90% 예측 구간과 생존곡선을 계산했습니다. 이 계산은 엔진이 아니라
"예측값 p" 하나에만 달려 있으므로(검증셋에서 예측이 p와 비슷했던 순간들의 실제 RUL 분포), 공식 test 248대
전부에 같은 방식으로 계산해 내보냅니다. 새로 학습하는 모델은 없고, 08번과 같은 상수·같은 시드를 씁니다.

  - 신뢰구간 = 검증셋(학습에 안 쓴 엔진 49대)에서 예측이 p±10 이내였던 순간들의 실제 RUL 5%~95% 구간
  - 생존곡선 = 그 순간들 중 실제 RUL이 x보다 컸던 비율 (x = 0, 10, 20, …), 음영은 검증 엔진 단위 부트스트랩 1000회의 5~95%
  - 08번이 엔진 9대에 낸 값과 소수점까지 같은지 이 스크립트가 직접 확인합니다 (같지 않으면 중단).

출력: outputs/<데이터셋>/dashboard_data/uncertainty.ts
      → 대시보드 레포의 src/data/uncertainty.ts 로 복사 (직접 고치지 말고 이 스크립트를 다시 돌릴 것)
"""

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from common import load_raw, OUT_METRICS, OUT_DASH, TRAIN_FILE

# 08번과 같은 값 (08번 상단 [임의 설정값 #9] 설명 참고)
NEIGHBOR_RADIUS = 10
MIN_NEIGHBORS = 30
INTERVAL_QUANTILES = (0.05, 0.95)
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
SURVIVAL_GRID = [0, 10, 20, 30, 40, 60, 80, 100, 120, 140]

summary = pd.DataFrame(json.load(open(f"{OUT_DASH}/engines_summary.json", encoding="utf-8"))).set_index("unit")

val = pd.read_csv(f"{OUT_METRICS}/gru_val_predictions.csv")
life = load_raw(TRAIN_FILE).groupby("unit")["cycle"].max()
val["RUL_raw"] = val["unit"].map(life) - val["cycle"]
val_units = np.sort(val["unit"].unique())


def neighbors(frame: pd.DataFrame, p: float) -> pd.DataFrame:
    dist = (frame["RUL_pred"] - p).abs()
    near = frame[dist <= NEIGHBOR_RADIUS]
    if len(near) < MIN_NEIGHBORS:
        near = frame.loc[dist.nsmallest(MIN_NEIGHBORS).index]
    return near


def survival(rul: np.ndarray) -> np.ndarray:
    return np.array([(rul > x).mean() for x in SURVIVAL_GRID])


rng = np.random.default_rng(BOOTSTRAP_SEED)
by_unit = {u: g for u, g in val.groupby("unit")}
boot_frames = [pd.concat([by_unit[u] for u in rng.choice(val_units, size=len(val_units), replace=True)]) for _ in range(BOOTSTRAP_N)]


def uncertainty(p: float):
    near = neighbors(val, p)["RUL_raw"].to_numpy()
    lo, hi = np.quantile(near, INTERVAL_QUANTILES)
    interval = {"median": round(p, 1), "lower": round(max(0.0, float(lo)), 1), "upper": round(float(hi), 1)}
    s_mid = survival(near)
    s_boot = np.array([survival(neighbors(bf, p)["RUL_raw"].to_numpy()) for bf in boot_frames])
    s_lo, s_hi = np.quantile(s_boot, INTERVAL_QUANTILES, axis=0)
    curve = [{"cycle": x, "median": round(float(m), 4), "lower": round(float(min(l, m)), 4), "upper": round(float(max(h, m)), 4)}
             for x, m, l, h in zip(SURVIVAL_GRID, s_mid, s_lo, s_hi)]
    return interval, curve


out = {}
cache = {}
for u in summary.index:
    p = round(float(summary.loc[u, "predicted_RUL"]), 1)
    if p not in cache:  # 예측값이 같으면 결과도 같다
        cache[p] = uncertainty(p)
    interval, curve = cache[p]
    out[int(u)] = {"iv": [interval["lower"], interval["upper"]], "s": [[c["median"], c["lower"], c["upper"]] for c in curve]}

# 08번이 만든 엔진 9대 값과 같은지 확인
mock = open(f"{OUT_DASH}/mock.ts", encoding="utf-8").read()
checked = 0
for m in re.finditer(r'"engine-(\d+)": \{', mock):
    u = int(m.group(1))
    block = mock[m.end(): m.end() + 6000]
    iv = re.search(r'"predictedRulCycle": \{\s*"median": ([\d.]+),\s*"lower": ([\d.]+),\s*"upper": ([\d.]+)', block)
    sv = re.search(r'"survivalCurve": \[(.*?)\n    \]', block, re.S)
    if not iv or u not in out or not sv:
        continue
    mock_curve = [float(x) for x in re.findall(r'"median": ([\d.]+)', sv.group(1))]
    mine = [x[0] for x in out[u]["s"]]
    if abs(out[u]["iv"][0] - float(iv.group(2))) > 1e-6 or abs(out[u]["iv"][1] - float(iv.group(3))) > 1e-6 or mock_curve != mine:
        raise SystemExit(f"엔진 {u}: 08번(mock.ts) 값과 달라졌습니다 — 구간 {out[u]['iv']} vs {iv.groups()[1:]}, 곡선 일치 {mock_curve == mine}")
    checked += 1
if checked == 0:
    raise SystemExit("mock.ts에서 비교할 엔진을 찾지 못했습니다 — 08번을 먼저 실행하세요")

stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
code = f"""// ⚠️ 자동 생성 파일 — 분석 레포의 scripts/21_export_uncertainty.py 가 만든다. 직접 고치지 말고 파이썬을 다시 돌려서 받아올 것.
// 엔진별 90% 예측 구간과 생존곡선 (검증셋에서 예측이 비슷했던 순간들의 분포, 08번과 같은 방식·같은 시드) (생성: {stamp}).
// iv = [하한, 상한] (cycle), s = 생존곡선 [중앙, 하한, 상한] × x = {SURVIVAL_GRID}

export const SURVIVAL_X = {json.dumps(SURVIVAL_GRID)} as const

export const uncertainty: Record<number, {{ iv: [number, number]; s: [number, number, number][] }}> = {json.dumps(out, separators=(",", ":"))}
"""
with open(f"{OUT_DASH}/uncertainty.ts", "w", encoding="utf-8") as f:
    f.write(code)
print(f"엔진 {len(out)}대 (서로 다른 예측값 {len(cache)}개), 08번 값과 일치 확인 {checked}대")
