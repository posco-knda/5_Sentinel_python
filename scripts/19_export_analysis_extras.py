"""
19. 분석 근거 내보내기 — 대시보드 '경보 기준' · '모델·근거' 화면용

이미 만들어진 결과 파일(metrics/*.csv·json)을 한곳에 모아 TypeScript 하나로 내보냅니다. 새로 학습하는 모델은 없습니다.
유일한 새 계산은 GRU 경보 기준 N=20~80을 1씩 바꿔 가며 13·15번과 같은 정비 시뮬레이션(common.simulate_maintenance)을 돌리는 것으로,
15번이 5 간격으로만 저장해 둔 비용을 모든 N에서 같은 방식으로 이어 붙이기 위해서입니다.

경보 기준 5가지 안 (발표 17번 슬라이드와 같은 구성 — 값은 결과 파일에서 규칙으로 뽑고 코드에 숫자를 박지 않음)
  A  GRU 예측 RUL ≤ 위험 기준(common.DANGER_RUL)                         비용 최우선
  B  GRU 예측 RUL ≤ N, RF(확률 기준)와 재현율이 비슷해지는 N              절충   (15번 summary의 GRU_matched_to_RF_recall)
  C  GRU 예측 RUL ≤ N, 공식 test 놓침이 0이 되는 가장 작은 N             안전 최우선
  D  RF 분류 확률 ≥ 09번이 고른 값                                         RF 현재 설정
  E  RF 분류 확률 ≥ p, 공식 test 놓침이 0이 되는 가장 큰 p               RF 놓침 0 (최대 기준)
  · 놓침·헛경보는 공식 test(위험 69대·안전 179대), 정비 전 고장·비용·낭비는 validation 49대 정비 시뮬레이션입니다.
  · 비용 = 운행 1,000사이클당(계획 정비 1회 = 1), 낭비 = 정비 시점에 아직 남아 있던 평균 수명(사이클).

출력: outputs/<데이터셋>/dashboard_data/analysis_extras.ts, analysis_extras.json
      → 대시보드 레포의 src/data/analysisExtras.ts 로 복사 (직접 고치지 말고 이 스크립트를 다시 돌릴 것)
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from common import (
    DANGER_RUL, OUT_METRICS, OUT_DASH, TRAIN_FILE, load_raw, simulate_maintenance,
)

COST_RATIOS = {"1:5": 5.0, "1:2": 2.0}
N_RANGE = range(20, 81)  # GRU 경보 기준 스윕 (1 간격)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(f"{OUT_METRICS}/{name}", encoding="utf-8-sig")


def read_json(name: str):
    with open(f"{OUT_METRICS}/{name}", encoding="utf-8") as f:
        return json.load(f)


def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


# 1. GRU 스윕: 공식 test 탐지·놓침·헛경보 (저장된 값) + 정비 시뮬레이션 비용 (N마다 새로 계산)
sweep = read_csv("alert_comparison_test_sweep.csv")
gru_sw = sweep[sweep.method == "GRU"].set_index("setting")
rf_sw = sweep[sweep.method == "RF 분류"].copy()

gru_val = pd.read_csv(f"{OUT_METRICS}/gru_val_predictions.csv")
life = load_raw(TRAIN_FILE).groupby("unit")["cycle"].max()


def gru_alarm(n: int) -> dict:
    out = {}
    for u, g in gru_val.groupby("unit"):
        hit = g.loc[g["RUL_pred"] <= n, "cycle"]
        out[int(u)] = int(hit.iloc[0]) if len(hit) else None
    return out


gru_rows = []
for n in N_RANGE:
    row = gru_sw.loc[float(n)]
    alarm = gru_alarm(n)
    sims = {k: simulate_maintenance(alarm, life, cf) for k, cf in COST_RATIOS.items()}
    gru_rows.append({
        "n": n, "tp": int(row.TP), "fn": int(row.FN), "fp": int(row.FP),
        "cost15": r(sims["1:5"]["cost_per_1000_cycles"]), "cost12": r(sims["1:2"]["cost_per_1000_cycles"]),
        "fail": r(sims["1:5"]["failure_rate"], 4), "wasted": r(sims["1:5"]["avg_wasted_cycles"], 1),
    })
gru_df = pd.DataFrame(gru_rows).set_index("n")

# 2. 저장된 15번 비용 중 GRU 값과 새로 계산한 값이 같은지 확인 (같은 정책·시드를 쓰는지 검증)
saved = read_csv("alert_comparison_cost.csv")
chk = saved[(saved.method == "GRU") & (saved.cost_ratio == "1:5")].copy()
chk["n"] = chk.setting.str.extract(r"(\d+)").astype(int)
max_gap = float(max(abs(gru_df.loc[int(n), "cost15"] - c) for n, c in zip(chk.n, chk.cost)))
if max_gap > 0.01:
    raise SystemExit(f"새로 계산한 GRU 비용이 15번 저장값과 어긋납니다 (최대 차이 {max_gap:.3f}) — 정책·시드가 달라졌는지 확인하세요")

# 3. RF: 저장된 비용(확률 격자)과 test 스윕
rf_cost = saved[saved.method == "RF 분류"].copy()
rf_cost["p"] = rf_cost.setting.str.extract(r"([\d.]+)$").astype(float)
rf_cost_rows = []
for p, g in rf_cost.groupby("p"):
    d = g.set_index("cost_ratio")
    rf_cost_rows.append({"p": r(p, 2), "cost15": r(d.loc["1:5", "cost"]), "cost12": r(d.loc["1:2", "cost"]),
                         "fail": r(d.loc["1:5", "failure_rate"], 4), "wasted": r(d.loc["1:5", "avg_wasted_cycles"], 1)})
rf_cost_df = pd.DataFrame(rf_cost_rows).set_index("p")
rf_sweep_rows = [{"p": r(x.setting, 2), "tp": int(x.TP), "fn": int(x.FN), "fp": int(x.FP)} for x in rf_sw.itertuples()]

# 4. 5가지 안 — 규칙으로 뽑기
cls = read_json("classification_metrics.json")
alert_sum = read_json("alert_comparison_summary.json")
rf_th = float(cls["threshold_adjustment"]["chosen"])
n_b = int(alert_sum["GRU_matched_to_RF_recall"]["setting"])
n_c = int(gru_sw[gru_sw.FN == 0].index.min())
p_e = float(rf_sw[rf_sw.FN == 0].setting.max())

options = []


def gru_option(oid, n, tag):
    g = gru_df.loc[n]
    return {"id": oid, "method": "GRU", "kind": "n", "setting": n, "tag": tag, "tp": int(g.tp), "fn": int(g.fn), "fp": int(g.fp),
            "cost15": g.cost15, "cost12": g.cost12, "fail": g.fail, "wasted": g.wasted}


def rf_option(oid, p, tag):
    s = rf_sw.set_index(rf_sw.setting.round(2)).loc[round(p, 2)]
    c = rf_cost_df.loc[round(p, 2)]
    return {"id": oid, "method": "RF", "kind": "p", "setting": round(p, 2), "tag": tag, "tp": int(s.TP), "fn": int(s.FN), "fp": int(s.FP),
            "cost15": c.cost15, "cost12": c.cost12, "fail": c.fail, "wasted": c.wasted}


options = [gru_option("A", DANGER_RUL, "cost"), gru_option("B", n_b, "balanced"), gru_option("C", n_c, "safety"),
           rf_option("D", rf_th, "rf-current"), rf_option("E", p_e, "rf-zero-miss")]
base_cost = options[0]["cost15"]
for o in options:
    o["extraCostPct"] = r((o["cost15"] / base_cost - 1) * 100, 1)

# 5. 모델 비교 (후보 8종 + 기준 모델)
reg = read_csv("model_screening_regression.csv")
alarm = read_csv("model_screening_alarm_rates.csv")
mcost = read_csv("model_screening_cost.csv")
models = []
for x in reg.itertuples():
    a = alarm[alarm.model == x.model].iloc[0]
    c15 = mcost[(mcost.model == x.model) & (mcost.cost_ratio.str.startswith("1:5"))].iloc[0]
    c12 = mcost[(mcost.model == x.model) & (mcost.cost_ratio.str.startswith("1:2"))].iloc[0]
    models.append({
        "name": x.model, "type": x.type, "valMae": r(x.val_MAE, 2), "valRmse": r(x.val_RMSE, 2), "testMae": r(x.test_MAE, 2),
        "testRmse": r(x.test_RMSE, 2), "testNasa": r(x.test_NASA, 1),
        "valMiss": r(a["val_미탐율"]), "valFalse": r(a["val_오탐율"]),
        "testTp": int(a.test_TP), "testFn": int(a.test_FN), "testFp": int(a.test_FP),
        "bestN15": int(c15.best_N), "bestCost15": r(c15.best_cost, 2), "costAt40_15": r(c15.cost_at_N40, 2), "failAt40": r(c15.failure_rate_at_N40, 3),
        "bestN12": int(c12.best_N), "bestCost12": r(c12.best_cost, 2),
    })
models.sort(key=lambda m: m["testMae"])
mae = np.array([m["valMae"] for m in models if m["type"] != "기준"])
miss = np.array([m["valMiss"] for m in models if m["type"] != "기준"])
corr = float(np.corrcoef(mae, miss)[0, 1])

# 6. 위험 기준 N 곡선 (13번)
tc = read_csv("threshold_cost.csv")
curves = {}
for (model, ratio), g in tc.groupby(["model", "cost_ratio"]):
    curves.setdefault(model, {})[ratio.split(" ")[0]] = [
        {"n": int(x.N), "cost": r(x.cost_per_1000_cycles), "fail": r(x.failure_rate, 4), "wasted": r(x.avg_wasted_cycles, 1)} for x in g.itertuples()]
tcs = read_csv("threshold_cost_summary.csv")
tc_summary = [{"model": x.model, "ratio": x.cost_ratio.split(" ")[0], "bestN": int(x.best_N), "bestCost": r(x.best_cost, 2),
               "within5pct": str(x.within_5pct_N), "n30FailRate": r(x.N30_failure_rate, 3), "n30ExtraPct": r(x.N30_extra_cost_pct, 1),
               "n40FailRate": r(x.N40_failure_rate, 3), "n40ExtraPct": r(x.N40_extra_cost_pct, 1)} for x in tcs.itertuples()]

# 7. 분류 비교·신뢰구간·시드
unc = read_json("classification_uncertainty.json")
comparison = []
for x in cls["comparison"]:
    wil = unc["test_wilson_95"].get(x["model"]) or unc["test_wilson_95"].get(x["model"].replace("임계값", "임곗값"))
    comparison.append({**{k: v for k, v in x.items()}, "wilson": wil})
seed = read_csv("seed_check.csv")
seeds = [{"model": x.model, "seed": int(x.seed), "mae": r(x.test_MAE, 2), "rmse": r(x.test_RMSE, 2), "nasa": r(x.test_NASA, 1)} for x in seed.itertuples()]
lead = cls["lead_time"]

extras = {
    "policy": {"testN": alert_sum["test_n"], "options": options, "gruSweep": gru_rows, "rfSweep": rf_sweep_rows, "rfCost": rf_cost_rows,
               "danger": DANGER_RUL},
    "models": models, "maeMissCorr": r(corr, 2),
    "thresholdCurves": curves, "thresholdSummary": tc_summary,
    "classification": comparison,
    "uncertainty": {"validation": unc["validation"], "bootstrap": unc["bootstrap"]},
    "seeds": seeds,
    "leadTime": {"rule": lead["rule"], "median": lead["median"], "perEngine": lead["per_engine"]},
}
with open(f"{OUT_DASH}/analysis_extras.json", "w", encoding="utf-8") as f:
    json.dump(extras, f, ensure_ascii=False)

stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
code = f"""// ⚠️ 자동 생성 파일 — 분석 레포의 scripts/19_export_analysis_extras.py 가 만든다. 직접 고치지 말고 파이썬을 다시 돌려서 받아올 것.
// 경보 기준 5가지 안 · 모델 후보 비교 · 위험 기준 비용 곡선 · 신뢰구간 · 시드 반복 (생성: {stamp}).
// 놓침·헛경보는 공식 test(위험 {alert_sum['test_n']['danger']}대·안전 {alert_sum['test_n']['safe']}대), 정비 전 고장·비용·낭비는 validation 49대 정비 시뮬레이션이다.

export const analysisExtras = {json.dumps(extras, ensure_ascii=False, separators=(",", ":"))} as const
"""
with open(f"{OUT_DASH}/analysis_extras.ts", "w", encoding="utf-8") as f:
    f.write(code)

print(f"검증: GRU 비용 재계산과 15번 저장값 최대 차이 {max_gap:.4f}")
print("5가지 안:", ", ".join(f"{o['id']}={o['method']}{'≤' if o['kind']=='n' else '≥'}{o['setting']}(놓침{o['fn']}·헛경보{o['fp']}·비용{o['cost15']})" for o in options))
print(f"MAE-미탐율 상관 {corr:.2f}, 모델 {len(models)}종")
