"""
17_engine_baseline_features.py
==============================
"엔진별 초기 상태 대비 변화량" 특성을 넣으면 오탐·미탐이 줄어드는지 시험합니다.

아이디어: 엔진마다 처음부터 센서 값이 조금씩 다릅니다(개체 차이). 그래서 같은 값이어도 어떤 엔진에서는 "정상",
다른 엔진에서는 "닳은 상태"일 수 있습니다. 각 엔진의 처음 K사이클 평균을 그 엔진의 기준(초기 상태)으로 삼고,
"지금 값 − 그 엔진의 초기 값"(변화량)을 모델에 추가로 줍니다. 처음 K사이클은 그 엔진이 운행하면서 이미 관측한
값이라 새 엔진·test 엔진에도 똑같이 쓸 수 있습니다.

주의(기본 방식의 한계): 기본 방식은 엔진 전체 행에서 "처음 K사이클 평균"을 빼기 때문에, K사이클 이전 행의 변화량은
그 시점에는 아직 관측하지 못한 뒤쪽 값(최대 K사이클째까지)을 포함합니다. 위험 구간(수명 후반)과는 멀리 떨어진
행이라 오탐·미탐 결과에는 거의 영향이 없지만, 엄밀한 온라인 사용 방식은 아닙니다. 그래서 환경변수
DELTA_CAUSAL=1 로 "그 시점까지 관측한 값만으로 평균"을 쓰는 인과적 방식을 선택할 수 있습니다
(K사이클까지는 지금까지의 평균, 이후는 처음 K사이클 평균 고정). 이 방식의 결과 파일은 이름 끝에 _causal 이 붙고,
기본 방식의 기존 결과(delta_exp_*.json)는 그대로입니다.

비교(전부 같은 엔진 분할 seed=42, 같은 정규화, 같은 학습 설정, 경보 기준 고정):
  RF 분류  : 기본 특성 60개 vs 기본 + 변화량 30개(현재값·최근 15사이클 평균의 변화량)  → 위험 확률 ≥ 09번이 고른 값(0.36)이면 경보
  GRU 회귀 : 입력 센서 15개 vs 변화량 15개 vs 센서 + 변화량 30개                     → 예측 남은 수명 ≤ 40이면 경보

실행: python scripts/17_engine_baseline_features.py rf
      python scripts/17_engine_baseline_features.py gru <base|delta|both> <seed>
      (인과적 방식: DELTA_CAUSAL=1 을 앞에 붙여 실행)
결과: outputs/FD004/metrics/delta_exp_*.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from common import (
    load_raw, find_constant_columns, add_rul_labels, split_engines, Normalizer, build_rf_features,
    SENSOR_COLS_RAW, DANGER_RUL, RUL_CLIP_VALUE, OUT_METRICS, TRAIN_FILE, TEST_FILE, RUL_FILE,
)

BASELINE_CYCLES = 30           # 엔진의 처음 30사이클 평균을 그 엔진의 초기 상태로
CAUSAL = os.environ.get("DELTA_CAUSAL") == "1"   # 위 설명 참고: 1이면 그 시점까지 관측한 값만으로 초기 평균을 계산
SUFFIX = "_causal" if CAUSAL else ""
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)   # 09번과 동일
# 확률 기준은 09번이 고른 값을 그대로 읽습니다 (다시 조정하지 않음).
with open(f"{OUT_METRICS}/classification_metrics.json", encoding="utf-8") as _f:
    _cls = json.load(_f)
RF_PROB_TH = float(_cls["threshold_adjustment"]["chosen"])
# 사례 엔진: 09번(RF 오탐·미탐)과 16번(GRU 미탐)이 저장한 값을 읽습니다 (번호를 코드에 박아 두지 않음).
def _unit_of(case):
    return None if case is None else int(case["unit"])  # 09·16번이 사례가 없으면 None으로 저장


with open(f"{OUT_METRICS}/gru_error_cases.json", encoding="utf-8") as _f:
    _gru_cases = json.load(_f)
FOCUS = {"FP": _unit_of(_cls["error_cases"]["false_alarm"]), "FN_rf": _unit_of(_cls["error_cases"]["missed"]),
         "FN_gru": _unit_of(_gru_cases["missed"])}

# ---------------------------------------------------------------------------
# 데이터 (09·04번과 같은 분할·정규화)
# ---------------------------------------------------------------------------
train_raw = add_rul_labels(load_raw(TRAIN_FILE))
train_units, val_units = split_engines(train_raw)
const_cols = find_constant_columns(train_raw[train_raw["unit"].isin(train_units)])
sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]
norm = Normalizer().fit(train_raw[train_raw["unit"].isin(train_units)], sensors)
train_df = norm.transform(train_raw[train_raw["unit"].isin(train_units)])
val_df = norm.transform(train_raw[train_raw["unit"].isin(val_units)])
test_raw = load_raw(TEST_FILE)
test_df = norm.transform(test_raw)
rul_true = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul_true["unit"] = np.arange(1, len(rul_true) + 1)


def add_delta(df: pd.DataFrame) -> pd.DataFrame:
    """각 엔진의 처음 BASELINE_CYCLES사이클 평균을 빼서 d_<센서> 열(변화량)을 추가."""
    df = df.sort_values(["unit", "cycle"]).copy()

    def initial_level(s: pd.Series) -> pd.Series:
        fixed = s.iloc[:BASELINE_CYCLES].mean()
        if not CAUSAL:
            return pd.Series(fixed, index=s.index)
        level = s.expanding().mean()          # K사이클까지는 "지금까지 관측한 평균"
        level.iloc[BASELINE_CYCLES:] = fixed  # 그 뒤는 처음 K사이클 평균으로 고정
        return level

    base = df.groupby("unit")[sensors].transform(initial_level)
    for c in sensors:
        df["d_" + c] = df[c] - base[c]
    return df


train_df, val_df, test_df = add_delta(train_df), add_delta(val_df), add_delta(test_df)
delta_cols = ["d_" + c for c in sensors]


def alert_stats(unit, cycle, rul_true_, alarm) -> dict:
    """validation 전체 사이클 기준 오탐·미탐 사이클 수 + 사례 엔진별 수."""
    d = pd.DataFrame({"unit": unit, "cycle": cycle, "danger": np.asarray(rul_true_) <= DANGER_RUL, "alarm": np.asarray(alarm)})
    fp = (d.alarm & ~d.danger); fn = (~d.alarm & d.danger)
    by_unit_fp = fp.groupby(d.unit).sum(); by_unit_fn = fn.groupby(d.unit).sum()
    safe_alarm_engines = int((d[~d.danger].groupby("unit")["alarm"].sum() > 0).sum())
    return {
        "val_FP_cycles": int(fp.sum()), "val_FN_cycles": int(fn.sum()),
        "val_engines_with_false_alarm": safe_alarm_engines,
        # 사례 엔진이 validation 집합에 없으면 0이 아니라 None (0건으로 오해하지 않도록)
        **{f"{name}_{u}": (int(series.get(u)) if u in series.index else None)
           for name, u, series in (("FP", FOCUS["FP"], by_unit_fp), ("FN", FOCUS["FN_rf"], by_unit_fn),
                                   ("FN", FOCUS["FN_gru"], by_unit_fn)) if u is not None},
        "worst_FP_engine": int(by_unit_fp.idxmax()), "worst_FP_cycles": int(by_unit_fp.max()),
        "worst_FN_engine": int(by_unit_fn.idxmax()), "worst_FN_cycles": int(by_unit_fn.max()),
    }


def test_stats(alarm, danger) -> dict:
    alarm = np.asarray(alarm).astype(bool); danger = np.asarray(danger).astype(bool)
    tp = int((alarm & danger).sum())
    return {"test_TP": tp, "test_FN": int((~alarm & danger).sum()), "test_FP": int((alarm & ~danger).sum()),
            "test_recall": round(tp / max(1, int(danger.sum())), 3)}


def run_rf():
    from sklearn.ensemble import RandomForestClassifier
    tr_f = build_rf_features(train_df, sensors); va_f = build_rf_features(val_df, sensors); te_f = build_rf_features(test_df, sensors)
    base_cols = [c for c in tr_f.columns if c not in ("unit", "cycle", "RUL")]

    def with_delta(feat: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
        feat = feat.merge(src[["unit", "cycle"] + delta_cols], on=["unit", "cycle"])
        for c in sensors:  # 변화량 = (현재 평균) − (엔진 초기 평균): 평균 특성의 변화량 = 현재 평균 − (현재값 − d_현재값)
            feat["dm_" + c] = feat[f"{c}_mean15"] - (feat[c] - feat["d_" + c])
        return feat

    tr2, va2, te2 = with_delta(tr_f, train_df), with_delta(va_f, val_df), with_delta(te_f, test_df)
    plus_cols = base_cols + delta_cols + ["dm_" + c for c in sensors]
    te_last = {name: f.loc[f.groupby("unit")["cycle"].idxmax()].merge(rul_true, on="unit")
               for name, f in (("base", te_f), ("plus", te2))}
    out = {}
    for name, cols, tr, va, te in (("RF_base", base_cols, tr_f, va_f, te_last["base"]),
                                   ("RF_plus", plus_cols, tr2, va2, te_last["plus"])):
        clf = RandomForestClassifier(**RF_PARAMS).fit(tr[cols], (tr["RUL"] <= DANGER_RUL).astype(int))
        pv = clf.predict_proba(va[cols])[:, 1]; pt = clf.predict_proba(te[cols])[:, 1]
        res = alert_stats(va["unit"], va["cycle"], va["RUL"], pv >= RF_PROB_TH)
        res.update(test_stats(pt >= RF_PROB_TH, te["RUL"] <= DANGER_RUL))
        res["n_features"] = len(cols)
        out[name] = res
        print(name, json.dumps(res, ensure_ascii=False))
    with open(f"{OUT_METRICS}/delta_exp_rf{SUFFIX}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def run_gru(variant: str, seed: int):
    import time
    from seq_models import RULGRU, engine_sequences, train_recurrent, predict_recurrent
    cols = {"base": sensors, "delta": delta_cols, "both": sensors + delta_cols}[variant]
    tr_s, tr_l, _ = engine_sequences(train_df, cols)
    va_s, va_l, va_ids = engine_sequences(val_df, cols)
    te_s, _, te_ids = engine_sequences(test_df, cols, has_label=False)
    t0 = time.time()
    model = train_recurrent(RULGRU, tr_s, tr_l, len(cols), seed=seed)
    rows = []
    for uid, seq, lab in zip(va_ids, va_s, va_l):
        pred = predict_recurrent(model, seq)
        for c, t_, p_ in zip(range(1, len(seq) + 1), lab.numpy() * RUL_CLIP_VALUE, pred):
            rows.append((uid, c, t_, p_))
    v = pd.DataFrame(rows, columns=["unit", "cycle", "RUL_true", "RUL_pred"])
    res = alert_stats(v["unit"], v["cycle"], v["RUL_true"], v["RUL_pred"] <= DANGER_RUL)
    res["val_MAE"] = round(float(np.mean(np.abs(v["RUL_true"] - v["RUL_pred"]))), 2)
    t = pd.DataFrame([{"unit": u, "pred": predict_recurrent(model, s)[-1]} for u, s in zip(te_ids, te_s)]).merge(rul_true, on="unit")
    res.update(test_stats(t["pred"] <= DANGER_RUL, t["RUL"] <= DANGER_RUL))
    res["test_MAE"] = round(float(np.mean(np.abs(t["RUL"] - t["pred"]))), 2)
    res["variant"], res["seed"], res["seconds"] = variant, seed, round(time.time() - t0)
    v.to_csv(f"{OUT_METRICS}/delta_exp_gru_{variant}_{seed}{SUFFIX}_val.csv", index=False)
    with open(f"{OUT_METRICS}/delta_exp_gru_{variant}_{seed}{SUFFIX}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(variant, seed, json.dumps(res, ensure_ascii=False))


USAGE = ("사용법: python scripts/17_engine_baseline_features.py rf"
         " | python scripts/17_engine_baseline_features.py gru <base|delta|both> <seed>")

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["rf"]:
        run_rf()
    elif len(args) == 3 and args[0] == "gru" and args[1] in ("base", "delta", "both") and args[2].lstrip("-").isdigit():
        run_gru(args[1], int(args[2]))
    else:
        raise SystemExit(USAGE)
