"""
데이터셋의 Anomaly_* 라벨(시뮬레이터가 생성한 정답)이
통계적 기준만으로 재현되는지 dataset_3~6 전체에서 검증한다.
라벨은 기준선을 만들 때 쓰지 않고, 예측 결과를 채점할 때만 사용한다(누수 방지).

Part 1. 단변량: 이상치 유형을 관련 센서 1개에 매핑해 z-score/IQR 임계값으로 예측.
   - Anomaly_Bearing_i (토크 증가)      -> torque_i
   - Anomaly_Electric_i (전기효율 저하)  -> motor_power_i
   - Anomaly_WorkRoll_i (마찰 증가)     -> force_i
   - Anomaly_Reduction (압하 스케줄 이상) -> reduction_1..5 (각각 검사)

Part 2. 다변량: 스탠드별 센서를 묶어 EllipticEnvelope(마할라노비스)와
   IsolationForest로 비지도 이상탐지 후 채점. 단변량으로 약했던
   WorkRoll/Reduction이 다변량에서 얼마나 개선되는지 비교한다.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

DATASETS = [3, 4, 5, 6]


# ---------- Part 1. 단변량 ----------

def univariate_mapping():
    mapping = []
    for i in range(1, 6):
        mapping.append((f"Anomaly_Bearing_{i}", f"torque_{i}"))
        mapping.append((f"Anomaly_Electric_{i}", f"motor_power_{i}"))
        mapping.append((f"Anomaly_WorkRoll_{i}", f"force_{i}"))
    for i in range(1, 6):
        mapping.append(("Anomaly_Reduction", f"reduction_{i}"))
    return mapping


def evaluate_univariate(df, label_col, sensor_col):
    y = df[label_col].astype(bool)
    x = df[sensor_col]

    mean, std = x.mean(), x.std()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1

    z = (x - mean) / std
    auc = roc_auc_score(y, z.abs())

    results = {}
    for name, pred in {
        "mean±2std": (z.abs() > 2),
        "mean±3std": (z.abs() > 3),
        "IQR*1.5": (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr),
        "IQR*3.0": (x < q1 - 3.0 * iqr) | (x > q3 + 3.0 * iqr),
    }.items():
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        results[name] = (precision, recall, f1)

    best_name, (precision, recall, f1) = max(results.items(), key=lambda kv: kv[1][2])
    return auc, best_name, precision, recall, f1


# ---------- Part 2. 다변량 ----------

STAND_FEATURES = ["torque", "force", "motor_power", "roll_speed", "gap", "reduction"]


def score_mahalanobis(X):
    model = EllipticEnvelope(contamination=0.01, random_state=42, support_fraction=0.9)
    model.fit(X)
    return -model.decision_function(X)


def score_isolation_forest(X):
    model = IsolationForest(contamination=0.01, random_state=42, n_estimators=200)
    model.fit(X)
    return -model.decision_function(X)


def evaluate_stand(df, stand):
    cols = [f"{f}_{stand}" for f in STAND_FEATURES]
    X = StandardScaler().fit_transform(df[cols])

    maha_score = score_mahalanobis(X)
    iforest_score = score_isolation_forest(X)

    labels = {
        "Bearing": df[f"Anomaly_Bearing_{stand}"].astype(bool),
        "Electric": df[f"Anomaly_Electric_{stand}"].astype(bool),
        "WorkRoll": df[f"Anomaly_WorkRoll_{stand}"].astype(bool),
    }
    labels["any"] = labels["Bearing"] | labels["Electric"] | labels["WorkRoll"]

    rows = []
    for name, y in labels.items():
        if y.sum() == 0:
            continue
        rows.append((f"stand{stand}_{name}", "Mahalanobis", roc_auc_score(y, maha_score)))
        rows.append((f"stand{stand}_{name}", "IsolationForest", roc_auc_score(y, iforest_score)))
    return rows


def evaluate_reduction_multivariate(df):
    cols = [f"reduction_{i}" for i in range(1, 6)]
    X = StandardScaler().fit_transform(df[cols])
    y = df["Anomaly_Reduction"].astype(bool)

    maha_score = score_mahalanobis(X)
    iforest_score = score_isolation_forest(X)

    return [
        ("Anomaly_Reduction", "Mahalanobis", roc_auc_score(y, maha_score)),
        ("Anomaly_Reduction", "IsolationForest", roc_auc_score(y, iforest_score)),
    ]


# ---------- 실행 ----------

uni_rows = []
multi_rows = []

for ds in DATASETS:
    df = pd.read_csv(f"data/tcm5_dataset_{ds}.csv")
    print(f"##### dataset_{ds} (n={len(df)}) #####")

    print("--- Part 1. 단변량 (best rule = F1 최대) ---")
    for label_col, sensor_col in univariate_mapping():
        auc, best_name, precision, recall, f1 = evaluate_univariate(df, label_col, sensor_col)
        print(
            f"  {label_col:<22}{sensor_col:<16}AUC={auc:.3f} | "
            f"{best_name:<10} P={precision:.2f} R={recall:.2f} F1={f1:.2f}"
        )
        uni_rows.append((ds, label_col, sensor_col, auc, f1))
    print()

    print("--- Part 2. 다변량 (Mahalanobis / IsolationForest) ---")
    ds_multi = []
    for stand in range(1, 6):
        ds_multi += evaluate_stand(df, stand)
    ds_multi += evaluate_reduction_multivariate(df)
    for label, method, auc in ds_multi:
        print(f"  {label:<16}{method:<16}AUC={auc:.3f}")
        multi_rows.append((ds, label, method, auc))
    print()

uni_df = pd.DataFrame(uni_rows, columns=["dataset", "label", "sensor", "auc", "f1"])
multi_df = pd.DataFrame(multi_rows, columns=["dataset", "label", "method", "auc"])

print("=== 종합 비교 (4개 데이터셋 평균) ===")
print(f"단변량 평균 AUC: {uni_df['auc'].mean():.3f} / 평균 F1: {uni_df['f1'].mean():.3f}")
print(multi_df.groupby("method")["auc"].mean().round(3))
print()

print("=== WorkRoll/Reduction: 단변량 vs 다변량 비교 ===")
uni_target = uni_df[uni_df["label"].str.contains("WorkRoll|Reduction")]
print("단변량 AUC:")
print(uni_target.groupby("label")["auc"].mean().round(3))
print("다변량 AUC:")
multi_target = multi_df[multi_df["label"].str.contains("WorkRoll|Reduction")]
print(multi_target.groupby(["label", "method"])["auc"].mean().round(3))
