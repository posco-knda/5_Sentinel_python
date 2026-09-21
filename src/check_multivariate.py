"""
단변량(z-score/IQR) 검증의 한계(특히 force_i, reduction_i)를 다변량으로 보완해서
dataset_3~6에 걸쳐 이상치 라벨이 재현되는지 확인한다.

방법 (라벨은 여전히 '채점'에만 사용, 모델 학습에는 사용하지 않음):
1. 스탠드별로 관련 센서를 묶어 다변량 피처를 만든다.
     [torque_i, force_i, motor_power_i, roll_speed_i, gap_i, reduction_i]
2. 표준화 후 두 가지 비지도 이상탐지기를 라벨 없이 피팅한다.
     - EllipticEnvelope: 강건 공분산 기반 마할라노비스 거리 (통계적 다변량 이상치 검정)
     - IsolationForest: 트리 기반 비지도 이상탐지 (비교용)
3. 각 스탠드의 이상치 점수를 Anomaly_Bearing_i / Anomaly_Electric_i / Anomaly_WorkRoll_i /
   (셋 중 하나라도 True인) any_i 라벨과 비교해 AUC로 채점한다.
4. Anomaly_Reduction은 reduction_1..5 5개 컬럼을 묶어 같은 방식으로 검증한다.
5. dataset_3~6 네 개에 대해 반복하고 평균을 낸다.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

DATASETS = [3, 4, 5, 6]
STAND_FEATURES = ["torque", "force", "motor_power", "roll_speed", "gap", "reduction"]


def score_mahalanobis(X):
    model = EllipticEnvelope(contamination=0.01, random_state=42, support_fraction=0.9)
    model.fit(X)
    return -model.decision_function(X)  # 클수록 이상치


def score_isolation_forest(X):
    model = IsolationForest(contamination=0.01, random_state=42, n_estimators=200)
    model.fit(X)
    return -model.decision_function(X)  # 클수록 이상치


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


def evaluate_reduction(df):
    cols = [f"reduction_{i}" for i in range(1, 6)]
    X = StandardScaler().fit_transform(df[cols])
    y = df["Anomaly_Reduction"].astype(bool)

    maha_score = score_mahalanobis(X)
    iforest_score = score_isolation_forest(X)

    return [
        ("Anomaly_Reduction", "Mahalanobis", roc_auc_score(y, maha_score)),
        ("Anomaly_Reduction", "IsolationForest", roc_auc_score(y, iforest_score)),
    ]


all_rows = []
for ds in DATASETS:
    df = pd.read_csv(f"data/tcm5_dataset_{ds}.csv")
    print(f"=== dataset_{ds} (n={len(df)}) ===")

    ds_rows = []
    for stand in range(1, 6):
        ds_rows += evaluate_stand(df, stand)
    ds_rows += evaluate_reduction(df)

    for label, method, auc in ds_rows:
        print(f"  {label:<16}{method:<16}AUC={auc:.3f}")
        all_rows.append((ds, label, method, auc))
    print()

result = pd.DataFrame(all_rows, columns=["dataset", "label", "method", "auc"])

print("=== 방법별 평균 AUC (전체 데이터셋 3~6, 모든 라벨) ===")
print(result.groupby("method")["auc"].mean().round(3))
print()

print("=== 단변량(이전 결과, dataset_3) vs 다변량 비교가 필요한 라벨: force/reduction 계열 ===")
target = result[result["label"].str.contains("WorkRoll|Reduction")]
print(target.groupby(["label", "method"])["auc"].mean().round(3))
