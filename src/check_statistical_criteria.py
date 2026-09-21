"""
데이터셋의 Anomaly_* 라벨(시뮬레이터가 생성한 정답)이
평균/중앙값/표준편차 기반의 단순 통계적 임계값으로도 재현되는지 검증한다.

방법:
1. 논문 설명에 따라 각 이상치 유형을 관련 센서 컬럼에 매핑한다.
   - Anomaly_Bearing_i (토크 증가)      -> torque_i
   - Anomaly_Electric_i (전기효율 저하)  -> motor_power_i
   - Anomaly_WorkRoll_i (마찰 증가)     -> force_i
   - Anomaly_Reduction (압하 스케줄 이상) -> reduction_1..5 (각각 검사)
2. 라벨을 전혀 보지 않고, 전체 데이터로 평균/표준편차/IQR 기준선을 만든다.
   (이상치 비율이 0.1~1%로 작아 "대부분 정상"이라는 가정만으로 기준선이 성립)
3. 그 기준선으로 z-score, IQR 임계값을 계산해 이상 여부를 예측하고,
   맨 마지막에만 실제 Anomaly_* 라벨을 꺼내 채점(AUC / Precision / Recall / F1)한다.
   -> 기준선을 만들 때 라벨을 쓰면 누수(leakage)이므로, 라벨은 평가 단계에서만 사용한다.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

df = pd.read_csv("data/tcm5_dataset_3.csv")

mapping = []
for i in range(1, 6):
    mapping.append((f"Anomaly_Bearing_{i}", f"torque_{i}"))
    mapping.append((f"Anomaly_Electric_{i}", f"motor_power_{i}"))
    mapping.append((f"Anomaly_WorkRoll_{i}", f"force_{i}"))
for i in range(1, 6):
    mapping.append(("Anomaly_Reduction", f"reduction_{i}"))


def evaluate(label_col: str, sensor_col: str):
    y = df[label_col].astype(bool)
    x = df[sensor_col]

    # 기준선은 라벨을 보지 않고 전체 데이터(x)로만 계산한다 (비지도 가정).
    mean, std = x.mean(), x.std()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1

    z = (x - mean) / std
    # 라벨(y)은 여기, 즉 예측 결과를 채점하는 데만 사용한다.
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
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        results[name] = (precision, recall, f1, tp, fp, fn)

    return auc, results


print(f"{'label':<22}{'sensor':<16}{'AUC':>7}   | best rule (F1 최대)")
print("-" * 90)

rows = []
for label_col, sensor_col in mapping:
    auc, results = evaluate(label_col, sensor_col)
    best_name, best = max(results.items(), key=lambda kv: kv[1][2])
    precision, recall, f1, tp, fp, fn = best
    rows.append((label_col, sensor_col, auc, best_name, precision, recall, f1, tp, fp, fn))
    print(
        f"{label_col:<22}{sensor_col:<16}{auc:7.3f}   | "
        f"{best_name:<10} P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
        f"(TP={tp}, FP={fp}, FN={fn})"
    )

print()
print("=== 요약 ===")
result_df = pd.DataFrame(
    rows,
    columns=["label", "sensor", "auc", "best_rule", "precision", "recall", "f1", "tp", "fp", "fn"],
)
print(f"평균 AUC: {result_df['auc'].mean():.3f}")
print(f"평균 F1(최적 임계값): {result_df['f1'].mean():.3f}")
