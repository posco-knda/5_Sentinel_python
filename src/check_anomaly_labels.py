"""
6개 데이터셋의 Anomaly_* 라벨 확인:
- 라벨별 개수/비율, 라벨이 하나라도 True인 행 비율
- 기초(1,2) vs 응용(3~6): 활성 라벨 수, 복합 라벨(2개 이상 동시 True) 여부
"""

import pandas as pd

BASIC = [1, 2]
APPLIED = [3, 4, 5, 6]

summary_rows = []

for i in BASIC + APPLIED:
    df = pd.read_csv(f"data/tcm5_dataset_{i}.csv")
    anomaly_cols = [c for c in df.columns if c.startswith("Anomaly")]

    counts = df[anomaly_cols].sum().sort_values(ascending=False)
    active_cols = counts[counts > 0].index.tolist()
    zero_cols = counts[counts == 0].index.tolist()

    any_anomaly = df[anomaly_cols].any(axis=1)
    multi_label = (df[anomaly_cols].sum(axis=1) > 1).sum()
    group = "기초" if i in BASIC else "응용"

    print(f"=== dataset_{i} ({group}), shape={df.shape} ===")
    print("라벨별 개수 (True):")
    print(counts)
    print(f"활성화된 라벨: {len(active_cols)}개 / 항상 0인 라벨: {len(zero_cols)}개")
    print(f"이상치 행: {any_anomaly.sum()} / {len(df)} ({any_anomaly.mean() * 100:.3f}%)")
    print(f"2개 이상 라벨 동시 True: {multi_label}건")
    print()

    summary_rows.append(
        {
            "dataset": i,
            "group": group,
            "rows": len(df),
            "active_labels": len(active_cols),
            "any_anomaly_rows": int(any_anomaly.sum()),
            "any_anomaly_pct": round(any_anomaly.mean() * 100, 3),
            "multi_label_rows": int(multi_label),
        }
    )

print("=== 요약: 기초 vs 응용 ===")
print(pd.DataFrame(summary_rows).to_string(index=False))
