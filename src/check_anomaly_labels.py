import pandas as pd

df = pd.read_csv("data/tcm5_dataset_3.csv")

anomaly_cols = [c for c in df.columns if c.startswith("Anomaly")]

print("=== shape ===")
print(df.shape)
print()

print("=== anomaly columns ===")
print(anomaly_cols)
print()

print("=== anomaly counts (True 개수) ===")
counts = df[anomaly_cols].sum().sort_values(ascending=False)
print(counts)
print()

print("=== anomaly ratio (%) ===")
print((counts / len(df) * 100).round(3))
print()

any_anomaly = df[anomaly_cols].any(axis=1)
print("=== 이상치 라벨이 하나라도 True인 행 ===")
print(f"{any_anomaly.sum()} / {len(df)} ({any_anomaly.mean() * 100:.3f}%)")
