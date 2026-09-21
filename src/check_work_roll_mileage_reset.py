"""
체크리스트 2단계 - 3: work_roll_mileage 리셋 지점(수명 주기) 탐지
tcm5_dataset_1.csv의 work_roll_mileage_1~5 컬럼에서 이전 행보다 값이 갑자기
작아지는(리셋되는) 지점을 스탠드별로 찾아 수명 주기를 요약한다.
(롤 교체로 마일리지가 0 근처로 초기화되는 시점 기준)
"""

import pandas as pd

df = pd.read_csv("data/tcm5_dataset_1.csv")

summary = []
for i in range(1, 6):
    col = f"work_roll_mileage_{i}"
    mileage = df[col]
    reset_rows = df.index[mileage.diff() < 0].tolist()

    cycle_lengths = pd.Series(reset_rows).diff().dropna()
    mileage_at_reset = mileage.iloc[[r - 1 for r in reset_rows]]

    summary.append(
        {
            "stand": i,
            "resets": len(reset_rows),
            "avg_cycle_rows": round(cycle_lengths.mean(), 1),
            "min_cycle_rows": int(cycle_lengths.min()),
            "max_cycle_rows": int(cycle_lengths.max()),
            "avg_mileage_at_reset": round(mileage_at_reset.mean(), 2),
            "max_mileage_at_reset": round(mileage_at_reset.max(), 2),
        }
    )

print("=== 스탠드별 work_roll_mileage 수명 주기 요약 (dataset_1) ===")
print(pd.DataFrame(summary).to_string(index=False))
