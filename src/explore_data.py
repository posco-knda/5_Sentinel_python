"""
체크리스트 2단계 - 1: 6개 CSV 각각 shape, 컬럼, head, 결측치 확인
"""

import pandas as pd

for i in range(1, 7):
    df = pd.read_csv(f"data/tcm5_dataset_{i}.csv")

    print(f"=== dataset_{i} ===")
    print("shape:", df.shape)
    print("columns:", df.columns.tolist())
    print()

    print("head:")
    print(df.head(3))
    print()

    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    print("결측치:", "없음" if null_cols.empty else "")
    if not null_cols.empty:
        print(null_cols)
    print()
