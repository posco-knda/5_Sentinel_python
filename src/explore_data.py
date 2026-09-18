import pandas as pd

df = pd.read_csv("data/tcm5_dataset_1.csv")

print("=== shape ===")
print(df.shape)
print()

print("=== head ===")
print(df.head())
print()

print("=== isnull sum ===")
print(df.isnull().sum())
