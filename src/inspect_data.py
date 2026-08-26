import pandas as pd

X_train = pd.read_csv("./data/X_train.csv")
y_train = pd.read_csv("./data/y_train.csv")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nX_train columns:")
print(X_train.columns.tolist())

print("\ny_train columns:")
print(y_train.columns.tolist())

print("\nSample X:")
print(X_train.head())

print("\nSample y:")
print(y_train.head())

print("\nMissing values in X:")
print(X_train.isnull().sum())

print("\nMissing values in y:")
print(y_train.isnull().sum())

print("\nUnique labels:")
for col in y_train.columns:
    print(col, y_train[col].unique())

print("\nLabel counts:")
for col in y_train.columns:
    print(y_train[col].value_counts())