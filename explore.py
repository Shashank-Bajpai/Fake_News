import pandas as pd

df = pd.read_excel("data/HinFakeNews-V1.xlsx")

# Rename for clarity: 1 = FAKE, 0 = REAL (matches BOOL meaning we just verified)
df["label"] = df["BOOL"].map({0: "FAKE", 1: "REAL"})

print("Label counts:")
print(df["label"].value_counts())

print("\nAny duplicate URLs?", df["URL"].duplicated().sum())
print("Any duplicate TITLE?", df["TITLE"].duplicated().sum())

print("\nCONTENT length stats (in words):")
df["content_len"] = df["CONTENT"].astype(str).str.split().apply(len)
print(df.groupby("label")["content_len"].describe())