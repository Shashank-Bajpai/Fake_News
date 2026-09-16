import pandas as pd

df = pd.read_excel("data/HinFakeNews-V1.xlsx")
df["label"] = df["BOOL"].map({0: "FAKE", 1: "REAL"})

print("Before dedup:", df.shape)

# Drop rows with duplicate URLs (same article scraped twice)
df = df.drop_duplicates(subset="URL", keep="first")

# Drop rows with duplicate TITLE (catches re-published/mirrored articles
# even if the URL differs slightly)
df = df.drop_duplicates(subset="TITLE", keep="first")

print("After dedup:", df.shape)

# Drop rows with very short content — near-empty articles add noise,
# not signal, and can't really be judged as fake/real from content alone
before = len(df)
df = df[df["CONTENT"].astype(str).str.split().apply(len) >= 20]
print(f"Dropped {before - len(df)} rows with < 20 words of content")

print("\nFinal label distribution:")
print(df["label"].value_counts())

# Save cleaned version so we don't repeat this every time
df.to_csv("data/hinfakenews_clean.csv", index=False)
print("\nSaved cleaned dataset to data/hinfakenews_clean.csv")