import pandas as pd
from urllib.parse import urlparse

df = pd.read_csv("data/hinfakenews_clean.csv")
df["domain"] = df["URL"].apply(lambda u: urlparse(u).netloc)

print("Domain vs label crosstab:")
print(pd.crosstab(df["domain"], df["label"]).sort_values("FAKE", ascending=False).head(15))