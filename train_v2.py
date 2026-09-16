import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from preprocess import clean_hindi_text

RANDOM_STATE = 42

df = pd.read_csv("data/hinfakenews_clean.csv")
df["domain"] = df["URL"].apply(lambda u: urlparse(u).netloc)
df["content"] = df["TITLE"].fillna("") + " " + df["CONTENT"].fillna("")
print("Cleaning text...")
df["clean_content"] = df["content"].apply(clean_hindi_text)
df["label_bin"] = (df["label"] == "FAKE").astype(int)

# Hold out entire domains for testing instead of random rows.
# Pick 2 FAKE-source domains and 2 REAL-source domains to leave out entirely.
test_domains = ["hindi.asianetnews.com", "hindi.oneindia.com",  # fake-side sources
                 "hindi.news18.com", "www.jagran.com"]          # real-side sources

test_df = df[df["domain"].isin(test_domains)]
train_df = df[~df["domain"].isin(test_domains)]

print(f"Train: {len(train_df)} rows from {train_df['domain'].nunique()} domains")
print(f"Test : {len(test_df)} rows from {test_df['domain'].nunique()} domains (UNSEEN publishers)")
print("Train label balance:\n", train_df["label"].value_counts())
print("Test label balance:\n", test_df["label"].value_counts())

vectorizer = TfidfVectorizer(
    max_df=0.7, ngram_range=(1, 2), max_features=50000,
    token_pattern=r"[^\s]+"
)
X_train = vectorizer.fit_transform(train_df["clean_content"])
X_test = vectorizer.transform(test_df["clean_content"])

model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
model.fit(X_train, train_df["label_bin"])
preds = model.predict(X_test)

acc = accuracy_score(test_df["label_bin"], preds)
prec, rec, f1, _ = precision_recall_fscore_support(test_df["label_bin"], preds, average="binary")
print(f"\n--- Domain-held-out evaluation ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1       : {f1:.4f}")
print(confusion_matrix(test_df["label_bin"], preds))
print(classification_report(test_df["label_bin"], preds, target_names=["REAL", "FAKE"]))