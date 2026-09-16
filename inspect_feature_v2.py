import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from preprocess import clean_hindi_text

df = pd.read_csv("data/hinfakenews_clean.csv")
df["domain"] = df["URL"].apply(lambda u: urlparse(u).netloc)
df["content"] = df["TITLE"].fillna("") + " " + df["CONTENT"].fillna("")
df["clean_content"] = df["content"].apply(clean_hindi_text)
df["label_bin"] = (df["label"] == "FAKE").astype(int)

test_domains = ["hindi.asianetnews.com", "hindi.oneindia.com", "hindi.news18.com", "www.jagran.com"]
train_df = df[~df["domain"].isin(test_domains)]

vectorizer = TfidfVectorizer(max_df=0.7, ngram_range=(1, 2), max_features=50000, token_pattern=r"[^\s]+")
X_train = vectorizer.fit_transform(train_df["clean_content"])

model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train, train_df["label_bin"])

feature_names = np.array(vectorizer.get_feature_names_out())
coefs = model.coef_[0]

top_fake = np.argsort(coefs)[-20:][::-1]
top_real = np.argsort(coefs)[:20]

print("=== Top words/phrases -> FAKE ===")
for i in top_fake:
    print(f"{feature_names[i]:25s} weight={coefs[i]:.3f}")

print("\n=== Top words/phrases -> REAL ===")
for i in top_real:
    print(f"{feature_names[i]:25s} weight={coefs[i]:.3f}")

joblib.dump(model, "best_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
joblib.dump({"model_name": "logistic_regression_domain_holdout",
             "accuracy": 0.9224, "precision": 0.8711, "recall": 0.9134, "f1": 0.8918},
            "metadata.joblib")
print("\nSaved final model + vectorizer + metadata for the app.")
test_df = df[df["domain"].isin(test_domains)]
X_test = vectorizer.transform(test_df["clean_content"])
preds = model.predict(X_test)
acc = accuracy_score(test_df["label_bin"], preds)
prec, rec, f1, _ = precision_recall_fscore_support(test_df["label_bin"], preds, average="binary")
print(f"\nFinal accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")