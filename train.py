import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from preprocess import clean_hindi_text

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv("data/hinfakenews_clean.csv")
    df["content"] = df["TITLE"].fillna("") + " " + df["CONTENT"].fillna("")
    print("Cleaning Hindi text (this takes a minute on 60k+ rows)...")
    df["clean_content"] = df["content"].apply(clean_hindi_text)
    df["label_bin"] = (df["label"] == "FAKE").astype(int)  # FAKE=1, REAL=0
    return df


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"], df["label_bin"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label_bin"]
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    vectorizer = TfidfVectorizer(
    max_df=0.7, ngram_range=(1, 2), max_features=50000,
    token_pattern=r"[^\s]+"   # treat any whitespace-separated chunk as one token
                               # (default \w+ pattern breaks Devanagari matras — see note below)
)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    results = {}

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(X_train_tfidf, y_train)
    results["logistic_regression"] = evaluate("Logistic Regression", y_test, lr.predict(X_test_tfidf))

    pac = PassiveAggressiveClassifier(max_iter=50, class_weight="balanced", random_state=RANDOM_STATE)
    pac.fit(X_train_tfidf, y_train)
    results["passive_aggressive"] = evaluate("Passive-Aggressive", y_test, pac.predict(X_test_tfidf))

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = lr if best_name == "logistic_regression" else pac
    print(f"\n>>> Best model: {best_name} (F1={results[best_name]['f1']:.4f})")

    joblib.dump(best_model, "best_model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    joblib.dump({"best_model_name": best_name, "results": results}, "metadata.joblib")
    print("Saved model, vectorizer, metadata.")


if __name__ == "__main__":
    main()