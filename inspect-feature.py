import joblib
import numpy as np

model = joblib.load("best_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

feature_names = np.array(vectorizer.get_feature_names_out())

# Works for both LogisticRegression and PassiveAggressiveClassifier —
# both expose a linear coef_ array of shape (1, n_features)
coefs = model.coef_[0]

top_fake_idx = np.argsort(coefs)[-25:][::-1]   # most positive = pushes toward FAKE (label=1)
top_real_idx = np.argsort(coefs)[:25]           # most negative = pushes toward REAL (label=0)

print("=== TOP 25 WORDS/PHRASES PUSHING TOWARD 'FAKE' ===")
for idx in top_fake_idx:
    print(f"{feature_names[idx]:30s}  weight={coefs[idx]:.3f}")

print("\n=== TOP 25 WORDS/PHRASES PUSHING TOWARD 'REAL' ===")
for idx in top_real_idx:
    print(f"{feature_names[idx]:30s}  weight={coefs[idx]:.3f}")