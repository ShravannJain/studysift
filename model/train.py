import os
import csv
import pickle
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train")

DATA_CSV   = os.path.join(os.path.dirname(__file__), "..", "data", "transcripts.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def load_data():
    rows = []
    with open(DATA_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    log.info("Loaded %d rows", len(rows))
    return rows


def extract_features(rows, kw_vectorizer=None, sum_vectorizer=None, fit=False):
    """
    Builds a feature matrix by combining:
      - TF-IDF on keywords
      - TF-IDF on summary
      - Numeric features (duration, length, efficiency, has_code, has_formulas, density)
    """
    keywords  = [r.get("keywords", "") for r in rows]
    summaries = [r.get("summary", "") for r in rows]

    numeric = []
    for r in rows:
        duration     = float(r.get("total_minutes", 0))
        length       = int(r.get("transcript_length", 0))
        efficiency   = int(r.get("efficiency_score", 0))
        has_code     = int(r.get("has_code", 0))
        has_formulas = int(r.get("has_formulas", 0))
        density      = length / max(duration, 0.1)
        numeric.append([duration, length, efficiency, has_code, has_formulas, density])

    numeric_matrix = csr_matrix(np.array(numeric, dtype=float))

    if fit:
        kw_vectorizer  = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
        sum_vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
        kw_matrix  = kw_vectorizer.fit_transform(keywords)
        sum_matrix = sum_vectorizer.fit_transform(summaries)
    else:
        kw_matrix  = kw_vectorizer.transform(keywords)
        sum_matrix = sum_vectorizer.transform(summaries)

    X = hstack([kw_matrix, sum_matrix, numeric_matrix])
    return X, kw_vectorizer, sum_vectorizer


def main():
    rows   = load_data()
    labels = np.array([int(r["should_watch"]) for r in rows])

    watch = labels.sum()
    skip  = len(labels) - watch
    log.info("Watch: %d | Skip: %d", watch, skip)

    if skip < 3 or watch < 3:
        log.error("Need at least 3 of each class. Add more skip examples.")
        return

    # ── Build features ────────────────────────────────────────────────
    X, kw_vec, sum_vec = extract_features(rows, fit=True)

    # ── Cross-validation ──────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )

    n_splits = min(3, int(skip), int(watch))
    cv       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores   = cross_val_score(clf, X, labels, cv=cv, scoring="f1")
    log.info("Cross-val F1 scores: %s", np.round(scores, 2))
    log.info("Mean F1: %.2f  (±%.2f)", scores.mean(), scores.std())

    # ── Train on full data ────────────────────────────────────────────
    clf.fit(X, labels)
    preds = clf.predict(X)

    print("\n── Training Report ──────────────────────────────")
    print(classification_report(labels, preds, target_names=["Skip", "Watch"]))
    cm = confusion_matrix(labels, preds)
    print("Confusion matrix:")
    print("             Skip  Watch")
    print(f"  Actual Skip  {cm[0]}")
    print(f"  Actual Watch {cm[1]}")

    # ── Save everything needed for inference ─────────────────────────
    bundle = {
        "clf":            clf,
        "kw_vectorizer":  kw_vec,
        "sum_vectorizer": sum_vec,
    }
    pickle.dump(bundle, open(MODEL_PATH, "wb"))
    log.info("✅ Model saved → %s", MODEL_PATH)

    # ── Quick sanity test ─────────────────────────────────────────────
    print("\n── Quick Prediction Test ────────────────────────")
    test_cases = [
        {
            "keywords": "reaction video, funny moments, memes, entertainment",
            "summary":  "Guy reacts to viral videos and laughs at memes",
            "total_minutes": "15", "transcript_length": "2000",
            "efficiency_score": "20", "has_code": "0", "has_formulas": "0",
        },
        {
            "keywords": "neural networks, backpropagation, PyTorch, deep learning, gradient descent",
            "summary":  "Deep dive into backpropagation with code examples in PyTorch",
            "total_minutes": "45", "transcript_length": "8000",
            "efficiency_score": "85", "has_code": "1", "has_formulas": "1",
        },
        {
            "keywords": "morning routine, hustle, success mindset, motivation, productivity",
            "summary":  "Influencer shares morning routine and motivational tips",
            "total_minutes": "12", "transcript_length": "1500",
            "efficiency_score": "30", "has_code": "0", "has_formulas": "0",
        },
    ]
    descriptions = [
        "Reaction / meme video",
        "Neural networks deep dive",
        "Morning routine / motivation fluff",
    ]

    for desc, test in zip(descriptions, test_cases):
        X_test, _, _ = extract_features(
            [test], kw_vectorizer=kw_vec, sum_vectorizer=sum_vec, fit=False
        )
        pred = clf.predict(X_test)[0]
        prob = clf.predict_proba(X_test)[0]
        result = "✅ WATCH" if pred == 1 else "❌ SKIP"
        print(f"{result} ({max(prob)*100:.0f}% confidence) ← {desc}")

    print("\n🎉 Model ready! Next: integrate into FastAPI backend.")


if __name__ == "__main__":
    main()