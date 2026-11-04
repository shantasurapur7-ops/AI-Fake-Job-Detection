from pathlib import Path
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

DATA_PATH = Path("data/job_posts.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/job_posts.csv was not found. Add a permitted, labeled dataset with columns "
            "'text' and 'label'. Labels must be 'fraudulent' or 'legitimate'."
        )

    df = pd.read_csv(DATA_PATH)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain columns named 'text' and 'label'.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"].str.len() > 0]
    allowed = {"fraudulent", "legitimate"}
    if not set(df["label"].unique()).issubset(allowed):
        raise ValueError("Labels must be exactly 'fraudulent' or 'legitimate'.")
    if df["label"].nunique() != 2 or len(df) < 10:
        raise ValueError("Provide at least 10 rows and examples from both classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    candidates = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=20000)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=20000)),
            ("classifier", MultinomialNB())
        ])
    }

    results = {}
    best_name, best_model, best_f1 = None, None, -1

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_fraudulent": float(precision_score(
                y_test, predictions, pos_label="fraudulent", zero_division=0
            )),
            "recall_fraudulent": float(recall_score(
                y_test, predictions, pos_label="fraudulent", zero_division=0
            )),
            "f1_fraudulent": float(f1_score(
                y_test, predictions, pos_label="fraudulent", zero_division=0
            )),
            "classification_report": classification_report(
                y_test, predictions, zero_division=0
            )
        }
        results[name] = metrics
        if metrics["f1_fraudulent"] > best_f1:
            best_name, best_model, best_f1 = name, model, metrics["f1_fraudulent"]

    joblib.dump(best_model, MODEL_DIR / "fake_job_pipeline.joblib")

    with open(REPORT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"selected_model": best_name, "models": results}, f, indent=2)

    predictions = best_model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=["fraudulent", "legitimate"], values_format="d"
    )
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print(f"Selected model: {best_name}")
    print("Evaluation results (held-out test split):")
    for name, metrics in results.items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.3f}, "
            f"precision={metrics['precision_fraudulent']:.3f}, "
            f"recall={metrics['recall_fraudulent']:.3f}, "
            f"F1={metrics['f1_fraudulent']:.3f}"
        )
    print("\nSaved model to models/fake_job_pipeline.joblib")
    print("Saved evaluation report and confusion matrix in reports/")

if __name__ == "__main__":
    main()
