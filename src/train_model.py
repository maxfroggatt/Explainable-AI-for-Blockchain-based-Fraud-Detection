import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from elliptic_preprocessing import PreprocessConfig, build_subgraph_feature_table
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

def main():
    # Always resolve output directory from this file location
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("PROJECT ROOT:", project_root)
    print("OUTPUTS DIR:", outputs_dir)
    print("OUTPUTS DIR EXISTS:", outputs_dir.exists())

    cfg = PreprocessConfig(dataset_dir="dataset", outputs_dir="outputs")
    X, y = build_subgraph_feature_table(cfg)

    if "component_id" in X.columns:
        X_feat = X.drop(columns=["component_id"])
    else:
        X_feat = X

    le = LabelEncoder()
    y_enc = le.fit_transform(pd.Series(y).astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    metrics_path = outputs_dir / "metrics.txt"
    print("WRITING METRICS TO:", metrics_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))
        f.write("\n\nConfusion matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
        f.write("\n\nLabel mapping (encoded -> original):\n")
        for idx, clsname in enumerate(le.classes_):
            f.write(f"{idx} -> {clsname}\n")

    # hard verification
    print("METRICS EXISTS AFTER WRITE:", metrics_path.exists())
    if not metrics_path.exists():
        raise RuntimeError("metrics.txt was not created. Check permissions / antivirus / path.")

    joblib.dump(clf, outputs_dir / "rf_pipeline.joblib")
    joblib.dump(le, outputs_dir / "label_encoder.joblib")
    X_test.to_csv(outputs_dir / "X_test.csv", index=False)
    pd.Series(y_test, name="y").to_csv(outputs_dir / "y_test.csv", index=False)

    print("Saved metrics to", metrics_path)
    print(f"Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
    print("Saved model + label encoder + test split to", outputs_dir)

if __name__ == "__main__":
    main()
