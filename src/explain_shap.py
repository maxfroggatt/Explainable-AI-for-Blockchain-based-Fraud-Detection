from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def main():
    # resolve project root and outputs directory
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("PROJECT ROOT:", project_root)
    print("OUTPUTS DIR:", outputs_dir)

    # load trained model artifacts
    clf = joblib.load(outputs_dir / "rf_pipeline.joblib")
    le = joblib.load(outputs_dir / "label_encoder.joblib")

    # load test data
    X_test = pd.read_csv(outputs_dir / "X_test.csv")

    # use a subset for speed
    n_samples = min(500, len(X_test))
    X_test = X_test.sample(n=n_samples, random_state=42)
    print(f"Using {len(X_test)} samples for SHAP.")

    # extract pipeline parts
    scaler = clf.named_steps["scaler"]
    rf = clf.named_steps["rf"]

    X_test_scaled = scaler.transform(X_test)

    # create SHAP explainer
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test_scaled)

    # handle binary/multiclass SHAP outputs safely
    if isinstance(shap_values, list):
        # choose suspicious class if available, else first class
        class_index = 1 if len(shap_values) > 1 else 0
        sv = shap_values[class_index]

        # extract base value safely
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = explainer.expected_value[class_index]
        else:
            base_value = explainer.expected_value

        class_name = (
            le.inverse_transform([class_index])[0]
            if len(le.classes_) > class_index
            else f"class_{class_index}"
        )
    else:
        sv = shap_values
        base_value = explainer.expected_value
        class_name = "model_output"

    # global SHAP summary plot
    plt.figure()
    shap.summary_plot(
        sv,
        X_test_scaled,
        feature_names=list(X_test.columns),
        show=False
    )
    summary_path = outputs_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP summary plot to: {summary_path}")
    print("SUMMARY PNG EXISTS:", summary_path.exists())
    print("Explained class:", class_name)

    # waterfall plots (optional)
    sample_indices = [0, 1] if len(X_test_scaled) >= 2 else [0]

    # ensure base value is scalar
    if isinstance(base_value, (list, np.ndarray)):
        base_scalar = float(np.array(base_value).flatten()[0])
    else:
        base_scalar = float(base_value)

    for i in sample_indices:
        vals = sv[i]

        # if SHAP values are (features x classes), select class column
        if isinstance(vals, np.ndarray) and vals.ndim == 2:
            col = 1 if vals.shape[1] > 1 else 0
            vals_1d = vals[:, col]
        else:
            vals_1d = vals

        explanation = shap.Explanation(
            values=np.array(vals_1d, dtype=float),
            base_values=base_scalar,
            data=np.array(X_test_scaled[i], dtype=float),
            feature_names=list(X_test.columns),
        )

        plt.figure()
        shap.plots.waterfall(explanation, show=False)
        local_path = outputs_dir / f"shap_local_{i}.png"
        plt.tight_layout()
        plt.savefig(local_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved local SHAP plot to: {local_path}")
        print("LOCAL PNG EXISTS:", local_path.exists())

if __name__ == "__main__":
    main()
