import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler,
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


SCALING = ["none", "standard", "minmax", "robust", "yeo-johnson", "quantile-normal", "quantile-uniform"]
CATEGORICAL = ["PART_NAME", "EQUIP_CD", "EQUIP_NAME"]
EXCLUDED = {"PassOrFail", "class", "defect", "Reason", "Reason_clean", "PassOrFail_bin",
            "_id", "TimeStamp", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL"}


def load_data(path):
    data = pd.read_csv(path)
    if "PassOrFail" not in data:
        raise ValueError("PassOrFail column is missing")
    data["PassOrFail"] = data["PassOrFail"].astype(str).str.strip()
    if not set(data["PassOrFail"]).issubset({"Y", "N"}):
        raise ValueError("PassOrFail must contain only Y and N")
    return data


def make_xy(data):
    target = data["PassOrFail"].map({"Y": 0, "N": 1}).astype(int)
    columns = [c for c in data if c not in EXCLUDED and not c.startswith("Reason_")]
    return data[columns].copy(), target


def split_data(x, y, seed=42):
    return train_test_split(x, y, test_size=0.2, stratify=y, random_state=seed)


class ProcessFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, scaling="standard", seed=42):
        self.scaling = scaling
        self.seed = seed

    def fit(self, x, y=None):
        self.categorical_ = [c for c in CATEGORICAL if c in x]
        self.numeric_ = [c for c in x.select_dtypes(include=np.number)
                         if c not in EXCLUDED and x[c].nunique(dropna=True) > 1]
        self.median_ = x[self.numeric_].median()
        self.clip_columns_ = [c for c in ["Injection_Time", "Filling_Time", "Max_Injection_Speed", "Average_Screw_RPM"]
                              if c in self.numeric_]
        quartiles = x[self.clip_columns_].quantile([0.25, 0.75])
        spread = quartiles.loc[0.75] - quartiles.loc[0.25]
        self.lower_ = quartiles.loc[0.25] - 1.5 * spread
        self.upper_ = quartiles.loc[0.75] + 1.5 * spread
        self.group_columns_ = [c for c in ["PART_NAME", "EQUIP_CD"] if c in x]
        self.group_bounds_ = {}
        if self.group_columns_:
            keys = x[self.group_columns_].fillna("unknown").astype(str)
            for key, positions in keys.groupby(self.group_columns_, sort=False).indices.items():
                group = x.iloc[positions][self.clip_columns_]
                if len(group) < 20:
                    continue
                bounds = group.quantile([0.25, 0.75])
                iqr = bounds.loc[0.75] - bounds.loc[0.25]
                lower = (bounds.loc[0.25] - 1.5 * iqr).where(iqr > 0, -np.inf)
                upper = (bounds.loc[0.75] + 1.5 * iqr).where(iqr > 0, np.inf)
                self.group_bounds_[key] = (lower, upper)
        self.category_fill_ = {c: x[c].mode().iloc[0] if len(x[c].mode()) else "unknown"
                               for c in self.categorical_}
        numeric = self.numeric_frame(x)
        scalers = {
            "none": None, "standard": StandardScaler(), "minmax": MinMaxScaler(),
            "robust": RobustScaler(), "yeo-johnson": PowerTransformer(),
            "quantile-normal": QuantileTransformer(n_quantiles=min(100, len(x)), output_distribution="normal", random_state=self.seed),
            "quantile-uniform": QuantileTransformer(n_quantiles=min(100, len(x)), output_distribution="uniform", random_state=self.seed),
        }
        self.scaler_ = scalers[self.scaling]
        if self.scaler_ is not None:
            self.scaler_.fit(numeric)
        return self

    def numeric_frame(self, x):
        numeric = x[self.numeric_].replace([np.inf, -np.inf], np.nan).fillna(self.median_).copy()
        if self.group_columns_:
            keys = x[self.group_columns_].fillna("unknown").astype(str)
            for key, positions in keys.groupby(self.group_columns_, sort=False).indices.items():
                if key in self.group_bounds_:
                    lower, upper = self.group_bounds_[key]
                    columns = numeric.columns.get_indexer(self.clip_columns_)
                    numeric.iloc[positions, columns] = numeric.iloc[positions][self.clip_columns_].clip(lower, upper, axis=1)
        else:
            numeric[self.clip_columns_] = numeric[self.clip_columns_].clip(self.lower_, self.upper_, axis=1)
        if "Injection_Time" in numeric:
            numeric["Injection_Time_log"] = np.log1p(numeric["Injection_Time"].clip(lower=0))
        if "Filling_Time" in numeric:
            numeric["Filling_Time_sqrt"] = np.sqrt(numeric["Filling_Time"].clip(lower=0))
        return numeric

    def transform(self, x):
        numeric = self.numeric_frame(x)
        if self.scaler_ is not None:
            numeric = pd.DataFrame(self.scaler_.transform(numeric), columns=numeric.columns, index=x.index)
        for column in self.categorical_:
            numeric[column] = x[column].fillna(self.category_fill_[column]).astype(str).astype("category")
        return numeric


def make_pipeline(scaling, classifier, seed=42):
    encoder = ColumnTransformer([
        ("numeric", "passthrough", make_column_selector(dtype_include=np.number)),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         make_column_selector(dtype_include="category")),
    ])
    return Pipeline([
        ("features", ProcessFeatures(scaling, seed)),
        ("sampling", SMOTENC(categorical_features="auto", sampling_strategy=0.3, k_neighbors=3, random_state=seed)),
        ("encoding", encoder), ("model", classifier),
    ])


def classifiers(seed=42):
    return {
        "Logistic regression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1),
        "Decision tree": DecisionTreeClassifier(random_state=seed),
        "Gradient boosting": GradientBoostingClassifier(random_state=seed),
        "Linear SVM": LinearSVC(dual="auto", max_iter=10000, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                                 eval_metric="logloss", random_state=seed, n_jobs=1),
    }


def compare(candidates, x, y, folds, seed):
    rows = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {"f1": "f1", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}
    for name, pipeline in candidates.items():
        scores = cross_validate(pipeline, x, y, cv=splitter, scoring=scoring, error_score="raise", n_jobs=1)
        row = {"name": name}
        for metric in scoring:
            row[metric] = float(scores["test_" + metric].mean())
            row[metric + "_std"] = float(scores["test_" + metric].std())
        rows.append(row)
        print(f"{name}: defect F1={row['f1']:.4f}", flush=True)
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def save_eda(data, output):
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = data["PassOrFail"].value_counts().reindex(["Y", "N"], fill_value=0)
    axes[0].bar(["Normal", "Defect"], counts, color=["#4477aa", "#cc6677"])
    for i, value in enumerate(counts):
        axes[0].text(i, value, str(value), ha="center", va="bottom")
    axes[0].set(ylabel="Production records", title="Class distribution")
    axes[1].hist(data["Cycle_Time"].dropna(), bins=30, color="#4477aa")
    axes[1].set(xlabel="Cycle time (s)", ylabel="Records", title="Cycle time")
    figure.tight_layout()
    figure.savefig(output / "data_overview.png", dpi=150)
    plt.close(figure)

    selected = [c for c in ["Injection_Time", "Filling_Time", "Cycle_Time", "Max_Injection_Pressure",
                            "Max_Injection_Speed", "Average_Screw_RPM", "Hopper_Temperature"] if c in data]
    correlation = data[selected].corr()
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set(xticks=range(len(selected)), yticks=range(len(selected)), xticklabels=selected, yticklabels=selected)
    plt.setp(axis.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    plt.setp(axis.get_yticklabels(), fontsize=8)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output / "correlations.png", dpi=150)
    plt.close(figure)


def run_experiment(data, output, folds=5, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, y = make_xy(data)
    x_train, x_test, y_train, y_test = split_data(x, y, seed)
    models = classifiers(seed)
    candidates = {name: make_pipeline(name, models["Logistic regression"], seed) for name in SCALING}
    scaling_results = compare(candidates, x_train, y_train, folds, seed)
    best_scaling = scaling_results.iloc[0]["name"]
    model_candidates = {name: make_pipeline(best_scaling, model, seed) for name, model in models.items()}
    model_results = compare(model_candidates, x_train, y_train, folds, seed)
    best_name = model_results.iloc[0]["name"]
    best = model_candidates[best_name].fit(x_train, y_train)
    predictions = best.predict(x_test)
    scores = best.predict_proba(x_test)[:, 1] if hasattr(best, "predict_proba") else best.decision_function(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "positive_class": "defect (N)", "seed": seed, "folds": folds,
        "rows": len(data), "train_rows": len(x_train), "test_rows": len(x_test),
        "defects_train": int(y_train.sum()), "defects_test": int(y_test.sum()),
        "best_scaling": best_scaling, "best_classifier": best_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores), "average_precision": average_precision_score(y_test, scores),
        "confusion_matrix": matrix.tolist(),
        "report": classification_report(y_test, predictions, target_names=["normal", "defect"], output_dict=True, zero_division=0),
    }
    scaling_results.to_csv(output / "preprocessing_cv.csv", index=False)
    model_results.to_csv(output / "classifier_cv.csv", index=False)
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"source_row": x_test.index, "defect_actual": y_test, "defect_predicted": predictions,
                  "score": scores}).to_csv(output / "test_predictions.csv", index=False)
    save_eda(data, output)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ordered = model_results.iloc[::-1]
    axes[0].barh(ordered["name"], ordered["f1"], xerr=ordered["f1_std"], color="#4477aa")
    axes[0].set(xlabel="Defect F1 (CV mean and standard deviation)", xlim=(0, 1))
    axes[1].imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
            axes[1].text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Defect"],
                yticklabels=["Normal", "Defect"], xlabel="Predicted", ylabel="Actual", title=f"Test / {best_name}")
    figure.tight_layout()
    figure.savefig(output / "model_comparison.png", dpi=150)
    plt.close(figure)

    features = best.named_steps["features"]
    clipped = features.numeric_frame(x_train)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(x_train["Injection_Time"], bins=30, color="#4477aa")
    axes[0].set(title="Training data: raw", xlabel="Injection time (s)")
    axes[1].hist(clipped["Injection_Time_log"], bins=30, color="#4477aa")
    axes[1].set(title="Training data: IQR clipping + log1p", xlabel="log1p(injection time)")
    figure.tight_layout()
    figure.savefig(output / "preprocessing.png", dpi=150)
    plt.close(figure)
    print(json.dumps({key: value for key, value in metrics.items() if key not in ["report"]}, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "data" / "labeled_data.csv")
    parser.add_argument("--output", type=Path, default=root / "results" / "current")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    run_experiment(load_data(args.data), args.output, args.folds, args.seed)
