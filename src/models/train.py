"""
Trains an XGBoost multiclass classifier to predict match outcomes
(home win / draw / away win) from engineered features.

Includes walk-forward (time-based) validation rather than random k-fold,
since shuffling football match data randomly leaks future information
into the training set.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
import mlflow
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "elo_diff", "home_elo_pre", "away_elo_pre",
    "form_diff", "home_form", "away_form",
    "rest_diff", "h2h_home_win_rate", "is_neutral", "is_knockout",
]


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits chronologically: oldest matches train, most recent matches test.
    This mirrors the real deployment scenario (predicting future matches from
    past data) and avoids the data leakage a random split would introduce.
    """
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_model(
    feature_df: pd.DataFrame,
    n_estimators: int = 250,
    max_depth: int = 4,
    learning_rate: float = 0.05,
) -> tuple[xgb.XGBClassifier, StandardScaler, dict]:
    """
    Trains the XGBoost classifier with walk-forward validation and logs
    the run to MLflow for experiment tracking.

    Returns the fitted model, the fitted feature scaler, and a metrics dict.
    """
    train_df, test_df = time_based_split(feature_df)

    X_train, y_train = train_df[FEATURE_COLS], train_df["result"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["result"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with mlflow.start_run(run_name="xgboost_match_outcome_v1"):
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "train_size": len(train_df),
            "test_size": len(test_df),
        })

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)

        accuracy = accuracy_score(y_test, preds)
        logloss = log_loss(y_test, probs)

        # Brier score computed per-class (one-vs-rest) and averaged
        y_test_onehot = pd.get_dummies(y_test).reindex(columns=[0, 1, 2], fill_value=0).values
        brier = np.mean([
            brier_score_loss(y_test_onehot[:, i], probs[:, i]) for i in range(3)
        ])

        baseline_preds = (test_df["elo_diff"] > 0).astype(int) * 2  # naive: favorite always wins
        baseline_preds = baseline_preds.replace({0: 0, 2: 2})
        baseline_acc = accuracy_score(y_test, baseline_preds.clip(0, 2))

        metrics = {
            "accuracy": accuracy,
            "log_loss": logloss,
            "brier_score": brier,
            "baseline_accuracy": baseline_acc,
        }

        mlflow.log_metrics(metrics)

        importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        logger.info("Feature importances:\n%s", importances)

        logger.info("Validation accuracy: %.3f (baseline: %.3f)", accuracy, baseline_acc)
        logger.info("Log loss: %.3f | Brier score: %.3f", logloss, brier)

        mlflow.xgboost.log_model(model, "model")

    joblib.dump(model, MODEL_DIR / "match_predictor.joblib")
    joblib.dump(scaler, MODEL_DIR / "feature_scaler.joblib")

    return model, scaler, metrics


if __name__ == "__main__":
    feature_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "match_features.csv"
    features = pd.read_csv(feature_path, parse_dates=["date"])

    model, scaler, metrics = train_model(features)
    print("Training complete.")
    print(metrics)
