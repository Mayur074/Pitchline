"""Tests for model training and evaluation."""

import pandas as pd
import numpy as np
import pytest

from src.models.train import time_based_split, train_model, FEATURE_COLS


@pytest.fixture
def synthetic_feature_data():
    """Larger synthetic dataset sufficient for training a real model."""
    np.random.seed(0)
    n = 500
    dates = pd.date_range("2015-01-01", periods=n, freq="3D")

    elo_diff = np.random.normal(0, 150, n)
    # outcome correlated with elo_diff so the model has real signal to learn
    win_prob = 1 / (1 + np.exp(-elo_diff / 200))
    results = []
    for p in win_prob:
        roll = np.random.random()
        if roll < p * 0.7:
            results.append(2)
        elif roll > 1 - (1 - p) * 0.7:
            results.append(0)
        else:
            results.append(1)

    df = pd.DataFrame({
        "date": dates,
        "home_team": np.random.choice(["A", "B", "C", "D"], n),
        "away_team": np.random.choice(["E", "F", "G", "H"], n),
        "home_score": np.random.randint(0, 4, n),
        "away_score": np.random.randint(0, 4, n),
        "elo_diff": elo_diff,
        "home_elo_pre": 1500 + elo_diff / 2,
        "away_elo_pre": 1500 - elo_diff / 2,
        "form_diff": np.random.normal(0, 1, n),
        "home_form": np.random.uniform(0, 3, n),
        "away_form": np.random.uniform(0, 3, n),
        "rest_diff": np.random.normal(0, 5, n),
        "h2h_home_win_rate": np.random.uniform(0, 1, n),
        "is_neutral": np.random.randint(0, 2, n),
        "is_knockout": np.random.randint(0, 2, n),
        "result": results,
    })
    return df


class TestTimeBasedSplit:
    def test_split_preserves_chronological_order(self, synthetic_feature_data):
        train, test = time_based_split(synthetic_feature_data, test_fraction=0.2)
        assert train["date"].max() <= test["date"].min()

    def test_split_proportions_approximately_correct(self, synthetic_feature_data):
        train, test = time_based_split(synthetic_feature_data, test_fraction=0.2)
        total = len(synthetic_feature_data)
        assert abs(len(test) / total - 0.2) < 0.02

    def test_no_overlap_between_train_and_test(self, synthetic_feature_data):
        train, test = time_based_split(synthetic_feature_data)
        assert set(train.index).isdisjoint(set(test.index))


class TestTrainModel:
    def test_model_trains_without_error(self, synthetic_feature_data):
        model, scaler, metrics = train_model(
            synthetic_feature_data, n_estimators=20, max_depth=3
        )
        assert model is not None
        assert scaler is not None

    def test_metrics_contain_expected_keys(self, synthetic_feature_data):
        _, _, metrics = train_model(synthetic_feature_data, n_estimators=20)
        for key in ["accuracy", "log_loss", "brier_score", "baseline_accuracy"]:
            assert key in metrics

    def test_accuracy_is_valid_probability_range(self, synthetic_feature_data):
        _, _, metrics = train_model(synthetic_feature_data, n_estimators=20)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_model_predicts_three_classes(self, synthetic_feature_data):
        model, scaler, _ = train_model(synthetic_feature_data, n_estimators=20)
        X = synthetic_feature_data[FEATURE_COLS].iloc[:5]
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)
        assert probs.shape[1] == 3

    def test_probabilities_sum_to_one(self, synthetic_feature_data):
        model, scaler, _ = train_model(synthetic_feature_data, n_estimators=20)
        X = synthetic_feature_data[FEATURE_COLS].iloc[:10]
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_model_beats_random_guessing(self, synthetic_feature_data):
        """With real signal in elo_diff, model should clearly beat 1/3 random baseline."""
        _, _, metrics = train_model(synthetic_feature_data, n_estimators=50, max_depth=4)
        assert metrics["accuracy"] > 0.40  # meaningfully above 33% random baseline
