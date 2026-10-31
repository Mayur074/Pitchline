"""Tests for the experiment design / hypothesis testing module."""

import pytest
import numpy as np
import pandas as pd

from src.models.experiment_design import (
    run_home_advantage_experiment,
    run_model_calibration_experiment,
    run_all_experiments,
    ExperimentResult,
    SIGNIFICANCE_LEVEL,
)


class TestHomeAdvantageExperiment:
    def test_returns_experiment_result(self):
        result = run_home_advantage_experiment()
        assert isinstance(result, ExperimentResult)

    def test_p_value_in_valid_range(self):
        result = run_home_advantage_experiment()
        assert 0.0 <= result.p_value <= 1.0

    def test_significance_flag_matches_p_value(self):
        result = run_home_advantage_experiment()
        assert result.significant == (result.p_value < SIGNIFICANCE_LEVEL)

    def test_sample_size_is_positive(self):
        result = run_home_advantage_experiment()
        assert result.sample_size > 0

    def test_excludes_neutral_venue_matches(self):
        """
        Sanity check: most World Cup matches are at neutral venues, so the
        non-neutral sample should be meaningfully smaller than the full
        dataset (552+ matches), not accidentally include everything.
        """
        result = run_home_advantage_experiment()
        assert result.sample_size < 200  # true home-venue matches are rare in World Cup play

    def test_real_finding_home_advantage_is_positive_and_significant(self):
        """
        This is a real, substantive finding from the real dataset: home
        teams should show a meaningfully higher win rate than away teams
        in actual home-venue World Cup matches.
        """
        result = run_home_advantage_experiment()
        assert result.effect_size > 0  # home win rate higher than away win rate
        assert result.significant  # and it's statistically significant


class TestModelCalibrationExperiment:
    def test_returns_result_and_bucket_dataframe(self):
        result, buckets = run_model_calibration_experiment()
        assert isinstance(result, ExperimentResult)
        assert isinstance(buckets, pd.DataFrame)

    def test_chi_square_does_not_raise(self):
        """
        Regression test: scipy.chisquare requires observed and expected
        totals to match exactly. An earlier version of this function did
        not normalize for this and raised a ValueError at runtime — this
        test locks in that the fix (normalizing expected_wins to share the
        observed total) actually works.
        """
        result, _ = run_model_calibration_experiment()
        assert not np.isnan(result.test_statistic)
        assert not np.isnan(result.p_value)

    def test_p_value_in_valid_range(self):
        result, _ = run_model_calibration_experiment()
        assert 0.0 <= result.p_value <= 1.0

    def test_bucket_summary_has_expected_columns(self):
        _, buckets = run_model_calibration_experiment()
        expected = {"bucket", "mean_predicted", "observed_win_rate", "n_matches"}
        assert expected.issubset(set(buckets.columns))

    def test_bucket_win_rates_are_valid_probabilities(self):
        _, buckets = run_model_calibration_experiment()
        assert (buckets["observed_win_rate"] >= 0).all()
        assert (buckets["observed_win_rate"] <= 1).all()

    def test_sample_size_covers_full_real_history(self):
        """Should process all real matches (552 historical + real 2026 so far)."""
        result, _ = run_model_calibration_experiment()
        assert result.sample_size > 550

    def test_different_bucket_counts_both_run_without_error(self):
        result_5, _ = run_model_calibration_experiment(n_buckets=5)
        result_10, _ = run_model_calibration_experiment(n_buckets=10)
        assert result_5.sample_size == result_10.sample_size


class TestRunAllExperiments:
    def test_returns_two_results(self):
        results = run_all_experiments()
        assert len(results) == 2

    def test_each_result_has_required_fields(self):
        results = run_all_experiments()
        for r in results:
            assert r.experiment_name
            assert r.null_hypothesis
            assert r.alternative_hypothesis
            assert r.test_used
            assert r.sample_size > 0
