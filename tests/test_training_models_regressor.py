"""Unit tests for the regressor model wrappers and their base class.

The regressors (XGBoostRegressor, RandomForestRegressor) train on proportion
labels — floats in [0, 1] produced by the profile-level pipeline's
``label_mode: proportion``. Structural wrapper tests mirror
``test_training_models.py`` via ``REGRESSOR_CASES``; the functional tests
exercise the ``SklearnRegressorModelBase`` lifecycle on a small synthetic
frame: clipped scores, thresholded predicted labels, regression report rows,
SHAP output shape, and the constant-label warning.
"""

import numpy as np
import polars as pl
import pytest

from aiqclib.train.models.model_suite import ModelSuite

from tests._model_cases import REGRESSOR_CASES


@pytest.fixture
def proportion_sets():
    """A small synthetic training/test pair with proportion labels."""
    rng = np.random.default_rng(42)
    n_train, n_test = 80, 20

    def _make(n):
        feature_a = rng.normal(size=n)
        feature_b = rng.normal(size=n)
        # Label correlated with feature_a so the model has signal to fit.
        label = np.clip(0.5 + 0.3 * feature_a + rng.normal(scale=0.1, size=n), 0, 1)
        return pl.DataFrame(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "label": label,
            }
        )

    return _make(n_train), _make(n_test)


def _build_regressor(case, config, training_set):
    config.data["step_class_set"]["steps"]["model"] = case.config_name
    model = case.wrapper_cls(config)
    model.training_set = training_set
    model.build()
    return model


@pytest.mark.parametrize("case", REGRESSOR_CASES, ids=lambda c: c.config_name)
class TestRegressorWrappers:
    """Structural checks mirroring TestModelWrappers for the regressors."""

    def test_init_class(self, case, training_config_001):
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        ds = case.wrapper_cls(training_config_001)
        assert ds.expected_class_name == case.wrapper_cls.__name__
        assert ds._get_model_class() == case.sklearn_cls
        assert ds.is_regressor is True
        assert ds.multi is False

    def test_default_params(self, case, training_config_001):
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        ds = case.wrapper_cls(training_config_001)
        for key, expected_value in case.defaults.items():
            assert ds.model_params.get(key) == expected_value
        for key in case.missing:
            assert key not in ds.model_params

    def test_config_params_override(self, case, training_config_001):
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        training_config_001.data["step_param_set"]["steps"]["model"]["model_params"] = (
            dict(case.override)
        )
        ds = case.wrapper_cls(training_config_001)
        for key, expected_value in case.override.items():
            assert ds.model_params[key] == expected_value


@pytest.mark.parametrize("case", REGRESSOR_CASES, ids=lambda c: c.config_name)
class TestRegressorLifecycle:
    """Functional checks of build/predict/report/SHAP on proportion labels."""

    def test_predict_scores_and_labels(
        self, case, training_config_001, proportion_sets
    ):
        """Scores are clipped to [0, 1]; predicted_label thresholds the score."""
        training_set, test_set = proportion_sets
        model = _build_regressor(case, training_config_001, training_set)
        model.test_set = test_set
        model.predict()

        scores = model.predictions["score"]
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

        expected_labels = (scores >= model.predicted_label_threshold).cast(pl.Int64)
        assert (
            model.predictions["predicted_label"].cast(pl.Int64) == expected_labels
        ).all()

    def test_report_metrics(self, case, training_config_001, proportion_sets):
        """The report holds mae / rmse / r2 / n_samples rows."""
        training_set, test_set = proportion_sets
        model = _build_regressor(case, training_config_001, training_set)
        model.test_set = test_set
        model.test()

        metric_types = model.report["metric_type"].to_list()
        assert metric_types == ["mae", "rmse", "r2", "n_samples"]
        values = dict(zip(metric_types, model.report["value"].to_list()))
        assert values["mae"] >= 0.0
        assert values["rmse"] >= values["mae"] - 1e-12
        assert values["n_samples"] == test_set.height

    def test_model_score_rows(self, case, training_config_001, proportion_sets):
        """update_model_score keeps float labels alongside the scores."""
        training_set, test_set = proportion_sets
        model = _build_regressor(case, training_config_001, training_set)
        model.test_set = test_set
        model.test()

        assert model.model_score is not None
        assert model.model_score.height == test_set.height
        assert model.model_score["label"].dtype.is_float()

    def test_shap_output(self, case, training_config_001, proportion_sets):
        """SHAP values come back 2-D: one _shap column per feature."""
        training_set, test_set = proportion_sets
        model = _build_regressor(case, training_config_001, training_set)
        model.enable_shap = True
        model.test_set = test_set
        model.test()

        assert model.shap_values is not None
        assert model.shap_values.height == test_set.height
        assert {"feature_a_shap", "feature_b_shap"} <= set(model.shap_values.columns)

    def test_constant_labels_warn_not_raise(self, case, training_config_001):
        """A constant proportion label warns instead of refusing to fit."""
        constant_set = pl.DataFrame(
            {
                "feature_a": [0.1, 0.2, 0.3, 0.4],
                "label": [0.5, 0.5, 0.5, 0.5],
            }
        )
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        model = case.wrapper_cls(training_config_001)
        model.training_set = constant_set
        with pytest.warns(UserWarning, match="constant"):
            model.build()


class TestModelSuiteHomogeneity:
    """ModelSuite refuses method sets mixing classifiers and regressors."""

    def test_mixed_methods_rejected(self, training_config_001):
        training_config_001.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        training_config_001.data["step_param_set"]["steps"]["model"]["methods"] = [
            "XGB",
            "XGBR",
        ]
        with pytest.raises(ValueError, match="mix classifiers and regressors"):
            ModelSuite(training_config_001)

    def test_all_regressor_suite_loads(self, training_config_001):
        training_config_001.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        training_config_001.data["step_param_set"]["steps"]["model"]["methods"] = [
            "XGBR",
            "RFR",
        ]
        suite = ModelSuite(training_config_001)
        assert set(suite.method_objs) == {"XGBR", "RFR"}
