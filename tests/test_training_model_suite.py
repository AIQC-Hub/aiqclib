"""Unit tests for the ``ModelSuite`` wrapper class.

ModelSuite is the multi-method wrapper consumed by BuildModelSuite,
KFoldValidationSuite, and ClassifyAllSuite. Tests verify:
- ``multi`` flag is True (single-model wrappers have it False)
- ``calculate_shap`` IS propagated (this is the wrapper's behaviour;
  individual step classes — like KFoldValidationSuite — may override)
- Default method-loading: all 9 default methods when no ``methods`` list
  is specified in config
- Custom method-loading: only the explicitly listed methods are loaded,
  with method-specific ``model_params`` correctly attached
- Both short (``XGB``, ``DT``) and long (``XGBoost``, ``DecisionTree``)
  method-name forms work
- Per-method overrides actually take effect

Refactored from a ``unittest.TestCase`` class. Each test mutates
``training_config_001`` differently, so a single conftest fixture doesn't
fit; a file-local ``training_config_001_modelsuite`` fixture applies just
the basic ``model=ModelSuite`` mutation that every test needs.

Note on SHAP behaviour:
ModelSuite itself **does** propagate the ``calculate_shap`` flag to its
underlying method objects. KFoldValidationSuite overrides this at the
step level (validation never uses SHAP); BuildModelSuite respects it
(SHAP is computed at the testing stage). This file tests the wrapper's
default behaviour, hence True propagation.
"""

import pytest

from aiqclib.train.models.model_suite import ModelSuite


# ---------------------------------------------------------------------------
# Suite-config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def training_config_001_modelsuite(training_config_001):
    """training_config_001 with just ``model=ModelSuite`` set.

    Narrower mutation than the test files for BuildModelSuite /
    KFoldValidationSuite (which also set the build/validate step class and
    a methods list). Each test mutates ``step_param_set.steps.model``
    further to specify methods and per-method params.
    """
    training_config_001.data["step_class_set"]["steps"]["model"] = "ModelSuite"
    return training_config_001


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelSuite:
    """Tests for ModelSuite's multi flag, SHAP propagation, and method loading."""

    # ----- Flags -----

    def test_multi_flag(self, training_config_001_modelsuite):
        """ModelSuite has ``multi == True`` (vs single-model wrappers' False)."""
        ds = ModelSuite(training_config_001_modelsuite)
        assert ds.multi is True

    def test_shap_flag(self, training_config_001_modelsuite):
        """``calculate_shap`` propagates from config to suite + each method object.

        Contrast with KFoldValidationSuite's test_shap_flag, which asserts
        the flag is NOT propagated for validation. That's because the
        validation step *overrides* this wrapper-level propagation; here
        we test the wrapper's own propagation behaviour.
        """
        # Unset == False, at both suite level and per-method.
        ds = ModelSuite(training_config_001_modelsuite)
        assert ds.enable_shap is False
        for method_obj in ds.method_objs.values():
            assert method_obj.enable_shap is False

        # True at suite level propagates to every method.
        training_config_001_modelsuite.data["step_param_set"]["steps"]["model"][
            "calculate_shap"
        ] = True
        ds = ModelSuite(training_config_001_modelsuite)
        assert ds.enable_shap is True
        for method_obj in ds.method_objs.values():
            assert method_obj.enable_shap is True

        # False at suite level also propagates.
        training_config_001_modelsuite.data["step_param_set"]["steps"]["model"][
            "calculate_shap"
        ] = False
        ds = ModelSuite(training_config_001_modelsuite)
        assert ds.enable_shap is False
        for method_obj in ds.method_objs.values():
            assert method_obj.enable_shap is False

    # ----- Method loading -----

    def test_init_default_methods(self, training_config_001_modelsuite):
        """With no ``methods`` list, ModelSuite loads all 9 default methods.

        The default list is exposed as ``suite.default_methods``; each name
        appears as a key in ``method_objs`` with a non-None value.
        """
        training_config_001_modelsuite.data["step_param_set"]["steps"]["model"] = {}
        suite = ModelSuite(training_config_001_modelsuite)

        assert suite.expected_class_name == "ModelSuite"
        assert suite.short_name == "MS"

        # Exactly 9 default methods, all present and instantiated.
        assert len(suite.method_objs) == 9
        for method_name in suite.default_methods:
            assert method_name in suite.method_objs
            assert suite.method_objs[method_name] is not None

    def test_init_custom_methods_with_params(self, training_config_001_modelsuite):
        """Explicitly listed short-form methods load with per-method ``model_params``.

        Three methods (DT, XGB, RF) are loaded. DT and RF receive a
        ``class_weight`` override; XGB doesn't (and its own defaults,
        like ``n_jobs``, are preserved).
        """
        training_config_001_modelsuite.data["step_param_set"]["steps"]["model"] = {
            "methods": ["DT", "XGB", "RF"],
            "model_params": {
                "DT": {"class_weight": "balanced"},
                "RF": {"class_weight": "balanced"},
            },
        }
        suite = ModelSuite(training_config_001_modelsuite)

        # All 3 requested methods loaded, nothing else.
        assert len(suite.method_objs) == 3
        for name in ("DT", "XGB", "RF"):
            assert name in suite.method_objs

        # Per-method params land on the right model objects.
        assert suite.method_objs["DT"].model_params.get("class_weight") == "balanced"
        assert suite.method_objs["RF"].model_params.get("class_weight") == "balanced"

        # XGB didn't inherit DT/RF's class_weight, and its own defaults are intact.
        assert "class_weight" not in suite.method_objs["XGB"].model_params
        assert "n_jobs" in suite.method_objs["XGB"].model_params

    def test_long_methods(self, training_config_001_modelsuite):
        """Long-form method names (DecisionTree, XGBoost, RandomForest) also work.

        Same behaviour as ``test_init_custom_methods_with_params`` but using
        the full class names rather than short aliases.
        """
        training_config_001_modelsuite.data["step_param_set"]["steps"]["model"] = {
            "methods": ["DecisionTree", "XGBoost", "RandomForest"],
            "model_params": {
                "DecisionTree": {"class_weight": "balanced"},
                "RandomForest": {"class_weight": "balanced"},
            },
        }
        suite = ModelSuite(training_config_001_modelsuite)

        assert len(suite.method_objs) == 3
        for name in ("DecisionTree", "XGBoost", "RandomForest"):
            assert name in suite.method_objs

        assert (
            suite.method_objs["DecisionTree"].model_params.get("class_weight")
            == "balanced"
        )
        assert (
            suite.method_objs["RandomForest"].model_params.get("class_weight")
            == "balanced"
        )

        assert "class_weight" not in suite.method_objs["XGBoost"].model_params
        assert "n_jobs" in suite.method_objs["XGBoost"].model_params

    def test_override_existing_defaults(self, training_config_001_modelsuite):
        """Config-level ``model_params`` override the wrapper's built-in defaults."""
        training_config_001_modelsuite.data["step_param_set"]["steps"]["model"] = {
            "methods": ["XGB", "KNN"],
            "model_params": {
                "XGB": {"n_estimators": 500, "max_depth": 3},
                "KNN": {"n_neighbors": 15},
            },
        }
        suite = ModelSuite(training_config_001_modelsuite)

        xgb = suite.method_objs["XGB"]
        assert xgb.model_params["n_estimators"] == 500
        assert xgb.model_params["max_depth"] == 3

        knn = suite.method_objs["KNN"]
        assert knn.model_params["n_neighbors"] == 15