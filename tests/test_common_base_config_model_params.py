"""Unit tests for ``ConfigBase.get_model_params``.

``model_params`` accepts shared parameters (plain names, applied to every
model), per-model sections (keyed by a model's long or short name), or both
together. These tests pin that resolution down.

The per-model form used to be unusable in practice: a model whose name was
not a key fell back to the *entire* ``model_params`` dict, so naming one model
handed the other models each other's sections as constructor arguments. That
raised ``TypeError`` from the estimator rather than anything explaining the
config, which is what ``test_named_section_is_not_given_to_other_models`` and
``test_suite_builds_with_a_single_named_section`` guard against.
"""

import pytest

from aiqclib.common.loader.model_registry import MODEL_REGISTRY
from aiqclib.common.loader.single_model_registry import SINGLE_MODEL_REGISTRY
from aiqclib.train.models.model_suite import ModelSuite
from aiqclib.train.models.random_forest import RandomForest
from aiqclib.train.models.xgboost import XGBoost


def _set_model_params(config, params):
    """Put ``params`` in the config's model step and return the config."""
    config.data["step_param_set"]["steps"]["model"]["model_params"] = params
    return config


class TestGetModelParams:
    """Resolution of shared vs per-model ``model_params``."""

    def test_absent_model_params(self, training_config_001):
        """A model step with no ``model_params`` resolves to nothing."""
        assert training_config_001.get_model_params("XGBoost", "XGB") == {}

    def test_shared_params_go_to_every_model(self, training_config_001):
        """Plain parameter names are not model names, so all models get them."""
        _set_model_params(training_config_001, {"n_estimators": 50})

        assert training_config_001.get_model_params("XGBoost", "XGB") == {
            "n_estimators": 50
        }
        assert training_config_001.get_model_params("RandomForest", "RF") == {
            "n_estimators": 50
        }

    @pytest.mark.parametrize("key", ["XGBoost", "XGB"])
    def test_named_section_is_found_by_long_or_short_name(
        self, key, training_config_001
    ):
        """A section may be keyed by either name the wrapper declares."""
        _set_model_params(training_config_001, {key: {"device": "cuda"}})

        assert training_config_001.get_model_params("XGBoost", "XGB") == {
            "device": "cuda"
        }

    def test_named_section_is_not_given_to_other_models(self, training_config_001):
        """An unnamed model gets nothing — not the other models' sections.

        This is the regression: the fallback returned the whole dict, so
        RandomForest received ``{"XGBoost": {...}}`` and its constructor
        rejected the model name as an unknown keyword argument.
        """
        _set_model_params(training_config_001, {"XGBoost": {"device": "cuda"}})

        assert training_config_001.get_model_params("RandomForest", "RF") == {}

    def test_shared_and_named_params_merge(self, training_config_001):
        """Shared parameters apply to all; the named section adds to them."""
        _set_model_params(
            training_config_001,
            {"n_estimators": 50, "XGBoost": {"device": "cuda"}},
        )

        assert training_config_001.get_model_params("XGBoost", "XGB") == {
            "n_estimators": 50,
            "device": "cuda",
        }
        assert training_config_001.get_model_params("RandomForest", "RF") == {
            "n_estimators": 50
        }

    def test_named_section_overrides_a_shared_param(self, training_config_001):
        """The more specific value wins for the model that names it."""
        _set_model_params(
            training_config_001,
            {"max_depth": 3, "XGBoost": {"max_depth": 10}},
        )

        assert training_config_001.get_model_params("XGBoost", "XGB")["max_depth"] == 10
        assert (
            training_config_001.get_model_params("RandomForest", "RF")["max_depth"] == 3
        )

    def test_long_name_wins_over_short_name(self, training_config_001):
        """With both spellings present the long form is the one applied."""
        _set_model_params(
            training_config_001,
            {"XGBoost": {"max_depth": 10}, "XGB": {"max_depth": 3}},
        )

        assert training_config_001.get_model_params("XGBoost", "XGB") == {
            "max_depth": 10
        }

    def test_non_mapping_section_is_rejected(self, training_config_001):
        """A scalar under a model name names the model in the error.

        Otherwise the failure surfaces as an unpacking TypeError with nothing
        pointing at the configuration.
        """
        _set_model_params(training_config_001, {"XGBoost": 5})

        with pytest.raises(ValueError, match="XGBoost"):
            training_config_001.get_model_params("XGBoost", "XGB")


class TestModelParamsReachTheWrapper:
    """The resolved parameters are what the wrapper actually constructs with."""

    def test_named_section_reaches_the_named_wrapper(self, training_config_001):
        """XGBoost picks up its own section."""
        training_config_001.data["step_class_set"]["steps"]["model"] = "XGBoost"
        _set_model_params(training_config_001, {"XGBoost": {"device": "cuda"}})

        assert XGBoost(training_config_001).model_params["device"] == "cuda"

    def test_other_wrappers_keep_their_defaults(self, training_config_001):
        """Naming one model leaves another's defaults intact and buildable."""
        training_config_001.data["step_class_set"]["steps"]["model"] = "RandomForest"
        _set_model_params(training_config_001, {"XGBoost": {"device": "cuda"}})

        params = RandomForest(training_config_001).model_params
        assert "device" not in params
        assert "XGBoost" not in params
        # Constructing the estimator is the check that matters: unknown keys
        # raise TypeError here, which is how the original bug presented.
        RandomForest(training_config_001)._get_model_class()(**params)

    def test_suite_builds_with_a_single_named_section(self, training_config_001):
        """Every model in the suite constructs when only one is named.

        The suite instantiates all nine wrappers in its constructor, so this
        fails outright if any of them is handed another's section.
        """
        training_config_001.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        _set_model_params(training_config_001, {"XGBoost": {"max_depth": 4}})

        suite = ModelSuite(training_config_001)

        assert len(suite.method_objs) == len(suite.default_methods)
        for method, obj in suite.method_objs.items():
            obj._get_model_class()(**obj.model_params)
            assert "XGBoost" not in obj.model_params, method
        assert suite.method_objs["XGB"].model_params["max_depth"] == 4


class TestRegistrySeparation:
    """The single-model registry stays free of the suite.

    ``MODEL_REGISTRY`` used to be an alias of ``SINGLE_MODEL_REGISTRY`` rather
    than a copy, so importing it mutated the single-model registry. That let a
    suite list itself among its own methods, and made the set of known model
    names — which ``get_model_params`` uses to tell a model section from a
    shared parameter — depend on which modules had been imported.
    """

    def test_suite_is_not_a_single_model(self):
        """Importing MODEL_REGISTRY does not add ModelSuite to the single one."""
        assert "ModelSuite" not in SINGLE_MODEL_REGISTRY
        assert "MS" not in SINGLE_MODEL_REGISTRY

    def test_suite_is_in_the_full_registry(self):
        """The full registry still resolves the suite under both names."""
        assert MODEL_REGISTRY["ModelSuite"] is ModelSuite
        assert MODEL_REGISTRY["MS"] is ModelSuite

    def test_full_registry_still_has_the_single_models(self):
        """Copying did not drop anything."""
        for name in SINGLE_MODEL_REGISTRY:
            assert MODEL_REGISTRY[name] is SINGLE_MODEL_REGISTRY[name]
