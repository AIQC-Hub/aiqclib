"""Unit tests for the concrete model wrapper classes.

This is the canonical "all rich fields used" example of the MODEL_CASES
pattern. Each test method runs against all 9 model wrappers via parametrize:

- ``test_init_class`` consumes ``case.wrapper_cls`` and ``case.sklearn_cls``
- ``test_multi_flag`` consumes ``case.wrapper_cls`` (and ``case.config_name`` for wiring)
- ``test_default_params`` consumes ``case.defaults`` and ``case.missing``
- ``test_config_params_override`` consumes ``case.override``

Refactored from the original which had:
- Nine ``unittest.TestCase`` classes (``TestXGBoost``, ``TestLogisticRegression``,
  ..., ``TestMLP``), each with an identical ``setUp`` and four near-identical
  test methods differing only in the model under test (collapsed into a
  single parametrized class here)

Test count is unchanged: 9 models × 4 methods = 36 tests in both the original
and the refactored file.
"""

import pytest

from tests._model_cases import MODEL_CASES


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda c: c.config_name)
class TestModelWrappers:
    """Per-wrapper checks: init, multi flag, default params, config override.

    Every test sets ``step_class_set.steps.model = case.config_name`` on the
    fresh function-scoped ``training_config_001`` fixture before constructing
    the wrapper. The wrapper itself is constructed via ``case.wrapper_cls(config)``
    so the test doesn't need a separate import per model.
    """

    def test_init_class(self, case, training_config_001):
        """``expected_class_name`` matches the wrapper class name; sklearn class resolves."""
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        ds = case.wrapper_cls(training_config_001)
        # ``wrapper_cls.__name__`` matches each wrapper's ``expected_class_name``,
        # including the non-obvious cases (SVM → "SupportVectorMachine",
        # MLP → "MultilayerPerceptron").
        assert ds.expected_class_name == case.wrapper_cls.__name__
        assert ds._get_model_class() == case.sklearn_cls

    def test_multi_flag(self, case, training_config_001):
        """Single-model wrappers expose ``multi == False``."""
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        ds = case.wrapper_cls(training_config_001)
        assert ds.multi is False

    def test_default_params(self, case, training_config_001):
        """Default ``model_params`` include ``case.defaults`` and exclude ``case.missing``."""
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        ds = case.wrapper_cls(training_config_001)

        # Every expected default key has the expected value.
        for key, expected_value in case.defaults.items():
            assert ds.model_params.get(key) == expected_value, (
                f"{case.config_name}: expected model_params[{key!r}] == "
                f"{expected_value!r}, got {ds.model_params.get(key)!r}"
            )

        # Keys explicitly recorded as "must not be present" are absent.
        # Examples: LDA doesn't support n_jobs; LogisticRegression no longer
        # uses penalty (after the sklearn 1.8 deprecation, replaced by l1_ratio).
        for key in case.missing:
            assert key not in ds.model_params, (
                f"{case.config_name}: model_params[{key!r}] should not be set, "
                f"got value {ds.model_params.get(key)!r}"
            )

    def test_config_params_override(self, case, training_config_001):
        """``model_params`` set in the YAML config flow into the wrapper's params."""
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        training_config_001.data["step_param_set"]["steps"]["model"]["model_params"] = (
            dict(case.override)
        )
        ds = case.wrapper_cls(training_config_001)

        for key, expected_value in case.override.items():
            assert ds.model_params[key] == expected_value, (
                f"{case.config_name}: override of {key!r} to {expected_value!r} "
                f"did not propagate; got {ds.model_params[key]!r}"
            )
