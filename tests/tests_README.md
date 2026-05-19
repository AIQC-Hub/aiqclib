# Test suite conventions

This document is the short version. The Phase 1 example files
(`test_common_utils_config.py`, `test_prepare_step1_input_a.py`,
`test_training_step2_validate_a.py`) are the long version — read them when
in doubt about a pattern.

## Quick start

```bash
bash scripts/fetch_test_data.sh   # one-time: download fixtures
uv run pytest -v                  # run the suite
uv run pytest tests/test_foo.py   # run one file
uv run pytest -k "logit"          # run tests whose id contains "logit"
```

## Style: pytest, not unittest

Use `def test_xxx(self, fixture_a, fixture_b)` inside plain (non-`TestCase`)
classes. Use `assert` rather than `self.assertEqual`. Group related tests
into a class only when they share a meaningful pattern (e.g. all exercising
the same step's output); otherwise let them be module-level functions.

## Fixtures live in `conftest.py`

Don't redefine path-or-config setup that already exists there. The common
ones:

| Fixture | What it gives you |
| --- | --- |
| `data_dir`, `config_dir`, `input_dir`, `training_dir` | `Path` constants |
| `test_output_dir` | `Path` for test-generated output files |
| `test_data_file` | `Path` to `nrt_cora_bo_test.parquet` |
| `dataset_config_001`..`_005` | Loaded + selected `DataSetConfig` |
| `training_config_001`..`_003` | Loaded + selected `TrainingConfig` |
| `classify_config_001`..`_003` | Loaded + selected `ClassificationConfig` |
| `dataset_yaml_001` etc. | Raw `Path` to a YAML file (use when testing config-loading itself) |
| `training_input_001` | A `ds_input` wired against `tests/data/training/` |

All config fixtures are function-scoped, so mutating
`training_config_001.data[...]` in one test does not leak to the next.

## Per-model parametrize: import from `_model_cases`

Files that previously had `TestXGBoost`, `TestLogisticRegression`, ..., one
per model, now look like:

```python
from tests._model_cases import MODEL_CASES

@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda c: c.config_name)
class TestModels:
    def test_base_model(self, case, training_config_001):
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        ds = KFoldValidation(training_config_001)
        assert isinstance(ds.base_model, case.wrapper_cls)
```

Pytest expands this into 9 cases, with ids like `[XGBoost]`,
`[LogisticRegression]`, etc.

## Per-target loops

Three targets — `temp`, `psal`, `pres` — almost always appear together.
Loop over `TARGETS` (imported from conftest) rather than writing the same
block three times:

```python
from tests.conftest import TARGETS

def test_something(self, ds):
    for tgt in TARGETS:
        assert isinstance(ds.training_sets[tgt], pl.DataFrame)
        assert ds.training_sets[tgt].shape[0] == EXPECTED_ROWS  # TODO
```

### Targets: TARGETS vs TARGETS_NONEMPTY
Two target-list constants live in conftest.py. Use the one that matches what your test actually exercises.

 - `TARGETS = ("temp", "psal", "pres")`: Canonical 3-target list.
 - `TARGETS_NONEMPTY = ("temp", "psal")`: 2-target subset. Use when the per-target dict does not have the `pres` key.

## Output files: write, assert, manually remove

Use `test_output_dir` for the destination. Clean up with `os.remove`
explicitly so you can comment out the cleanup to inspect outputs when a
test fails:

```python
def test_write_reports(self, ..., test_output_dir):
    output_paths = {
        tgt: str(test_output_dir / f"test_validation_report_{tgt}.tsv")
        for tgt in TARGETS
    }
    ds.output_file_names["report"] = output_paths
    ds.process_targets()
    ds.write_reports()
    for tgt, path in output_paths.items():
        assert os.path.exists(path)
        os.remove(path)  # comment out to debug
```

## TODO markers

Many assertions reference shapes, year lists, and other data-dependent
numbers that may need updating after a fixture-data change. These are
flagged inline with `# TODO: update to actual value`. Resolve them by
running the test, observing the actual value, and editing.
