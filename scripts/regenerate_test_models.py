"""
Regenerate the joblib model fixtures under tests/data/training/.

Uses the aiqclib training pipeline against the existing test_set/train_set
parquet fixtures to rebuild every algorithm's model artefacts against the
currently-installed sklearn/XGBoost versions. Run this whenever scikit-learn
or XGBoost is bumped past a major.minor boundary and the classify tests
start emitting InconsistentVersionWarning.

After running, re-zip tests/data/ and publish a new test-data release
(see scripts/fetch_test_data.sh and .github/workflows/check_package.yml
for the matching consumer side).

Usage:
    uv run python scripts/regenerate_test_models.py
"""

from pathlib import Path

from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.common.loader.training_loader import (
    load_step1_input_training_set,
    load_step4_build_model_class,
)


# Map of algorithm short suffix -> the YAML model class name.
# Suffix "" means the default file (model_temp.joblib without algo suffix).
# Note: "xgb" and "" both produce XGBoost models; the codebase uses both
# naming conventions in different tests.
ALGOS = {
    "":      "XGBoost",
    "xgb":   "XGBoost",
    "logit": "LogisticRegression",
    "lda":   "LinearDiscriminantAnalysis",
    "svm":   "SVM",
    "dt":    "DecisionTree",
    "rf":    "RandomForest",
    "knn":   "KNearestNeighbors",
    "gnb":   "GaussianNaiveBayes",
    "mlp":   "MLP",
}

TARGETS = ("temp", "psal", "pres")
REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = REPO_ROOT / "tests" / "data"
CONFIG_FILE = TEST_DATA / "config" / "test_training_001.yaml"
TRAINING_DIR = TEST_DATA / "training"

NEGX5_CONFIG = TEST_DATA / "config" / "test_training_003.yaml"
NEGX5_INPUT  = TEST_DATA / "negx5_training"
NEGX5_OUTPUT = TEST_DATA / "negx5_model"


def regenerate_one(algo_name: str, suffix: str) -> None:
    """Train one algorithm against the test fixtures and write 3 joblibs.

    Uses the library's own ``save_model()`` method (which serializes the
    underlying sklearn/xgboost estimator from ``model.model``, not the
    aiqclib wrapper itself). This matches the format the load path expects:
    ``model_base.load_model()`` calls ``isinstance(loaded, sklearn_cls)``
    after unpickling, so the file must contain the bare estimator.
    """
    config = TrainingConfig(str(CONFIG_FILE))
    config.select("NRT_BO_001")
    config.data["step_class_set"]["steps"]["model"] = algo_name

    # Point the input loader at the existing parquet fixtures.
    input_file_names = {
        kind: {tgt: str(TRAINING_DIR / f"{kind}_set_{tgt}.parquet") for tgt in TARGETS}
        for kind in ("train", "test")
    }

    ds_input = load_step1_input_training_set(config)
    ds_input.input_file_names = input_file_names
    ds_input.process_targets()

    ds_build = load_step4_build_model_class(
        config, ds_input.training_sets, ds_input.test_sets
    )
    ds_build.build_final_model_targets()

    # Write each target's model using the library's public save_model() API.
    # This is the same call path the production training pipeline uses
    # via build_model_base.write_models() -> model_ref.save_model(path).
    suffix_part = f"_{suffix}" if suffix else ""
    for tgt, model_wrapper in ds_build.final_models.items():
        out = TRAINING_DIR / f"model_{tgt}{suffix_part}.joblib"
        model_wrapper.save_model(str(out))
        print(f"  wrote {out.relative_to(REPO_ROOT)}")


def regenerate_negx5() -> None:
    """Regenerate the negx5 XGBoost models used by TestClassifyDataSetNegX5."""
    config = TrainingConfig(str(NEGX5_CONFIG))
    config.select("NRT_BO_001")
    # negx5 only uses XGBoost — no algorithm fan-out here
    config.data["step_class_set"]["steps"]["model"] = "XGBoost"

    input_file_names = {
        kind: {tgt: str(NEGX5_INPUT / f"{kind}_set_{tgt}.parquet") for tgt in TARGETS}
        for kind in ("train", "test")
    }

    ds_input = load_step1_input_training_set(config)
    ds_input.input_file_names = input_file_names
    ds_input.process_targets()

    ds_build = load_step4_build_model_class(
        config, ds_input.training_sets, ds_input.test_sets
    )
    ds_build.build_final_model_targets()

    NEGX5_OUTPUT.mkdir(parents=True, exist_ok=True)
    for tgt, model_wrapper in ds_build.final_models.items():
        out = NEGX5_OUTPUT / f"model_{tgt}.joblib"
        model_wrapper.save_model(str(out))
        print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    if not TRAINING_DIR.exists():
        raise SystemExit(
            f"{TRAINING_DIR} not found. Run scripts/fetch_test_data.sh first."
        )

    for suffix, algo_name in ALGOS.items():
        print(f"Regenerating {algo_name} (suffix='{suffix}') ...")
        regenerate_one(algo_name, suffix)

    print("Regenerating negx5 XGBoost models ...")
    regenerate_negx5()

    print("\nDone. Next steps:")
    print("  1. Run the full test suite to confirm nothing broke:")
    print("       uv run pytest -q")
    print("  2. Rebuild the archive:")
    print("       cd tests/data && zip -r ../../test-data.zip . && cd ../..")
    print("  3. Publish a new release:")
    print("       gh release create test-data-v1.0.2 test-data.zip \\")
    print("         --title 'Test data v1.0.2' \\")
    print("         --notes 'Restructured all unit tests'")
    print("  4. Bump TEST_DATA_VERSION in both workflow files and in")
    print("     scripts/fetch_test_data.sh.")
    print("  5. Delete the local test-data.zip after upload succeeds.")


if __name__ == "__main__":
    main()