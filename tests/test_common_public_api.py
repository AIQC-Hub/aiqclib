"""Contract tests for the ``aiqclib`` API that ``nrtqclib`` builds on.

``nrtqclib`` (the NRT QC pipeline, split out of ``aiqclib``) imports the
names below. Renaming or removing one breaks that package, so the break is
caught here first: when ``nrtqclib`` starts importing a new ``aiqclib`` name,
add it to :data:`CONTRACT`. See ``docs/design/NRTQCLIB_plan.md``.

The tests also cover the extension points ``nrtqclib`` relies on (the
:class:`ConfigBase` schema and template hooks, and
:func:`aiqclib.interface.batch.run_phases`) and check that importing the
contract modules does not load the training stack.
"""

import importlib
import subprocess
import sys

import polars as pl
import pytest

import aiqclib
from aiqclib.common.base.config_base import ConfigBase
from aiqclib.interface.batch import Phase, available_modes, run_phases

#: ``(module, name)`` pairs ``nrtqclib`` imports from ``aiqclib``.
CONTRACT = [
    ("aiqclib.common.base.config_base", "ConfigBase"),
    ("aiqclib.common.base.dataset_base", "DataSetBase"),
    ("aiqclib.prepare.step1_read_input.input_base", "InputDataSetBase"),
    ("aiqclib.common.loader.feature_registry", "FEATURE_REGISTRY"),
    ("aiqclib.prepare.features.qc_item_base", "QCItemFeatureBase"),
    ("aiqclib.prepare.features.qc_item_base", "OBSERVATION_KEYS"),
    ("aiqclib.common.utils.qc_flags", "FLAG_GOOD"),
    ("aiqclib.common.utils.qc_flags", "FLAG_BAD"),
    ("aiqclib.common.utils.qc_flags", "worst_flag"),
    ("aiqclib.common.utils.qc_flags", "flag_as_int"),
    ("aiqclib.common.utils.qc_flags", "normalize_flag_values"),
    ("aiqclib.common.utils.config", "get_config_item"),
    ("aiqclib.common.utils.config", "get_config_file"),
    ("aiqclib.common.utils.config", "read_config"),
    ("aiqclib.common.utils.file", "ensure_output_file"),
    ("aiqclib.common.utils.input_validation", "REQUIRED_INPUT_COLUMNS"),
    ("aiqclib.common.utils.progress", "report_progress"),
    ("aiqclib.interface.batch", "Phase"),
    ("aiqclib.interface.batch", "run_phases"),
    ("aiqclib.interface.batch", "SUMMARY_SCHEMA"),
]

#: Modules that must stay importable without the training stack.
HEAVY_MODULES = ("xgboost", "sklearn", "matplotlib", "shap")


@pytest.mark.parametrize(("module_name", "name"), CONTRACT)
def test_contract_name_is_importable(module_name, name):
    module = importlib.import_module(module_name)
    assert hasattr(module, name), f"{module_name}.{name} is part of the contract"


def test_contract_modules_do_not_load_the_training_stack():
    """``nrtqclib`` should not pay for xgboost or matplotlib just to load."""
    modules = sorted({module_name for module_name, _ in CONTRACT})
    code = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        f"print(','.join(m for m in {HEAVY_MODULES!r} if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


class TestLazyPackage:
    def test_every_public_name_resolves(self):
        for name in aiqclib.__all__:
            assert callable(getattr(aiqclib, name))

    def test_public_names_are_listed(self):
        assert set(aiqclib.__all__) <= set(dir(aiqclib))

    def test_unknown_name_raises(self):
        with pytest.raises(AttributeError, match="no_such_function"):
            aiqclib.no_such_function  # noqa: B018


SCHEMA_TEXT = """\
type: object
properties:
  my_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        path_info:
          type: string
      required: [name, path_info]
required: [my_sets]
"""

TEMPLATE_TEXT = """\
path_info_sets:
  - name: path_1
    common:
      base_path: /path/to/data
my_sets:
  - name: my_0001
    path_info: path_1
"""


class MyConfig(ConfigBase):
    """A configuration class for a section ``aiqclib`` does not know."""

    expected_class_name = "MyConfig"

    def __init__(self, config_file: str, auto_select: bool = False) -> None:
        super().__init__("my_sets", config_file, auto_select)

    @classmethod
    def get_yaml_schema(cls, section_name: str) -> str:
        if section_name == "my_sets":
            return SCHEMA_TEXT
        return super().get_yaml_schema(section_name)

    @classmethod
    def get_template_text(cls, template_name: str) -> str:
        if template_name == "template:my_sets":
            return TEMPLATE_TEXT
        return super().get_template_text(template_name)


class TestConfigHooks:
    def test_subclass_serves_its_own_section_and_template(self):
        config = MyConfig("template:my_sets", auto_select=True)
        assert config.dataset_name == "my_0001"
        assert config.data["path_info"]["name"] == "path_1"

    def test_subclass_reads_its_section_from_a_file(self, tmp_path):
        path = tmp_path / "my.yaml"
        path.write_text(TEMPLATE_TEXT, encoding="utf-8")
        config = MyConfig(str(path))
        config.select("my_0001")
        assert config.dataset_name == "my_0001"

    def test_subclass_still_reaches_the_built_in_registries(self):
        assert "data_sets:" in MyConfig.get_template_text("template:data_sets")
        assert "properties" in MyConfig.get_yaml_schema("data_sets")

    def test_unknown_section_is_rejected(self):
        with pytest.raises(ValueError, match="no_sets is not supported"):
            ConfigBase.get_yaml_schema("no_sets")

    def test_unknown_template_is_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            ConfigBase.get_template_text("template:no_sets")


class TestRunPhases:
    @pytest.fixture
    def my_phase(self):
        log = []

        def read_my_config(file_name, set_name=None):
            return {"file": file_name, "set": set_name}

        def run_my_qc(config, verbose=False):
            log.append(config)

        phase = Phase(
            "my_qc",
            "my_qc_set_name",
            "my_qc_config",
            run_my_qc,
            in_all=False,
            config_reader=read_my_config,
        )
        return phase, log

    def test_external_phase_runs_with_its_own_reader(self, my_phase):
        phase, log = my_phase
        table = pl.DataFrame({"name": ["a", "b"], "my_qc_set_name": ["set_a", ""]})
        summary = run_phases(
            (phase,),
            mode="my_qc",
            config_files={"my_qc_config": "my.yaml"},
            table=table,
        )
        assert log == [{"file": "my.yaml", "set": "set_a"}]
        assert summary["status"].to_list() == ["ok", "skipped"]
        assert summary["phase"].to_list() == ["my_qc", "my_qc"]

    def test_external_phase_without_table_auto_selects(self, my_phase):
        phase, log = my_phase
        summary = run_phases(
            (phase,), mode="my_qc", config_files={"my_qc_config": "my.yaml"}
        )
        assert log == [{"file": "my.yaml", "set": None}]
        assert summary["status"].to_list() == ["ok"]

    def test_unknown_mode_lists_the_given_phases(self, my_phase):
        phase, _ = my_phase
        with pytest.raises(ValueError, match="Expected one of: my_qc, all"):
            run_phases((phase,), mode="prepare", config_files={})

    def test_available_modes_for_given_phases(self, my_phase):
        phase, _ = my_phase
        assert available_modes((phase,)) == ["my_qc", "all"]
