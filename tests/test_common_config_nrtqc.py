"""Unit tests for the ``NRTQCConfig`` class and NRT QC config schema.

Coverage:
- ``validate()`` accepts the bundled template and rejects malformed YAML
- ``select()`` resolves every referenced sub-configuration section and
  aliases ``qc_variable_set`` to ``target_set`` for the ConfigBase helpers
- ``get_qc_items()`` merges built-in defaults with config overrides and
  resolves ``fail_flag``
- ``get_variable_flag()`` returns the configured flag column or None
- Schema strictness: variables require only ``name``; unknown item keys and
  invalid ``fail_flag`` values are rejected
- Template YAMLs resolve folder paths through the step-path machinery
"""

import pytest
import yaml
from jsonschema import validate as jsonschema_validate
from jsonschema.exceptions import ValidationError

from aiqclib.common.config.nrtqc_config import NRTQCConfig
from aiqclib.common.config.yaml_schema import get_nrtqc_config_schema

TEMPLATE = "template:nrt_qc_sets"

#: Item names enabled in the bundled template, in order.
TEMPLATE_ITEM_NAMES = [
    "impossible_date",
    "impossible_location",
    "global_range",
    "regional_range",
    "pressure_increasing",
    "spike",
    "gradient",
    "digit_rollover",
    "stuck_value",
    "density_inversion",
    "temp_to_psal",
]


@pytest.fixture
def template_config() -> NRTQCConfig:
    """An NRTQCConfig loaded and selected from the bundled template."""
    ds = NRTQCConfig(TEMPLATE)
    ds.select("nrt_qc_0001")
    return ds


# ---------------------------------------------------------------------------
# Validation and selection
# ---------------------------------------------------------------------------


class TestNRTQCConfig:
    """Tests for ``NRTQCConfig`` loading and selection."""

    def test_valid_template(self):
        """The bundled template validates as 'valid'."""
        ds = NRTQCConfig(TEMPLATE)
        assert "valid" in ds.validate()

    def test_invalid_config(self, config_dir):
        """A YAML for another module (missing NRT QC sections) is invalid."""
        ds = NRTQCConfig(str(config_dir / "test_dataset_invalid.yaml"))
        assert "invalid" in ds.validate()

    def test_select_resolves_sections(self, template_config):
        """select() resolves each referenced sub-configuration in data."""
        data = template_config.data
        assert data["path_info"]["name"] == "nrt_qc_path_1"
        assert data["qc_variable_set"]["name"] == "qc_variable_set_1"
        assert data["qc_item_set"]["name"] == "qc_item_set_1"
        assert data["step_class_set"]["name"] == "nrt_qc_step_set_1"
        assert data["step_param_set"]["name"] == "nrt_qc_param_set_1"

    def test_target_set_alias(self, template_config):
        """qc_variable_set is aliased to target_set for ConfigBase helpers."""
        assert (
            template_config.data["target_set"]
            is (template_config.data["qc_variable_set"])
        )
        assert template_config.get_target_names() == ["temp", "psal"]

    def test_select_twice(self, template_config):
        """Calling select() twice with the same name is idempotent."""
        template_config.select("nrt_qc_0001")  # Should not raise

    def test_invalid_dataset_name(self):
        """select() with an unknown set name raises ValueError."""
        ds = NRTQCConfig(TEMPLATE)
        with pytest.raises(ValueError):
            ds.select("INVALID_NAME")

    def test_auto_select(self):
        """auto_select=False defers loading; auto_select=True loads."""
        ds = NRTQCConfig(TEMPLATE, False)
        assert ds.data is None

        ds = NRTQCConfig(TEMPLATE, True)
        assert ds.data is not None

    def test_step_classes(self, template_config):
        """The step class names resolve via get_base_class()."""
        assert template_config.get_base_class("input") == "InputDataSetAll"
        assert template_config.get_base_class("qc") == "QCDataSetAll"
        assert template_config.get_base_class("concat") == "ConcatDataSetAll"
        assert template_config.get_base_class("compare") == "CompareFlagsAll"


# ---------------------------------------------------------------------------
# QC item resolution
# ---------------------------------------------------------------------------


class TestGetQCItems:
    """Tests for ``get_qc_items`` param merging and ``fail_flag``."""

    def test_items_in_config_order(self, template_config):
        """All template items are returned in configuration order."""
        items = template_config.get_qc_items()
        assert [x["name"] for x in items] == TEMPLATE_ITEM_NAMES
        assert template_config.get_qc_item_names() == TEMPLATE_ITEM_NAMES

    def test_config_params_kept(self, template_config):
        """Params from the config are passed through untouched."""
        by_name = {x["name"]: x for x in template_config.get_qc_items()}
        assert by_name["global_range"]["params"]["temp"] == {
            "min": -2.5,
            "max": 40.0,
        }
        assert by_name["density_inversion"]["params"] == {"threshold": 0.03}
        assert by_name["stuck_value"]["params"] == {}

    def test_default_params_merged(self, template_config):
        """Defaults fill missing params; config values win per key."""
        defaults = {
            "density_inversion": {"threshold": 0.99, "extra": True},
            "stuck_value": {"min_observations": 2},
        }
        by_name = {x["name"]: x for x in template_config.get_qc_items(defaults)}
        # Config threshold overrides the default; extra default key kept.
        assert by_name["density_inversion"]["params"] == {
            "threshold": 0.03,
            "extra": True,
        }
        # Item with no config params gets the defaults verbatim.
        assert by_name["stuck_value"]["params"] == {"min_observations": 2}

    def test_fail_flag_default_and_override(self, template_config):
        """fail_flag defaults to 4 and honours a per-item override."""
        items = template_config.data["qc_item_set"]["items"]
        by_name = {x["name"]: x for x in items}
        by_name["regional_range"]["fail_flag"] = 3

        resolved = {x["name"]: x for x in template_config.get_qc_items()}
        assert resolved["regional_range"]["fail_flag"] == 3
        assert resolved["global_range"]["fail_flag"] == 4


# ---------------------------------------------------------------------------
# Existing-flag resolution (for the comparison step)
# ---------------------------------------------------------------------------


class TestGetVariableFlag:
    """Tests for ``get_variable_flag`` optional-flag handling."""

    def test_configured_flags(self, template_config):
        """Template variables carry their existing flag columns."""
        assert template_config.get_variable_flag("temp") == "temp_qc"
        assert template_config.get_variable_flag("psal") == "psal_qc"

    @pytest.mark.parametrize("flag_value", [None, "", "   "])
    def test_missing_flag_values(self, template_config, flag_value):
        """None/empty/whitespace flags resolve to None."""
        variables = template_config.data["qc_variable_set"]["variables"]
        by_name = {v["name"]: v for v in variables}
        by_name["temp"]["flag"] = flag_value
        assert template_config.get_variable_flag("temp") is None

    def test_absent_flag_key(self, template_config):
        """A variable without a flag key resolves to None."""
        variables = template_config.data["qc_variable_set"]["variables"]
        by_name = {v["name"]: v for v in variables}
        by_name["psal"].pop("flag")
        assert template_config.get_variable_flag("psal") is None

    def test_unknown_variable(self, template_config):
        """An unknown variable name resolves to None."""
        assert template_config.get_variable_flag("chlorophyll") is None


# ---------------------------------------------------------------------------
# Schema strictness
# ---------------------------------------------------------------------------


class TestNRTQCSchema:
    """Direct schema checks for optional/forbidden keys."""

    @pytest.fixture
    def schema(self) -> dict:
        """The NRT QC schema as a dictionary."""
        return yaml.safe_load(get_nrtqc_config_schema())

    @pytest.fixture
    def template_dict(self) -> dict:
        """The bundled template as a mutable dictionary."""
        return NRTQCConfig(TEMPLATE).full_config

    def test_variables_require_only_name(self, schema):
        """QC variables require only ``name``; flag is nullable."""
        variables = schema["properties"]["qc_variable_sets"]["items"]["properties"][
            "variables"
        ]
        assert variables["items"]["required"] == ["name"]
        assert "null" in variables["items"]["properties"]["flag"]["type"]

    def test_flagless_variable_validates(self, schema, template_dict):
        """A variable with only a name passes validation."""
        template_dict["qc_variable_sets"][0]["variables"] = [{"name": "temp"}]
        jsonschema_validate(instance=template_dict, schema=schema)

    def test_unknown_item_key_rejected(self, schema, template_dict):
        """Unknown keys on a QC item fail validation."""
        template_dict["qc_item_sets"][0]["items"][0]["typo_key"] = 1
        with pytest.raises(ValidationError):
            jsonschema_validate(instance=template_dict, schema=schema)

    def test_invalid_fail_flag_rejected(self, schema, template_dict):
        """fail_flag outside {3, 4} fails validation."""
        template_dict["qc_item_sets"][0]["items"][0]["fail_flag"] = 2
        with pytest.raises(ValidationError):
            jsonschema_validate(instance=template_dict, schema=schema)

    def test_missing_section_rejected(self, schema, template_dict):
        """Dropping a required top-level section fails validation."""
        del template_dict["qc_item_sets"]
        with pytest.raises(ValidationError):
            jsonschema_validate(instance=template_dict, schema=schema)


# ---------------------------------------------------------------------------
# Path resolution through the template
# ---------------------------------------------------------------------------


class TestNRTQCTemplatePaths:
    """Path resolution for the bundled NRT QC template."""

    def test_input_folder(self, template_config):
        """Input file path resolves without dataset/step folders."""
        input_file_name = template_config.get_full_file_name(
            "input",
            template_config.data["input_file_name"],
            use_dataset_folder=False,
            folder_name_auto=False,
        )
        assert input_file_name == "/path/to/input/nrt_cora_bo_4.parquet"

    def test_concat_folder(self, template_config):
        """Concat output resolves under the dataset and nrt_qc folders."""
        assert (
            template_config.get_full_file_name("concat", "test.parquet")
            == "/path/to/data/nrt_qc_0001/nrt_qc/test.parquet"
        )

    def test_qc_folder_auto(self, template_config):
        """Steps without path_info entries fall back to their own name."""
        assert (
            template_config.get_full_file_name("qc", "test.parquet")
            == "/path/to/data/nrt_qc_0001/qc/test.parquet"
        )

    def test_target_file_names(self, template_config):
        """Per-variable file names expand the {target_name} placeholder."""
        names = template_config.get_target_file_names(
            "compare", "nrt_qc_flag_comparison_{target_name}.tsv"
        )
        assert names == {
            "temp": (
                "/path/to/data/nrt_qc_0001/compare/nrt_qc_flag_comparison_temp.tsv"
            ),
            "psal": (
                "/path/to/data/nrt_qc_0001/compare/nrt_qc_flag_comparison_psal.tsv"
            ),
        }
