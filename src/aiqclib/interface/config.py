"""
Module providing utilities for writing YAML configuration templates, reading
them as instantiated configuration objects, and building a configuration
object directly from a built-in template. Supports "prepare", "train",
"classify", and "nrt_qc" stages using corresponding registry lookups.
"""

from typing import Dict, Optional, Tuple, Type

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.config.nrtqc_config import NRTQCConfig
from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.common.config.yaml_templates import get_template_text
from aiqclib.common.utils.config import get_config_file
from aiqclib.common.utils.config import read_config as utils_read_config
from aiqclib.common.utils.file import ensure_output_file


#: Maps a ``"<stage>_<extension>"`` key to the built-in template that serves
#: it and the configuration class that reads it. Shared by
#: :func:`write_config_template` and :func:`read_config_template` so the two
#: always offer the same stages and hand back the same YAML.
_STAGE_TEMPLATES: Dict[str, Tuple[str, Type[ConfigBase]]] = {
    "prepare_": ("template:data_sets_all", DataSetConfig),
    "prepare_full": ("template:data_sets_full", DataSetConfig),
    "prepare_reduced": ("template:data_sets", DataSetConfig),
    "train_": ("template:training_sets", TrainingConfig),
    "classify_": ("template:classification_sets", ClassificationConfig),
    "classify_full": ("template:classification_sets_full", ClassificationConfig),
    "nrt_qc_": ("template:nrt_qc_sets", NRTQCConfig),
}


def _get_stage_template(stage: str, extension: str) -> Tuple[str, Type[ConfigBase]]:
    """
    Look up the template and configuration class for a stage.

    :param stage: One of "prepare", "train", "classify" or "nrt_qc".
    :type stage: str
    :param extension: The template variant; "", "full" or "reduced".
    :type extension: str
    :return: The template identifier and the class that reads it.
    :rtype: tuple[str, type[ConfigBase]]
    :raises ValueError: If the stage and extension name no known template.
    """
    if f"{stage}_{extension}" not in _STAGE_TEMPLATES:
        raise ValueError(f"Module {stage} is not supported.")

    return _STAGE_TEMPLATES[f"{stage}_{extension}"]


def write_config_template(
    file_name: str,
    stage: str,
    extension: str = "",
    create_dirs: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Write a YAML configuration template for the specified stage
    ("prepare", "train", or "classify") to a file.

    This function:
      1. Chooses a template generator based on the combination of ``stage`` and ``extension``.
      2. Validates that the directory for ``file_name`` exists, creating it when
         ``create_dirs`` is True, and that no file would be replaced unless
         ``overwrite`` is True.
      3. Writes the generated YAML template text to the specified file.

    :param file_name: The path (including filename) where the YAML file will be
                      written. A leading ``~`` is expanded to the user's home
                      directory.
    :type file_name: str
    :param stage: Determines which template to write; must be one of "prepare",
                  "train", "classify", or "nrt_qc".
    :type stage: str
    :param extension: Determines template extensions; must be one of "", "full", or "reduced".
    :type extension: str
    :param create_dirs: If True, create the output directory (and any missing
                        parents) instead of raising when it does not exist.
                        Defaults to False, so a mistyped path is reported rather
                        than silently creating directories.
    :type create_dirs: bool
    :param overwrite: If True, replace ``file_name`` when it already exists.
                      Defaults to False, so a customized configuration is never
                      silently reset to the template.
    :type overwrite: bool
    :raises ValueError: If the combined stage and extension is not found in the registry.
    :raises IOError: If the directory of the specified file path does not exist
                     and ``create_dirs`` is False.
    :raises FileExistsError: If ``file_name`` exists and ``overwrite`` is False.

    Example Usage:
      >>> # write_config_template("~/new/dir/prepare.yaml", "prepare")
      >>> # IOError: Directory '/home/you/new/dir' does not exist. Create it
      >>> #          first, or pass create_dirs=True to create it automatically.
      >>> # write_config_template("/tmp/new/dir/prepare.yaml", "prepare", create_dirs=True)
      >>> # write_config_template("/tmp/prepare.yaml", "prepare")   # a second time
      >>> # FileExistsError: File '/tmp/prepare.yaml' already exists. Pass
      >>> #                  overwrite=True to replace it, ...
    """
    template_name, _ = _get_stage_template(stage, extension)

    yaml_text = get_template_text(template_name)
    file_name = ensure_output_file(
        file_name, create_dirs=create_dirs, overwrite=overwrite
    )

    with open(file_name, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(yaml_text)


def read_config_template(
    stage: str, extension: str = "", auto_select: bool = True
) -> ConfigBase:
    """
    Read a built-in YAML configuration template as a configuration object.

    This is the counterpart of :func:`write_config_template`: it takes the same
    ``stage`` and ``extension`` and resolves the same template, but returns the
    configuration object directly instead of writing the YAML to a file. It is
    the quickest way to see what a stage's defaults are — ``print()`` on the
    result summarizes the targets, features, steps and output directories —
    and it lets a configuration be built in code, by adjusting the returned
    object, without a file on disk.

    .. note::

       A template carries placeholder paths (``/path/to/data``) and a
       placeholder input file name, so a returned object is not ready to run a
       pipeline with. Set at least ``path_info`` and ``input_file_name`` on it
       first, or write the template out with :func:`write_config_template`,
       edit it, and load it with :func:`read_config`.

    :param stage: Determines which template to read; must be one of "prepare",
                  "train", "classify", or "nrt_qc".
    :type stage: str
    :param extension: Determines template extensions; must be one of "",
                      "full", or "reduced".
    :type extension: str
    :param auto_select: If True, select the template's single entry, so the
                        returned object is fully resolved. Pass False to get
                        the object with nothing selected, e.g. to inspect
                        ``full_config`` before choosing.
    :type auto_select: bool
    :return: An instantiated configuration object (:class:`DataSetConfig`,
             :class:`TrainingConfig`, :class:`ClassificationConfig`, or
             :class:`NRTQCConfig`).
    :rtype: ConfigBase
    :raises ValueError: If the combined stage and extension is not found in
                        the registry.

    Example Usage:
      >>> config = read_config_template("prepare")
      >>> config.dataset_name
      'dataset_0001'
      >>> config.get_target_names()
      ['temp', 'psal']
      >>> config.data["path_info"]["common"]["base_path"] = "~/aiqc_project/data"
    """
    template_name, config_class = _get_stage_template(stage, extension)

    return config_class(template_name, auto_select)


def read_config(
    file_name: str, set_name: Optional[str] = None, auto_select: bool = True
) -> ConfigBase:
    """
    Read a YAML configuration file as a :class:`ConfigBase` object,
    automatically selecting the appropriate subclass based on the content.

    This function:
      1. Resolves the file path by calling :func:`aiqclib.common.utils.config.get_config_file`.
      2. Reads the specified YAML file and identifies the main key
         (e.g., "data_sets", "training_sets", "classification_sets",
         or "nrt_qc_sets") to map to the corresponding configuration class.
      3. Instantiates and returns the matched configuration class with the resolved path.
      4. If ``set_name`` is provided, it calls the ``select`` method on the instantiated
         configuration object.

    :param file_name: The path (including filename) to the YAML file.
    :type file_name: str
    :param set_name: The name (key) of the desired configuration set within the YAML's dictionary.
                     Defaults to None.
    :type set_name: Optional[str]
    :param auto_select: If True and no ``set_name`` is given, select the file's
                        only set automatically; a file holding several sets
                        raises, since there is nothing to choose between them.
                        Ignored when ``set_name`` names the set to select.
                        Defaults to True.
    :type auto_select: bool
    :return: An instantiated configuration object (:class:`DataSetConfig`,
             :class:`TrainingConfig`, :class:`ClassificationConfig`, or
             :class:`NRTQCConfig`).
    :rtype: ConfigBase
    :raises ValueError: If no valid top-level configuration key is found in the YAML file.
    """
    config_file_name = get_config_file(file_name)
    config = utils_read_config(config_file_name)

    config_classes = {
        "data_sets": DataSetConfig,
        "training_sets": TrainingConfig,
        "classification_sets": ClassificationConfig,
        "nrt_qc_sets": NRTQCConfig,
    }
    matching_key = next((key for key in config_classes.keys() if key in config), None)
    if matching_key is None:
        raise ValueError("No valid 'set' name found in the provided YAML file.")

    # Auto-selection only makes sense when the caller did not name a set: it
    # rejects a file holding several sets, which is precisely the file a caller
    # passing 'set_name' is selecting from.
    config = config_classes[matching_key](
        config_file_name, auto_select and set_name is None
    )

    if set_name is not None:
        config.select(set_name)

    return config
