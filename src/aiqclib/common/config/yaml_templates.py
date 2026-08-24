"""
Module providing YAML templates for both dataset preparation
and training configurations. These templates can be customized
to fit various data pipeline requirements.

Each template is also registered under a ``template:`` identifier (see
:func:`get_template_names` and :func:`get_template_text` at the end of this
module), so a template can be named wherever a configuration file path is
expected.
"""

from typing import List


def _get_dataset_path_info_sets() -> str:
    """
    Retrieves a YAML template string for dataset path information sets.

    This template defines common, input, and split path configurations for dataset preparation.

    :returns: A string containing the YAML template for path information sets.
    :rtype: str
    """
    return """
---
path_info_sets:
  - name: data_set_1
    common:
      base_path: /path/to/data # EDIT: Root output directory
    input:
      base_path: /path/to/input # EDIT: Directory with input files
      step_folder_name: ""
    split:
      step_folder_name: training

"""


def _get_dataset_target_sets() -> str:
    """
    Retrieves a YAML template string for dataset target variable sets.

    This template specifies variables to be processed along with their positive
    and negative quality flag values.

    :returns: A string containing the YAML template for target sets.
    :rtype: str
    """
    return """
target_sets:
  - name: target_set_1
    variables:
      - name: temp
        flag: temp_qc
        # Flag values may be written as 4 or as "4"; the flag column itself may
        # be integer or string. Both sides are read as whole numbers.
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]
      - name: psal
        flag: psal_qc
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]
        
"""


def _get_dataset_summary_stats_sets() -> str:
    """
    Retrieves a YAML template string for dataset summary statistics sets.

    This template defines sets of column names for which summary statistics
    (e.g., location, profile summary stats, basic values) will be calculated.

    :returns: A string containing the YAML template for summary statistics sets.
    :rtype: str
    """
    return """
summary_stats_sets:
  - name: summary_stats_set_1
    stats:
      - name: location
        col_names: [ longitude, latitude ]
      - name: profile_summary_stats
        col_names: [ temp, psal, pres ]
      - name: basic_values3
        col_names: [ temp, psal, pres ]

"""


def _get_dataset_feature_sets() -> str:
    """
    Retrieves a YAML template string for dataset feature sets.

    This template lists named sets of features that will be used in the
    dataset preparation process, such as location, day of year, profile
    summary stats, basic values, and flank features.

    :returns: A string containing the YAML template for feature sets.
    :rtype: str
    """
    return """
feature_sets:
  - name: feature_set_1
    features:
      - location
      - day_of_year
      - profile_summary_stats
      - basic_values
      - flank_up
      - flank_down

"""


def _get_dataset_feature_param_sets() -> str:
    """
    Retrieves a YAML template string for dataset feature parameter sets.

    This template defines detailed parameters for each feature, including
    statistics type, column names, conversion methods (e.g., cosine for day_of_year),
    and summary statistics names.

    :returns: A string containing the YAML template for feature parameter sets.
    :rtype: str
    """
    return """
feature_param_sets:
  - name: feature_set_1_param_set_1
    params:
      - feature: location
        stats_set: { type: raw }
        col_names: [ longitude, latitude ]
      - feature: day_of_year
        convert: cosine
        col_names: [ profile_timestamp ]
      - feature: profile_summary_stats
        stats_set: { type: raw }
        col_names: [ temp, psal, pres ]
        summary_stats_names: [ mean, median, sd, pct25, pct75 ]
      - feature: basic_values
        stats_set: { type: raw }
        col_names: [ temp, psal, pres ]
      - feature: flank_up
        flank_up: 5
        stats_set: { type: raw }
        col_names: [ temp, psal, pres ]
      - feature: flank_down
        flank_down: 5
        stats_set: { type: raw }
        col_names: [ temp, psal, pres ]

"""


def _get_dataset_feature_param_sets_full() -> str:
    """
    Retrieves a YAML template string for dataset feature parameter sets with full normalization.

    This template defines detailed parameters for each feature, specifying
    normalization types (e.g., min_max) and associated statistics sets.

    :returns: A string containing the YAML template for full feature parameter sets.
    :rtype: str
    """
    return """
feature_param_sets:
  - name: feature_set_1_param_set_1
    params:
      - feature: location
        stats_set: { type: min_max, name: location }
        col_names: [ longitude, latitude ]
      - feature: day_of_year
        convert: cosine
        col_names: [ profile_timestamp ]
      - feature: profile_summary_stats
        stats_set: { type: min_max, name: profile_summary_stats }
        col_names: [ temp, psal, pres ]
        summary_stats_names: [ mean, median, sd, pct25, pct75 ]
      - feature: basic_values
        stats_set: { type: min_max, name: basic_values3 }
        col_names: [ temp, psal, pres ]
      - feature: flank_up
        flank_up: 5
        stats_set: { type: min_max, name: basic_values3 }
        col_names: [ temp, psal, pres ]
      - feature: flank_down
        flank_down: 5
        stats_set: { type: min_max, name: basic_values3 }
        col_names: [ temp, psal, pres ]

"""


def _get_dataset_feature_stats_sets() -> str:
    """
    Retrieves a YAML template string for dataset feature statistics sets.

    This template is typically used to define methods and parameters for
    feature normalization or other statistical transformations. In this
    basic version, it's an empty placeholder.

    :returns: A string containing the YAML template for feature statistics sets.
    :rtype: str
    """
    return """
feature_stats_sets:
  - name: feature_set_1_stats_set_1

"""


def _get_dataset_feature_stats_sets_full() -> str:
    """
    Retrieves a YAML template string for dataset feature statistics sets with full normalization details.

    This template defines explicit min-max statistics for various features
    (e.g., longitude, latitude, temperature, salinity, pressure) used for normalization.

    :returns: A string containing the YAML template for full feature statistics sets.
    :rtype: str
    """
    return """
feature_stats_sets:
  - name: feature_set_1_stats_set_1
    min_max:
      - name: location
        stats: { longitude: { min: 14.5, max: 23.5 },
                 latitude: { min: 55, max: 66 } }
      - name: profile_summary_stats
        stats: { temp: { mean: { min: 0, max: 12.5 },
                         median: { min: 0, max: 15 },
                         sd: { min: 0, max: 6.5 },
                         pct25: { min: 0, max: 12 },
                         pct75: { min: 1, max: 19 } },
                 psal: { mean: { min: 2.9, max: 12 },
                         median: { min: 2.9, max: 12 },
                         sd: { min: 0, max: 4 },
                         pct25: { min: 2.5, max: 8.5 },
                         pct75: { min: 3, max: 16 } },
                 pres: { mean: { min: 24, max: 105 },
                         median: { min: 24, max: 105 },
                         sd: { min: 13, max: 60 },
                         pct25: { min: 12, max: 53 },
                         pct75: { min: 35, max: 156 } } }
      - name: basic_values3
        stats: { temp: { min: 0, max: 20 },
                 psal: { min: 0, max: 20 },
                 pres: { min: 0, max: 200 } }

"""


def _get_dataset_step_class_sets() -> str:
    """
    Retrieves a YAML template string for dataset step class sets.

    This template maps each step in the dataset preparation pipeline (e.g.,
    input, summary, select, locate, extract, split) to its corresponding
    Python class name.

    :returns: A string containing the YAML template for step class sets.
    :rtype: str
    """
    return """
step_class_sets:
  - name: data_set_step_set_1
    steps:
      input: InputDataSetA
      summary: SummaryDataSetA
      select: SelectDataSetA
      locate: LocateDataSetA
      extract: ExtractDataSetA
      split: SplitDataSetA

"""


def _get_dataset_step_class_sets_all() -> str:
    """
    Retrieves a YAML template string for dataset step class sets with 'All' variants.

    This template maps each step in the dataset preparation pipeline (e.g.,
    input, summary, select, locate, extract, split) to its 'All' variant
    Python class name, indicating a broader application or default behavior.

    :returns: A string containing the YAML template for 'All' step class sets.
    :rtype: str
    """
    return """
step_class_sets:
  - name: data_set_step_set_1
    steps:
      input: InputDataSetA
      summary: SummaryDataSetA
      select: SelectDataSetAll
      locate: LocateDataSetAll
      extract: ExtractDataSetA
      split: SplitDataSetAll

"""


def _get_dataset_step_param_sets() -> str:
    """
    Retrieves a YAML template string for dataset step parameter sets.

    This template defines optional parameters for each step in the dataset
    preparation pipeline, such as input filtering, select ratio, locate neighbors,
    and split fractions or k-fold values.

    :returns: A string containing the YAML template for step parameter sets.
    :rtype: str
    """
    return """
step_param_sets:
  - name: data_set_param_set_1
    steps:
      input: { sub_steps: { rename_columns: false,
                            filter_rows: true },
               rename_dict: { },
               filter_method_dict: { remove_years: [ 2023 ],
                                     keep_years: [ ] } }
      summary: { }
      select: { neg_pos_ratio: 5 }
      locate: { neighbor_n: 5 }
      extract: { }
      split: { test_set_fraction: 0.1,
               k_fold: 5 }

"""


def _get_dataset_step_param_sets_all() -> str:
    """
    Retrieves a YAML template string for dataset step parameter sets with 'All' variants.

    This template defines optional parameters for each step in the dataset
    preparation pipeline, specifically designed for 'All' step variants,
    often implying default or less specific configurations.

    :returns: A string containing the YAML template for 'All' step parameter sets.
    :rtype: str
    """
    return """
step_param_sets:
  - name: data_set_param_set_1
    steps:
      input: { sub_steps: { rename_columns: false,
                            filter_rows: true },
               rename_dict: { },
               filter_method_dict: { remove_years: [ 2023 ],
                                     keep_years: [ ] } }
      summary: { }
      select: { }
      locate: { }
      extract: { }
      split: { test_set_fraction: 0.1,
               k_fold: 5 }

"""


def _get_dataset_data_sets() -> str:
    """
    Retrieves a YAML template string for defining individual data sets.

    This template specifies configurations for a particular dataset, including
    its folder and input file names, and references to other configuration
    sets (e.g., path info, target, summary stats, features, step classes, and step parameters).

    :returns: A string containing the YAML template for data sets.
    :rtype: str
    """
    return """
data_sets:
  - name: dataset_0001  # EDIT: Your data set name
    dataset_folder_name: dataset_0001  # EDIT: Your output folder
    input_file_name: nrt_cora_bo_4.parquet # EDIT: Your input filename
    path_info: data_set_1
    target_set: target_set_1
    summary_stats_set: summary_stats_set_1
    feature_set: feature_set_1
    feature_param_set: feature_set_1_param_set_1
    feature_stats_set: feature_set_1_stats_set_1
    step_class_set: data_set_step_set_1
    step_param_set: data_set_param_set_1

"""


def get_config_data_set_template() -> str:
    """
    Retrieve a YAML template string for dataset preparation configurations.

    This template includes:

    - ``path_info_sets``: specifying common, input, and split paths.
    - ``target_sets``: defining which variables to process and their flags.
    - ``summary_stats_sets``: defining summary statistics.
    - ``feature_sets``: listing named sets of feature extraction modules.
    - ``feature_param_sets``: detailing parameters for each feature.
    - ``feature_stats_sets``: detailing methods and stats for normalization.
    - ``step_class_sets``: referencing classes for each preparation step
      (e.g., input, summary, select, locate, extract, split).
    - ``step_param_sets``: referencing parameters for the preparation steps.
    - ``data_sets``: referencing specific dataset folders, files, and
      associated configuration sets (e.g., ``step_class_set``, ``step_param_set``).

    :returns: A string containing the YAML template.
    :rtype: str
    """
    return (
        _get_dataset_path_info_sets()
        + _get_dataset_target_sets()
        + _get_dataset_summary_stats_sets()
        + _get_dataset_feature_sets()
        + _get_dataset_feature_param_sets()
        + _get_dataset_feature_stats_sets()
        + _get_dataset_step_class_sets()
        + _get_dataset_step_param_sets()
        + _get_dataset_data_sets()
    )


def get_config_data_set_full_template() -> str:
    """
    Retrieve a YAML template string for dataset preparation configurations with normalization.

    This template includes:

    - ``path_info_sets``: specifying common, input, and split paths.
    - ``target_sets``: defining which variables to process and their flags.
    - ``summary_stats_sets``: defining summary statistics.
    - ``feature_sets``: listing named sets of feature extraction modules.
    - ``feature_param_sets``: detailing parameters for each feature.
    - ``feature_stats_sets``: detailing methods and stats for normalization.
    - ``step_class_sets``: referencing classes for each preparation step
      (e.g., input, summary, select, locate, extract, split).
    - ``step_param_sets``: referencing parameters for the preparation steps.
    - ``data_sets``: referencing specific dataset folders, files, and
      associated configuration sets (e.g., ``step_class_set``, ``step_param_set``).

    :returns: A string containing the YAML template.
    :rtype: str
    """
    return (
        _get_dataset_path_info_sets()
        + _get_dataset_target_sets()
        + _get_dataset_summary_stats_sets()
        + _get_dataset_feature_sets()
        + _get_dataset_feature_param_sets_full()
        + _get_dataset_feature_stats_sets_full()
        + _get_dataset_step_class_sets()
        + _get_dataset_step_param_sets()
        + _get_dataset_data_sets()
    )


def get_config_data_set_all_template() -> str:
    """
    Retrieve a YAML template string for dataset preparation configurations with 'All' step variants.

    This template includes:

    - ``path_info_sets``: specifying common, input, and split paths.
    - ``target_sets``: defining which variables to process and their flags.
    - ``summary_stats_sets``: defining summary statistics.
    - ``feature_sets``: listing named sets of feature extraction modules.
    - ``feature_param_sets``: detailing parameters for each feature.
    - ``feature_stats_sets``: detailing methods and stats for normalization.
    - ``step_class_sets``: referencing classes for each preparation step
      (e.g., input, summary, select, locate, extract, split) with 'All' variants.
    - ``step_param_sets``: referencing parameters for the preparation steps with 'All' variants.
    - ``data_sets``: referencing specific dataset folders, files, and
      associated configuration sets (e.g., ``step_class_set``, ``step_param_set``).

    :returns: A string containing the YAML template.
    :rtype: str
    """
    return (
        _get_dataset_path_info_sets()
        + _get_dataset_target_sets()
        + _get_dataset_summary_stats_sets()
        + _get_dataset_feature_sets()
        + _get_dataset_feature_param_sets()
        + _get_dataset_feature_stats_sets()
        + _get_dataset_step_class_sets_all()
        + _get_dataset_step_param_sets_all()
        + _get_dataset_data_sets()
    )


def _get_dataset_target_sets_profile() -> str:
    """
    Retrieves a YAML template string for profile-level target variable sets.

    Adds the ``label_mode`` key controlling the per-profile label: ``binary``
    (any bad observation, the default) or ``proportion`` (fraction of
    bad-flagged observations).

    :returns: A string containing the YAML template for profile target sets.
    :rtype: str
    """
    return """
target_sets:
  - name: target_set_1
    variables:
      - name: temp
        flag: temp_qc
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]
        label_mode: binary  # EDIT: binary or proportion
      - name: psal
        flag: psal_qc
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]
        label_mode: binary  # EDIT: binary or proportion

"""


def _get_dataset_feature_sets_profile() -> str:
    """
    Retrieves a YAML template string for profile-level feature sets.

    Profile-native features are used directly; observation-level features
    (e.g. ``basic_values``) are included via per-profile aggregation.

    :returns: A string containing the YAML template for profile feature sets.
    :rtype: str
    """
    return """
feature_sets:
  - name: feature_set_1
    features:
      - location
      - day_of_year
      - profile_summary_stats
      - basic_values

"""


def _get_dataset_feature_param_sets_profile() -> str:
    """
    Retrieves a YAML template string for profile-level feature parameter sets.

    Observation-level features must carry an ``agg`` list naming the
    per-profile aggregations of their columns (mean, min, max, median, std,
    sum, first, fail_frac, fail_any); profile-level features are used as-is.

    :returns: A string containing the YAML template for profile feature
              parameter sets.
    :rtype: str
    """
    return """
feature_param_sets:
  - name: feature_set_1_param_set_1
    params:
      - feature: location
        stats_set: { type: raw }
        col_names: [ longitude, latitude ]
      - feature: day_of_year
        convert: cosine
        col_names: [ profile_timestamp ]
      - feature: profile_summary_stats
        stats_set: { type: raw }
        col_names: [ temp, psal, pres ]
        summary_stats_names: [ mean, median, sd, pct25, pct75 ]
      - feature: basic_values
        stats_set: { type: raw }
        col_names: [ temp, psal, pres ]
        # EDIT: per-profile aggregations. Every output column must be unique:
        # e.g. 'mean' here would collide with profile_summary_stats' temp_mean.
        agg: [ min, max, std ]

"""


def _get_dataset_step_class_sets_profile() -> str:
    """
    Retrieves a YAML template string for profile-level step class sets.

    Steps 1-3 reuse the observation-level classes; the locate, extract, and
    split steps use their profile-level variants.

    :returns: A string containing the YAML template for profile step class sets.
    :rtype: str
    """
    return """
step_class_sets:
  - name: data_set_step_set_1
    steps:
      input: InputDataSetA
      summary: SummaryDataSetA
      select: SelectDataSetAll
      locate: LocateDataSetProfile
      extract: ExtractDataSetProfile
      split: SplitDataSetProfile

"""


def _get_dataset_step_param_sets_profile() -> str:
    """
    Retrieves a YAML template string for profile-level step parameter sets.

    Profile-level datasets are much smaller than observation-level ones, so
    the template suggests a smaller ``k_fold``.

    :returns: A string containing the YAML template for profile step
              parameter sets.
    :rtype: str
    """
    return """
step_param_sets:
  - name: data_set_param_set_1
    steps:
      input: { sub_steps: { rename_columns: false,
                            filter_rows: true },
               rename_dict: { },
               filter_method_dict: { remove_years: [ 2023 ],
                                     keep_years: [ ] } }
      summary: { }
      select: { }
      locate: { }
      extract: { }
      split: { test_set_fraction: 0.1,
               k_fold: 5 }

"""


def get_config_data_set_profile_template() -> str:
    """
    Retrieve a YAML template string for profile-level dataset preparation.

    Produces one labeled row per profile instead of one per observation:

    - ``target_sets``: variables with ``label_mode`` (binary or proportion).
    - ``feature_param_sets``: profile-native features used directly and
      observation-level features aggregated per profile via ``agg``.
    - ``step_class_sets``: the profile-level locate/extract/split classes.

    All other sections match the observation-level templates.

    :returns: A string containing the YAML template.
    :rtype: str
    """
    return (
        _get_dataset_path_info_sets()
        + _get_dataset_target_sets_profile()
        + _get_dataset_summary_stats_sets()
        + _get_dataset_feature_sets_profile()
        + _get_dataset_feature_param_sets_profile()
        + _get_dataset_feature_stats_sets()
        + _get_dataset_step_class_sets_profile()
        + _get_dataset_step_param_sets_profile()
        + _get_dataset_data_sets()
    )


def get_config_train_set_template() -> str:
    """
    Retrieve a YAML template string for training configurations.

    This template includes:

    - ``path_info_sets``: specifying common paths and subfolders for input, validate, and build.
    - ``target_sets``: defining variables and associated flags for training.
    - ``step_class_sets``: mapping each step (input, validate, model, build)
      to corresponding Python class names.
    - ``step_param_sets``: detailing optional parameters for each training step.
    - ``training_sets``: referencing specific dataset folders, the ``path_info`` used,
      the target set, and which ``step_class_set`` and ``step_param_set`` apply.

    :returns: A string containing the YAML template.
    :rtype: str
    """
    yaml_template = """
---
path_info_sets:
  - name: data_set_1
    common:
      base_path: /path/to/data # EDIT: Root output directory
    input:
      step_folder_name: training

target_sets:
  - name: target_set_1
    variables:
      - name: temp
        flag: temp_qc
        # Flag values may be written as 4 or as "4"; the flag column itself may
        # be integer or string. Both sides are read as whole numbers.
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]
      - name: psal
        flag: psal_qc
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]

step_class_sets:
  - name: training_step_set_1
    steps:
      input: InputTrainingSetA
      validate: KFoldValidation
      model: XGBoost
      build: BuildModel

step_param_sets:
  - name: training_param_set_1
    steps:
      input: { }
      validate: { k_fold: 5 }
      model: { calculate_shap: False,
               model_params: { scale_pos_weight: 200,
                               n_jobs: -1 } }
      build: { }

training_sets:
  - name: training_0001  # EDIT: Your training name
    dataset_folder_name: dataset_0001  # EDIT: Your output folder
    path_info: data_set_1
    target_set: target_set_1
    step_class_set: training_step_set_1
    step_param_set: training_param_set_1
"""
    return yaml_template


def get_config_train_set_profile_template() -> str:
    """
    Retrieve a YAML template string for profile-level training configurations.

    Mirrors :func:`get_config_train_set_template` but reads the profile-level
    prepare output (one row per profile) and shows both label modes:
    ``binary`` profile labels train with any classifier, while
    ``proportion`` labels require a regressor model (``XGBoostRegressor`` /
    ``RandomForestRegressor``). For a regressor, ``predicted_label_threshold``
    means "flag the profile when the predicted bad fraction reaches the
    threshold".

    :returns: A string containing the YAML template.
    :rtype: str
    """
    yaml_template = """
---
path_info_sets:
  - name: data_set_1
    common:
      base_path: /path/to/data # EDIT: Root output directory
    input:
      step_folder_name: training

target_sets:
  - name: target_set_1
    variables:
      - name: temp
        flag: temp_qc
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]
        label_mode: proportion  # EDIT: binary or proportion
      - name: psal
        flag: psal_qc
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]
        label_mode: proportion  # EDIT: binary or proportion

step_class_sets:
  - name: training_step_set_1
    steps:
      input: InputTrainingSetA
      validate: KFoldValidation
      # EDIT: proportion labels need a regressor (XGBoostRegressor or
      # RandomForestRegressor); binary labels use any classifier.
      model: XGBoostRegressor
      build: BuildModel

step_param_sets:
  - name: training_param_set_1
    steps:
      input: { }
      # Profile-level datasets are much smaller than observation-level ones;
      # keep k_fold modest and match the k_fold used at preparation time.
      validate: { k_fold: 5 }
      model: { calculate_shap: False,
               predicted_label_threshold: 0.5,
               model_params: { n_jobs: -1 } }
      build: { }

training_sets:
  - name: training_0001  # EDIT: Your training name
    dataset_folder_name: dataset_0001  # EDIT: Your output folder
    path_info: data_set_1
    target_set: target_set_1
    step_class_set: training_step_set_1
    step_param_set: training_param_set_1
"""
    return yaml_template


def _get_classify_path_info_sets() -> str:
    """
    Retrieves a YAML template string for classification path information sets.

    This template defines common, input, model, and concatenation path
    configurations for the classification process.

    :returns: A string containing the YAML template for classification path info sets.
    :rtype: str
    """
    return """
---
path_info_sets:
  - name: data_set_1
    common:
      base_path: /path/to/data # EDIT: Root output directory
    input:
      base_path: /path/to/input # EDIT: Directory with input files
      step_folder_name: ""
    model:
      base_path: /path/to/data/dataset_0001  # EDIT: Directory with model files
      step_folder_name: "model"
    concat:
      step_folder_name: classify # EDIT: Directory with classification results

"""


def _get_classify_step_class_sets() -> str:
    """
    Retrieves a YAML template string for classification step class sets.

    This template maps each step in the classification pipeline (e.g.,
    input, summary, select, locate, extract, model, classify, concat) to
    its corresponding Python class name, typically using 'All' variants.

    :returns: A string containing the YAML template for classification step class sets.
    :rtype: str
    """
    return """
step_class_sets:
  - name: data_set_step_set_1
    steps:
      input: InputDataSetAll
      summary: SummaryDataSetAll
      select: SelectDataSetAll
      locate: LocateDataSetAll
      extract: ExtractDataSetAll
      model: XGBoost
      classify: ClassifyAll
      concat: ConcatDataSetAll

"""


def _get_classify_step_param_sets() -> str:
    """
    Retrieves a YAML template string for classification step parameter sets.

    This template defines optional parameters for each step in the
    classification pipeline, such as input filtering rules, and general
    empty parameters for other steps like summary, select, locate, extract, model, classify, and concat.

    :returns: A string containing the YAML template for classification step parameter sets.
    :rtype: str
    """
    return """
step_param_sets:
  - name: data_set_param_set_1
    steps:
      input: { sub_steps: { rename_columns: false,
                            filter_rows: true },
               rename_dict: { },
               filter_method_dict: { remove_years: [ ],
                                     keep_years: [ 2023 ] } }
      summary: { }
      select: { }
      locate: { }
      extract: { }
      # skip_evaluation: True classifies unlabeled data (no QC flag) and skips
      # performance evaluation. Omit it to auto-detect per target from `flag`.
      model: { calculate_shap: False }
      classify: { }
      concat: { }

"""


def _get_classification_sets() -> str:
    """
    Retrieves a YAML template string for defining individual classification sets.

    This template specifies configurations for a particular classification run,
    including its folder and input file names, and references to other
    configuration sets (e.g., path info, target, summary stats, features,
    step classes, and step parameters).

    :returns: A string containing the YAML template for classification sets.
    :rtype: str
    """
    return """
classification_sets:
  - name: classification_0001  # EDIT: Your classification name
    dataset_folder_name: dataset_0001  # EDIT: Your output folder
    input_file_name: nrt_cora_bo_4.parquet   # EDIT: Your input filename
    path_info: data_set_1
    target_set: target_set_1
    summary_stats_set: summary_stats_set_1
    feature_set: feature_set_1
    feature_param_set: feature_set_1_param_set_1
    feature_stats_set: feature_set_1_stats_set_1
    step_class_set: data_set_step_set_1
    step_param_set: data_set_param_set_1

"""


def get_config_classify_set_template() -> str:
    """
    Retrieve a YAML template string for classification configurations.

    This template includes:

    - ``path_info_sets``: specifying common, input, model, and concatenation paths.
    - ``target_sets``: defining which variables to process and their flags.
    - ``summary_stats_sets``: defining summary statistics.
    - ``feature_sets``: listing named sets of feature extraction modules.
    - ``feature_param_sets``: detailing parameters for each feature.
    - ``feature_stats_sets``: detailing methods and stats for normalization.
    - ``step_class_sets``: referencing classes for each classification step
      (e.g., input, summary, select, locate, extract, model, classify, concat).
    - ``step_param_sets``: referencing parameters for the classification steps.
    - ``classification_sets``: referencing specific dataset folders, files, and
      associated configuration sets (e.g., ``step_class_set``, ``step_param_set``).

    :returns: A string containing the YAML template.
    :rtype: str
    """
    return (
        _get_classify_path_info_sets()
        + _get_dataset_target_sets()
        + _get_dataset_summary_stats_sets()
        + _get_dataset_feature_sets()
        + _get_dataset_feature_param_sets()
        + _get_dataset_feature_stats_sets()
        + _get_classify_step_class_sets()
        + _get_classify_step_param_sets()
        + _get_classification_sets()
    )


def get_config_classify_set_full_template() -> str:
    """
    Retrieve a YAML template string for classification configurations with normalization.

    This template includes:

    - ``path_info_sets``: specifying common, input, model, and concatenation paths.
    - ``target_sets``: defining which variables to process and their flags.
    - ``summary_stats_sets``: defining summary statistics.
    - ``feature_sets``: listing named sets of feature extraction modules.
    - ``feature_param_sets``: detailing parameters for each feature.
    - ``feature_stats_sets``: detailing methods and stats for normalization.
    - ``step_class_sets``: referencing classes for each classification step
      (e.g., input, summary, select, locate, extract, model, classify, concat).
    - ``step_param_sets``: referencing parameters for the classification steps.
    - ``classification_sets``: referencing specific dataset folders, files, and
      associated configuration sets (e.g., ``step_class_set``, ``step_param_set``).

    :returns: A string containing the YAML template.
    :rtype: str
    """
    return (
        _get_classify_path_info_sets()
        + _get_dataset_target_sets()
        + _get_dataset_summary_stats_sets()
        + _get_dataset_feature_sets()
        + _get_dataset_feature_param_sets_full()
        + _get_dataset_feature_stats_sets_full()
        + _get_classify_step_class_sets()
        + _get_classify_step_param_sets()
        + _get_classification_sets()
    )


def _get_nrtqc_path_info_sets() -> str:
    """
    Retrieves a YAML template string for NRT QC path information sets.

    This template defines the common, input, and concatenation path
    configurations for the NRT QC process.

    :returns: A string containing the YAML template for NRT QC path info sets.
    :rtype: str
    """
    return """
---
path_info_sets:
  - name: nrt_qc_path_1
    common:
      base_path: /path/to/data # EDIT: Root output directory
    input:
      base_path: /path/to/input # EDIT: Directory with input files
      step_folder_name: ""
    concat:
      step_folder_name: nrt_qc # EDIT: Directory with NRT QC results

"""


def _get_nrtqc_variable_sets() -> str:
    """
    Retrieves a YAML template string for NRT QC variable sets.

    This template lists the variables to run QC items on. The ``flag`` entry
    and the pos/neg flag values are optional; they are only used by the flag
    comparison step when the input already carries NRT QC flags.

    :returns: A string containing the YAML template for QC variable sets.
    :rtype: str
    """
    return """
qc_variable_sets:
  - name: qc_variable_set_1
    variables:
      - name: temp
        flag: temp_qc # Optional: existing NRT QC flag column for comparison
        # Values may be written as 4 or as "4"; the flag column itself may be
        # integer or string. Both sides are read as whole numbers.
        pos_flag_values: [ 4, 6, 7 ] # Optional: for agreement metrics
        neg_flag_values: [ 1 ]
      - name: psal
        flag: psal_qc
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]

"""


def _get_nrtqc_item_sets() -> str:
    """
    Retrieves a YAML template string for NRT QC item sets.

    This template enables QC items by name and shows the built-in default
    parameters, which can be overridden per item. Items also accept an
    optional ``fail_flag`` (3 or 4, default 4) to soften a failing test, and
    an optional ``include_in_final_flag`` (default :obj:`True`) to keep an
    item running while leaving it out of the aggregated NRT flag.
    Region-dependent values (e.g. regional_range) are edited per region,
    with one configuration file prepared for each region.

    :returns: A string containing the YAML template for QC item sets.
    :rtype: str
    """
    return """
qc_item_sets:
  - name: qc_item_set_1
    items:
      - name: impossible_date
      - name: impossible_location
      - name: global_range
        params: { temp: { min: -2.5, max: 40.0 },
                  psal: { min: 2.0, max: 41.0 } }
      - name: regional_range # EDIT: ranges of this file's region (Mediterranean)
        params: { temp: { min: 10.0, max: 40.0 },
                  psal: { min: 2.0, max: 40.0 } }
      - name: pressure_increasing
      - name: spike
        params: { temp: { shallow: 6.0, deep: 2.0 },
                  psal: { shallow: 0.9, deep: 0.3 },
                  depth_threshold: 500 }
      - name: gradient
        params: { temp: { shallow: 9.0, deep: 3.0 },
                  psal: { shallow: 1.5, deep: 0.5 },
                  depth_threshold: 500 }
      - name: digit_rollover
        params: { temp: 10.0, psal: 5.0 }
      - name: stuck_value
        # Add 'include_in_final_flag: false' to any item to keep its flag
        # column while leaving it out of the aggregated {variable}_nrt_flag.
      - name: density_inversion
        params: { threshold: 0.03 }
      - name: temp_to_psal

"""


def _get_nrtqc_step_class_sets() -> str:
    """
    Retrieves a YAML template string for NRT QC step class sets.

    This template maps each step in the NRT QC pipeline (input, qc, concat,
    compare) to its corresponding Python class name.

    :returns: A string containing the YAML template for NRT QC step class sets.
    :rtype: str
    """
    return """
step_class_sets:
  - name: nrt_qc_step_set_1
    steps:
      input: InputDataSetAll
      qc: QCDataSetAll
      concat: ConcatDataSetAll
      compare: CompareFlagsAll

"""


def _get_nrtqc_step_param_sets() -> str:
    """
    Retrieves a YAML template string for NRT QC step parameter sets.

    This template defines optional parameters for each step in the NRT QC
    pipeline (input, qc, concat, compare).

    :returns: A string containing the YAML template for NRT QC step parameter sets.
    :rtype: str
    """
    return """
step_param_sets:
  - name: nrt_qc_param_set_1
    steps:
      input: { sub_steps: { rename_columns: false,
                            filter_rows: false } }
      qc: { }
      concat: { }
      compare: { }

"""


def _get_nrtqc_sets() -> str:
    """
    Retrieves a YAML template string for defining individual NRT QC sets.

    This template specifies configurations for a particular NRT QC run,
    including its folder and input file names, and references to other
    configuration sets (path info, QC variables, QC items, step classes,
    and step parameters).

    :returns: A string containing the YAML template for NRT QC sets.
    :rtype: str
    """
    return """
nrt_qc_sets:
  - name: nrt_qc_0001  # EDIT: Your NRT QC set name
    dataset_folder_name: nrt_qc_0001  # EDIT: Your output folder
    input_file_name: nrt_cora_bo_4.parquet # EDIT: Your input filename
    path_info: nrt_qc_path_1
    qc_variable_set: qc_variable_set_1
    qc_item_set: qc_item_set_1
    step_class_set: nrt_qc_step_set_1
    step_param_set: nrt_qc_param_set_1

"""


def _get_classify_step_class_sets_profile() -> str:
    """
    Retrieves a YAML template string for profile-level classification step
    class sets.

    Steps 1-3 and the classify step reuse the observation-level classes; the
    locate, extract, and concat steps use their profile-level variants. The
    model must match the one the profiles were trained with (a regressor for
    ``label_mode: proportion``).

    :returns: A string containing the YAML template for profile classification
              step class sets.
    :rtype: str
    """
    return """
step_class_sets:
  - name: data_set_step_set_1
    steps:
      input: InputDataSetAll
      summary: SummaryDataSetAll
      select: SelectDataSetAll
      locate: LocateDataSetProfile
      extract: ExtractDataSetProfile
      model: XGBoost  # EDIT: XGBoostRegressor for proportion-label models
      classify: ClassifyAll
      concat: ConcatDataSetProfile

"""


def _get_classify_step_param_sets_profile() -> str:
    """
    Retrieves a YAML template string for profile-level classification step
    parameter sets.

    :returns: A string containing the YAML template for profile classification
              step parameter sets.
    :rtype: str
    """
    return """
step_param_sets:
  - name: data_set_param_set_1
    steps:
      input: { sub_steps: { rename_columns: false,
                            filter_rows: true },
               rename_dict: { },
               filter_method_dict: { remove_years: [ ],
                                     keep_years: [ 2023 ] } }
      summary: { }
      select: { }
      locate: { }
      extract: { }
      # skip_evaluation: True classifies unlabeled data (no QC flag) and skips
      # performance evaluation. Omit it to auto-detect per target from `flag`.
      model: { calculate_shap: False }
      classify: { }
      # broadcast_to_observations: true joins each profile's prediction onto
      # every observation of the profile instead of one row per profile.
      concat: { broadcast_to_observations: false }

"""


def get_config_classify_set_profile_template() -> str:
    """
    Retrieve a YAML template string for profile-level classification.

    Classifies one row per profile using models trained by the profile-level
    training stage: profile-native features are used directly, and
    observation-level features carry the same ``agg`` aggregations as at
    preparation time. Targets keep ``label_mode`` for evaluation; targets
    without a flag are classified label-free.

    :returns: A string containing the YAML template.
    :rtype: str
    """
    return (
        _get_classify_path_info_sets()
        + _get_dataset_target_sets_profile()
        + _get_dataset_summary_stats_sets()
        + _get_dataset_feature_sets_profile()
        + _get_dataset_feature_param_sets_profile()
        + _get_dataset_feature_stats_sets()
        + _get_classify_step_class_sets_profile()
        + _get_classify_step_param_sets_profile()
        + _get_classification_sets()
    )


def get_config_nrtqc_template() -> str:
    """
    Retrieve a YAML template string for NRT QC configurations.

    This template includes:

    - ``path_info_sets``: specifying common, input, and concatenation paths.
    - ``qc_variable_sets``: the variables to run QC items on, with optional
      existing-flag entries for the comparison step.
    - ``qc_item_sets``: the enabled QC items and their parameters.
    - ``step_class_sets``: referencing classes for each NRT QC step
      (input, qc, concat, compare).
    - ``step_param_sets``: referencing parameters for the NRT QC steps.
    - ``nrt_qc_sets``: referencing specific dataset folders, files, and
      associated configuration sets.

    :returns: A string containing the YAML template.
    :rtype: str
    """
    return (
        _get_nrtqc_path_info_sets()
        + _get_nrtqc_variable_sets()
        + _get_nrtqc_item_sets()
        + _get_nrtqc_step_class_sets()
        + _get_nrtqc_step_param_sets()
        + _get_nrtqc_sets()
    )


#: The built-in templates, keyed by the ``template:`` identifier that selects
#: one in place of a file path. This is the single registry: the config
#: classes read it to resolve a ``template:`` name, and
#: :mod:`aiqclib.interface.config` reads it to write a template out or to
#: build a configuration object from one.
_TEMPLATES = {
    "template:data_sets": get_config_data_set_template,
    "template:data_sets_all": get_config_data_set_all_template,
    "template:data_sets_full": get_config_data_set_full_template,
    "template:data_sets_profile": get_config_data_set_profile_template,
    "template:training_sets": get_config_train_set_template,
    "template:training_sets_profile": get_config_train_set_profile_template,
    "template:classification_sets": get_config_classify_set_template,
    "template:classification_sets_full": get_config_classify_set_full_template,
    "template:classification_sets_profile": get_config_classify_set_profile_template,
    "template:nrt_qc_sets": get_config_nrtqc_template,
}


def get_template_names() -> List[str]:
    """
    List the identifiers of every built-in template.

    :return: The ``template:``-prefixed names accepted wherever a
             configuration file path is expected.
    :rtype: list[str]
    """
    return list(_TEMPLATES)


def get_template_text(template_name: str) -> str:
    """
    Retrieve the YAML text of a built-in template by its identifier.

    :param template_name: A ``template:``-prefixed name, e.g.
                          ``"template:data_sets_full"``.
    :type template_name: str
    :return: The template's YAML text.
    :rtype: str
    :raises ValueError: If no template goes by that name.
    """
    if template_name not in _TEMPLATES:
        raise ValueError(f"Template name {template_name} is not supported.")

    return _TEMPLATES[template_name]()
