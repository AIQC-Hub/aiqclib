"""
Module providing YAML templates for both dataset preparation
and training configurations. These templates can be customized
to fit various data pipeline requirements.
"""


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
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]
      - name: psal
        flag: psal_qc
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]
      - name: pres
        flag: pres_qc
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
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]
      - name: psal
        flag: psal_qc
        pos_flag_values: [ 4, 6, 7 ]
        neg_flag_values: [ 1 ]
      - name: pres
        flag: pres_qc
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
