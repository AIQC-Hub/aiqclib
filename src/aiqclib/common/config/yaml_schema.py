"""
Module providing YAML-based JSON schemas used to validate dataset, training,
and classification configuration files. Each function returns a YAML string
describing the structure and constraints for a specific configuration schema.
"""


def get_data_set_config_schema() -> str:
    """
    Retrieve the YAML-based JSON schema for dataset configurations.

    The returned schema requires certain objects and properties (e.g.,
    path_info_sets, target_sets, feature_sets, etc.), each with nested
    type constraints and additional properties set to false when appropriate.

    :return: A YAML string representing the JSON schema for dataset configurations.
    :rtype: str
    """
    yaml_schema = """
---
type: object
properties:
  path_info_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        common:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          required:
            - base_path
          additionalProperties: false
        input:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          required:
            - base_path
            - step_folder_name
          additionalProperties: false
        select:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
        locate:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
        split:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
      required:
        - name
        - common
        - input
      additionalProperties: false

  target_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        variables:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              flag:
                type: string
              pos_flag_values:
                type: array
              neg_flag_values:
                type: array
            required:
              - name
              - flag
              - pos_flag_values
              - neg_flag_values
            additionalProperties: false
      required:
        - name
        - variables
      additionalProperties: false

  summary_stats_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        stats:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              col_names:
                type: array
                items:
                  type: string
            required:
              - name
              - col_names
      required:
        - name
        - stats
      additionalProperties: false

  feature_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        features:
          type: array
          items:
            type: string
      required:
        - name
        - features
      additionalProperties: false

  feature_param_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        params:
          type: array
          items:
            type: object
            properties:
              feature:
                type: string
              col_names:
                type: array
                items:
                  type: string
              stats_set:
                type: object
                properties:
                  name:
                    type: string
                  type:
                    type: string
              convert:
                type: string
              flank_up:
                type: integer
              flank_down:
                type: integer
              summary_stats_names:
                type: array
                items:
                  type: string
              stats:
                type: object
            required:
              - feature
              - col_names
            additionalProperties: false
      required:
        - name
        - params
      additionalProperties: false

  feature_stats_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        min_max:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              stats:
                type: object    
            required:
              - name  
              - stats
      required:
        - name
      additionalProperties: false

  step_class_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        steps:
          type: object
          properties:
            input:
              type: string
            summary:
              type: string
            select:
              type: string
            locate:
              type: string
            extract:
              type: string
            split:
              type: string
          required:
            - input
            - summary
            - select
            - locate
            - extract
            - split
          additionalProperties: false
      required:
        - name
        - steps
      additionalProperties: false

  step_param_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        type:
          type: string
        steps:
          type: object
          properties:
            input:
              type: object
              properties:
                sub_steps:
                  type: object
                  properties:
                    rename_columns:
                      type: boolean
                    filter_rows:
                      type: boolean
                  required:
                    - rename_columns
                    - filter_rows
                  additionalProperties: false
                rename_dict:
                  type: object
                filter_method_dict:
                  type: object
                  properties:
                    remove_years:
                      type: array
                    keep_years:
                      type: array
                  additionalProperties: false
              required:
                - sub_steps
              additionalProperties: false
            summary:
              type: object
            select:
              type: object
            locate:
              type: object
            extract:
              type: object
              properties:
                drop_key_columns:
                  type: boolean 
            split:
              type: object
          required:
            - input
            - summary
            - select
            - locate
            - extract
            - split
          additionalProperties: false
      required:
        - name
        - steps
      additionalProperties: false

  data_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        dataset_folder_name:
          type: string
        input_file_name:
          type: string
        path_info:
          type: string
        target_set:
          type: string
        summary_stats_set:
          type: string
        feature_set:
          type: string
        feature_param_set:
          type: string
        feature_stats_set:
          type: string
        step_class_set:
          type: string
        step_param_set:
          type: string
      required:
        - name
        - dataset_folder_name
        - input_file_name
        - path_info
        - target_set
        - summary_stats_set
        - feature_set
        - feature_param_set
        - feature_stats_set
        - step_class_set
        - step_param_set
      additionalProperties: false

additionalProperties: false
required:
  - path_info_sets
  - target_sets
  - summary_stats_sets
  - feature_sets
  - feature_param_sets
  - feature_stats_sets
  - step_class_sets
  - step_param_sets
  - data_sets
"""
    return yaml_schema


def get_training_config_schema() -> str:
    """
    Retrieve the YAML-based JSON schema for training configurations.

    The returned schema specifies required objects and properties under
    categories such as path_info_sets, target_sets, step_class_sets,
    step_param_sets, and training_sets. Additional properties are
    disallowed to ensure constraints remain strict.

    :return: A YAML string representing the JSON schema for training configurations.
    :rtype: str
    """
    yaml_schema = """
---
type: object
properties:
  path_info_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        common:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          required:
            - base_path
          additionalProperties: false
        input:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          required:
            - step_folder_name
          additionalProperties: false
        model:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
        validate:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
        build:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
      required:
        - name
        - common
        - input
      additionalProperties: false

  target_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        variables:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              flag:
                type: string
              pos_flag_values:
                type: array
              neg_flag_values:
                type: array
            required:
              - name
              - flag
              - pos_flag_values
              - neg_flag_values
            additionalProperties: false
      required:
        - name
        - variables
      additionalProperties: false

  step_class_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        steps:
          type: object
          properties:
            input:
              type: string
            validate:
              type: string
            model:
              type: string
            build:
              type: string
          required:
            - input
            - validate
            - model
            - build
          additionalProperties: false
      required:
        - name
        - steps
      additionalProperties: false

  step_param_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        type:
          type: string
        steps:
          type: object
          properties:
            input:
              type: object
            validate:
              type: object
            model:
              type: object
            build:
              type: object
          required:
            - input
            - validate
            - model
            - build
          additionalProperties: false
      required:
        - name
        - steps
      additionalProperties: false

  training_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        dataset_folder_name:
          type: string
        path_info:
          type: string
        target_set:
          type: string
        step_class_set:
          type: string
        step_param_set:
          type: string
      required:
        - name
        - dataset_folder_name
        - path_info
        - target_set
        - step_class_set
        - step_param_set
      additionalProperties: false

additionalProperties: false
required:
  - path_info_sets
  - target_sets
  - step_class_sets
  - step_param_sets
  - training_sets
"""
    return yaml_schema


def get_classification_config_schema() -> str:
    """
    Retrieve the YAML-based JSON schema for classification configurations.

    The returned schema requires certain objects and properties (e.g.,
    path_info_sets, target_sets, feature_sets, etc.), each with nested
    type constraints and additional properties set to false when appropriate.

    :return: A YAML string representing the JSON schema for classification configurations.
    :rtype: str
    """
    yaml_schema = """
---
type: object
properties:
  path_info_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        common:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          required:
            - base_path
          additionalProperties: false
        input:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          required:
            - base_path
            - step_folder_name
          additionalProperties: false
        select:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
        locate:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
        model:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
        classify:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
        concat:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
      required:
        - name
        - common
        - input
      additionalProperties: false

  target_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        variables:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              flag:
                type: string
              pos_flag_values:
                type: array
              neg_flag_values:
                type: array
            required:
              - name
              - flag
              - pos_flag_values
              - neg_flag_values
            additionalProperties: false
      required:
        - name
        - variables
      additionalProperties: false

  summary_stats_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        stats:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              col_names:
                type: array
                items:
                  type: string
            required:
              - name
              - col_names
      required:
        - name
        - stats
      additionalProperties: false

  feature_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        features:
          type: array
          items:
            type: string
      required:
        - name
        - features
      additionalProperties: false

  feature_param_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        params:
          type: array
          items:
            type: object
            properties:
              feature:
                type: string
              col_names:
                type: array
                items:
                  type: string
              stats_set:
                type: object
                properties:
                  name:
                    type: string
                  type:
                    type: string
              convert:
                type: string
              flank_up:
                type: integer
              flank_down:
                type: integer
              summary_stats_names:
                type: array
                items:
                  type: string
              stats:
                type: object
            required:
              - feature
              - col_names
            additionalProperties: false
      required:
        - name
        - params
      additionalProperties: false

  feature_stats_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        min_max:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              stats:
                type: object    
            required:
              - name  
              - stats
      required:
        - name
      additionalProperties: false

  step_class_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        steps:
          type: object
          properties:
            input:
              type: string
            summary:
              type: string
            select:
              type: string
            locate:
              type: string
            extract:
              type: string
            model:
              type: string
            classify:
              type: string
            concat:
              type: string
          required:
            - input
            - summary
            - select
            - locate
            - extract
            - model
            - classify
            - concat
          additionalProperties: false
      required:
        - name
        - steps
      additionalProperties: false

  step_param_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        type:
          type: string
        steps:
          type: object
          properties:
            input:
              type: object
              properties:
                sub_steps:
                  type: object
                  properties:
                    rename_columns:
                      type: boolean
                    filter_rows:
                      type: boolean
                  required:
                    - rename_columns
                    - filter_rows
                  additionalProperties: false
                rename_dict:
                  type: object
                filter_method_dict:
                  type: object
                  properties:
                    remove_years:
                      type: array
                    keep_years:
                      type: array
                  additionalProperties: false
              required:
                - sub_steps
              additionalProperties: false
            summary:
              type: object
            select:
              type: object
            locate:
              type: object
            extract:
              type: object
            model:
              type: object
            classify:
              type: object
            concat:
              type: object
          required:
            - input
            - summary
            - select
            - locate
            - extract
            - model
            - classify
            - concat
          additionalProperties: false
      required:
        - name
        - steps
      additionalProperties: false

  classification_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        dataset_folder_name:
          type: string
        input_file_name:
          type: string
        path_info:
          type: string
        target_set:
          type: string
        summary_stats_set:
          type: string
        feature_set:
          type: string
        feature_param_set:
          type: string
        feature_stats_set:
          type: string
        step_class_set:
          type: string
        step_param_set:
          type: string
      required:
        - name
        - dataset_folder_name
        - input_file_name
        - path_info
        - target_set
        - summary_stats_set
        - feature_set
        - feature_param_set
        - feature_stats_set
        - step_class_set
        - step_param_set
      additionalProperties: false

additionalProperties: false
required:
  - path_info_sets
  - target_sets
  - summary_stats_sets
  - feature_sets
  - feature_param_sets
  - feature_stats_sets
  - step_class_sets
  - step_param_sets
  - classification_sets
"""
    return yaml_schema
