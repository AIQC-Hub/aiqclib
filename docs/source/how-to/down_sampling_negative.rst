Down-sampling of the Negative Dataset
=====================================

This guide demonstrates how to control the negative dataset during the data preparation stage.

Methods
-------

``aiqclib`` provides two methods to control the negative dataset:

1. Selection of negative profiles
2. Selection of neighboring observations within positive profiles

Preparation
-----------

The generation of negative data is controlled by a configuration file. The following command with ``extension="reduced"`` produces a template configuration file for controlling the negative dataset.

.. code-block:: python

   import aiqclib as aq
   import os

   config_path = os.path.expanduser("~/aiqc_project/config/prepare_config.yaml")
   aq.write_config_template(
       file_name=config_path,
       stage="prepare",
       extension="reduced"
   )

The ``step_class_sets`` and ``step_param_sets`` sections in this configuration template are different from the default template produced by ``extension=""``.

.. code-block:: yaml
   :emphasize-lines: 6, 7, 9

   step_class_sets:
     - name: data_set_step_set_1
       steps:
         input: InputDataSetA
         summary: SummaryDataSetA
         select: SelectDataSetA   # Not SelectDataSetAll
         locate: LocateDataSetA   # Not LocateDataSetAll
         extract: ExtractDataSetA
         split: SplitDataSetA     # Not SplitDataSetAll

.. code-block:: yaml
   :emphasize-lines: 10, 11

   step_param_sets:
     - name: data_set_param_set_1
       steps:
         input: { sub_steps: { rename_columns: false,
                               filter_rows: true },
                  rename_dict: { },
                  filter_method_dict: { remove_years:,
                                        keep_years: [] } }
         summary: { }
         select: { neg_pos_ratio: 5 }
         locate: { neighbor_n: 5 }
         extract: { }
         split: { test_set_fraction: 0.1,
                  k_fold: 10 }

Selection of Negative Profiles
------------------------------

Positive profiles are selected before the negative profile selection. Positive profiles are defined as profiles that have at least one flagged (bad/invalid) observation.

1. **Profile identification**: Positive profiles (with flagged observations) and negative profiles (without) are identified.
2. **Profile pairing**: Each positive profile is paired with several negative profiles based on date differences for contextual similarity.

The number of paired negative profiles is defined by ``neg_pos_ratio`` in the ``step_param_sets`` section.

.. code-block:: yaml
   :emphasize-lines: 10

   step_param_sets:
     - name: data_set_param_set_1
       steps:
         input: { sub_steps: { rename_columns: false,
                               filter_rows: true },
                  rename_dict: { },
                  filter_method_dict: { remove_years:,
                                        keep_years: [] } }
         summary: { }
         select: { neg_pos_ratio: 5 }
         locate: { neighbor_n: 5 }
         extract: { }
         split: { test_set_fraction: 0.1,
                  k_fold: 10 }

Once the pairs are formed, the observations of similar depth between the pairs are used to select negative observations. A pair usually produces a pair of positive and negative observations. For example, ``neg_pos_ratio: 5`` selects five negative profiles for each positive profile, which then produces five negative observations per positive observation.

Selection of Neighboring Observations
-------------------------------------
Negative observations can also be selected from positive profiles. When positive observations are identified and ``neighbor_n`` is set in the ``step_param_sets`` section, several upward and downward neighboring observations are selected unless they are also positive observations.

The number of neighboring negative observations is defined by ``neighbor_n`` in the ``step_param_sets`` section.

.. code-block:: yaml
   :emphasize-lines: 11

   step_param_sets:
     - name: data_set_param_set_1
       steps:
         input: { sub_steps: { rename_columns: false,
                               filter_rows: true },
                  rename_dict: { },
                  filter_method_dict: { remove_years:,
                                        keep_years: [] } }
         summary: { }
         select: { neg_pos_ratio: 5 }
         locate: { neighbor_n: 5 }
         extract: { }
         split: { test_set_fraction: 0.1,
                  k_fold: 10 }

For example, ``neighbor_n: 5`` selects up to 10 negative observations from both the upward and downward neighbors around a positive observation.
