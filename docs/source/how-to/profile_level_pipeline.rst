Profile-Level Pipeline
======================

By default, ``aiqclib`` works at the **observation level**: one row per CTD
observation is labeled, trained on, and classified. The profile-level mode
instead produces **one row per profile** (a cast, identified by
``platform_code`` + ``profile_no``), so a model flags whole casts rather
than single observations.

The mode is selected the usual way, by naming the profile-level step
classes in ``step_class_sets``. Templates for all three stages exist:

.. code-block:: python

   import aiqclib

   aiqclib.write_config_template("prepare_profile.yaml", "prepare", "profile")
   aiqclib.write_config_template("train_profile.yaml", "train", "profile")
   aiqclib.write_config_template("classify_profile.yaml", "classify", "profile")

Profile labels
--------------

Each target variable gets an optional ``label_mode`` key:

.. code-block:: yaml

   target_sets:
     - name: target_set_1
       variables:
         - name: temp
           flag: temp_qc
           pos_flag_values: [ 4 ]
           neg_flag_values: [ 1 ]
           label_mode: proportion   # or binary (the default)

* ``binary`` (default): the profile label is ``1`` when **any** observation
  of the profile carries a positive flag, else ``0``. Trains with the
  existing classifiers.
* ``proportion``: the profile label is the **fraction of positive-flagged
  observations** among the observations carrying a valid (positive or
  negative) flag, a float in ``[0, 1]``. Trains with the regressor models
  ``XGBoostRegressor`` (``XGBR``) / ``RandomForestRegressor`` (``RFR``).

Profiles without any valid-flagged observation are dropped during
preparation. Observations with other flag values do not enter the
proportion's denominator, consistent with the observation-level pipeline,
which only ever trains on valid-flagged rows.

Feature levels and aggregation
------------------------------

Every feature class declares its level. Profile-level features produce one
value per profile and are used directly:

============================ =========================================
Feature                       Why profile-level
============================ =========================================
``location``                  longitude/latitude are profile metadata
``day_of_year``               derived from ``profile_timestamp``
``profile_summary_stats``     per-profile statistics by construction
``qc_impossible_date``        checks ``profile_timestamp``
``qc_impossible_location``    checks longitude/latitude
``qc_position_on_land``       checks the depth at the profile position
``qc_stuck_value``            compares all values of a profile
============================ =========================================

All other features are observation-level: ``basic_values``,
``flank_up``/``flank_down``, the seven profile-shape groups
(``geo_context``, ``derived_values``, ``stratification``,
``profile_smooth``, ``neighbor_diff``, ``rolling_stats`` and
``regime_flags``; see :doc:`profile_signal_features`) and the remaining
``qc_*`` items (including ``qc_pressure_increasing``, whose flag varies
within a profile). At profile level they **must** carry an ``agg`` list
naming per-profile aggregations of their columns; an observation-level
feature without ``agg`` is rejected with an error naming the feature:

.. code-block:: yaml

   feature_param_sets:
     - name: profile_features
       params:
         - feature: profile_summary_stats     # profile-level: used directly
           stats_set: { type: raw }
           col_names: [ temp, psal, pres ]
           summary_stats_names: [ mean, median, sd, pct25, pct75 ]
         - feature: basic_values              # observation-level: aggregated
           stats_set: { type: raw }
           col_names: [ temp, psal ]
           agg: [ min, max, std ]
         - feature: qc_spike                  # observation-level QC item
           col_names: [ temp ]
           agg: [ fail_frac, fail_any ]

Available aggregations: ``mean``, ``min``, ``max``, ``median``, ``std``,
``sum``, ``first``, plus two QC-flag-oriented ones: ``fail_frac`` (the
fraction of observations failing the item) and ``fail_any`` (1 when any
observation fails). Aggregated columns are named ``{column}_{agg}``.

.. note::

   Every output column must be unique. ``agg: [mean]`` on ``basic_values``
   would produce ``temp_mean`` (colliding with the ``temp_mean`` from
   ``profile_summary_stats``) and is rejected with an error listing the
   duplicated columns. Pick non-overlapping aggregation and summary
   statistic names.

Preparation
-----------

Steps 1-3 reuse the observation-level classes; locate, extract, and split
have profile-level variants:

.. code-block:: yaml

   step_class_sets:
     - name: profile_steps
       steps:
         input: InputDataSetA
         summary: SummaryDataSetA
         select: SelectDataSetAll
         locate: LocateDataSetProfile
         extract: ExtractDataSetProfile
         split: SplitDataSetProfile

The locate step writes two files per target: ``selected_rows_{target}``
(one labeled row per profile) and ``selected_observation_rows_{target}``
(the valid-flagged observation rows used to aggregate observation-level
features). The split step keeps the label-stratified splitting for binary
labels; proportion labels get a plain random test split and uniformly
random k-fold assignment.

.. note::

   Profile-level datasets are far smaller than observation-level ones (one
   row per cast). Use a modest ``k_fold`` and a ``test_set_fraction`` large
   enough that the sampled test set is not empty.

Training
--------

The training stage needs **no profile-specific step classes**; it consumes
the profile-level train/test files unchanged. Binary profile labels work
with all nine classifiers. Proportion labels require a regressor model:

.. code-block:: yaml

   step_class_sets:
     - name: training_step_set_1
       steps:
         input: InputTrainingSetA
         validate: KFoldValidation
         model: XGBoostRegressor    # or RandomForestRegressor
         build: BuildModel

For regressors, the ``score`` column of the prediction files holds the
predicted proportion (clipped to ``[0, 1]``), and
``predicted_label_threshold`` reads as "flag the profile when the predicted
bad fraction reaches the threshold" (see :doc:`prediction_threshold`).
Reports carry regression metrics (``mae``, ``rmse``, ``r2``,
``n_samples``) instead of the classification report, and the metric plots
show predicted-vs-actual and residual panels instead of ROC /
precision-recall curves.

A ``ModelSuite`` must be homogeneous: mixing classifier and regressor
methods in one suite is rejected at load time. Use separate configurations
for classifier and regressor runs.

Classification
--------------

The classify stage mirrors preparation: steps 1-3 and the classify step
reuse the observation-level classes:

.. code-block:: yaml

   step_class_sets:
     - name: profile_classify_steps
       steps:
         input: InputDataSetAll
         summary: SummaryDataSetAll
         select: SelectDataSetAll
         locate: LocateDataSetProfile
         extract: ExtractDataSetProfile
         model: XGBoostRegressor    # must match the trained model
         classify: ClassifyAll
         concat: ConcatDataSetProfile

Targets with a QC flag are evaluated against the same binary/proportion
labels as in preparation; targets without a flag are classified label-free
(see :doc:`classification_labels`); every profile is kept with a null
label.

The concat step writes ``predictions_profile.parquet`` with **one row per
profile**: the profile keys, ``profile_timestamp``/``longitude``/``latitude``,
and per target the ``{target}_label`` / ``{target}_predicted`` /
``{target}_score`` columns. To repeat each profile's prediction on every
observation instead, set:

.. code-block:: yaml

   step_param_sets:
     - name: profile_classify_params
       steps:
         concat: { broadcast_to_observations: true }

Known limitations
-----------------

* No positive/negative profile **pairing** (the ``SelectDataSetA``-style
  down-sampling) at profile level yet.
* The model step class is global per configuration, so one run cannot mix a
  classifier target with a regressor target; use separate configurations.
* Only ``XGBoostRegressor`` and ``RandomForestRegressor`` are provided;
  regressor variants of the other methods may follow.
