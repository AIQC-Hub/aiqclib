Data Preprocessing Utilities
============================

``aiqclib`` expects a few identity and coordinate columns in your raw input.
Most of the preparation is now handled for you: after the optional column
**rename** step, ``aiqclib`` validates the required columns, auto-corrects their
data types, and can derive ``profile_no`` and ``observation_no``. The only thing
it cannot infer is a timestamp stored as a plain number (see below).

Required Input Data Columns
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Column
     - Type
     - Description
   * - ``platform_code``
     - text
     - Identifier for the platform (e.g. a buoy ID).
   * - ``profile_no``
     - integer
     - Sequential within a ``platform_code``; identifies one profile (a single
       measurement event).
   * - ``profile_timestamp``
     - datetime
     - The profile's date and time (a real ``Datetime``).
   * - ``longitude``
     - float
     - Longitude of the profile.
   * - ``latitude``
     - float
     - Latitude of the profile.
   * - ``observation_no``
     - integer
     - Sequential within a profile; the order of observations.
   * - ``pres``
     - float
     - Pressure for each observation.

Other columns (targets such as ``temp``, QC flags, etc.) are passed through
untouched. QC flag columns need no preparation whatever type they use — see
`QC Flag Columns`_ below.

What ``aiqclib`` does for you
-----------------------------

The ``input`` step can perform three preprocessing tasks, in this order:

.. list-table::
   :header-rows: 1
   :widths: 42 38 20

   * - Step
     - Controlled by (``input`` sub-step)
     - Default
   * - Rename columns to the required names
     - ``rename_columns`` + ``rename_dict``
     - off
   * - Create ``profile_no`` / ``observation_no``
     - ``create_columns`` + ``create_column_dict``
     - off
   * - Validate columns and correct their types
     - ``validate_columns``
     - on

**Renaming.** If your raw columns use different names, map them with
``rename_dict`` (``{ original_name: required_name }``) and set
``rename_columns: true``. Renaming runs first, so the create and validate steps
see the final column names.

**Creating identifiers.** ``profile_no`` is numbered within each
``platform_code`` and ``observation_no`` within each profile (ordered by
``pres``). Choose ``key_columns`` so they genuinely identify a profile: jittered
coordinates would split one profile, while identical timestamps at the same
location would merge two.

**Validating.** The required columns are checked, and mismatched types are
converted where possible — handy for CSV/TSV inputs, where numbers and dates
often arrive as strings.

Configure all three in the ``input`` step:

.. code-block:: yaml

   step_param_sets:
     - name: data_set_param_set_1
       steps:
         input:
           sub_steps:
             rename_columns: true
             filter_rows: false
             validate_columns: true      # on by default
             create_columns: true        # opt-in
           rename_dict: { date: profile_timestamp }
           create_column_dict:           # all keys optional
             key_columns:  [ platform_code, profile_timestamp, longitude, latitude ]
             sort_columns: [ platform_code, profile_timestamp, longitude, latitude, pres ]
             columns:      [ profile_no, observation_no ]

.. note::
   ``validate_columns`` is enabled by default. ``create_columns`` is **off** by
   default, so inputs that already provide these identifiers are never
   overwritten. Set ``columns: [ observation_no ]`` to create only one of them.

.. _qc-flag-columns:

QC Flag Columns
---------------

QC flag columns (``temp_qc``, ``psal_qc``, …) are not required columns, so the
validation above does not touch them. They are instead read leniently wherever
a configuration compares them against ``pos_flag_values`` / ``neg_flag_values``:

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Stored in the input as
     - Read as
   * - an integer ``4`` (any width, signed or unsigned)
     - ``4``
   * - a string ``"4"`` or ``"4.0"``
     - ``4``
   * - a float ``4.0``
     - ``4``
   * - ``""``, ``NaN``, ``null``, or a non-numeric code
     - nothing: matches neither the positive nor the negative list

The configured values are read the same way, so ``[ 4, 6, 7 ]`` and
``[ "4", "6", "7" ]`` select identically:

.. code-block:: yaml

   target_sets:
     - name: target_set_1
       variables:
         - name: temp
           flag: temp_qc
           pos_flag_values: [ 4, 6, 7 ]     # or [ "4", "6", "7" ]
           neg_flag_values: [ 1 ]           # or [ "1" ]

One configuration therefore serves inputs from different sources: a file whose
flags are integers and one whose flags are single-character strings (as
produced by ``ctddump``) both work, unchanged and with the same result.

.. note::
   The flag columns keep their original type in every output that carries input
   columns through, so a comparison report still shows the values your source
   wrote. Only the ``flag`` column that ``aiqclib`` *derives* in the
   row-selection step is normalized, to a 64-bit integer, so that datasets from
   different sources can be concatenated downstream.

A configured value that is not a whole number (``"bad"``, ``4.5``) raises a
``ValueError`` naming the offending value, rather than silently matching
nothing.

Converting a numeric timestamp
------------------------------

``profile_timestamp`` must be a real datetime. A numeric epoch (e.g. days since
1950-01-01) is ambiguous, so ``aiqclib`` cannot convert it automatically —
do this yourself before running the workflow:

.. code-block:: python

   import polars as pl
   from datetime import datetime

   # 'profile_time' is days since 1950-01-01
   df = df.with_columns(
       (
           pl.lit(datetime(1950, 1, 1))
           + pl.duration(days=pl.col("profile_time").floor())
           + pl.duration(
               seconds=(pl.col("profile_time") - pl.col("profile_time").floor()) * 86400
           )
       )
       .cast(pl.Datetime("ms"))
       .alias("profile_timestamp")
   )

.. important::
   Remove duplicate rows at the platform/profile level first — duplicates can
   produce incorrect datasets even when everything else is correct.

Save the Preprocessed Data
--------------------------

Write the result to Parquet and point ``input_file_name`` in your
``prepare_config.yaml`` at it:

.. code-block:: python

   output_file = "~/aiqc_project/input/nrt_cora_bo_preprocessed.parquet"
   df.write_parquet(output_file)

Next Steps
----------

Return to the tutorial: :doc:`../tutorial/input_data`.