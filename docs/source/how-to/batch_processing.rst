Running a Batch of Datasets
===========================

Working across several regions means repeating the same calls per dataset,
differing only in which named set each configuration selects:

.. code-block:: python

   import aiqclib as aq

   config = aq.read_config(prepare_config, set_name="dataset_ar_ar_0001")
   aq.create_training_dataset(config)

   config = aq.read_config(training_config, set_name="training_ar_ar_0001")
   aq.train_and_evaluate(config)

   # ... and again for every other region

``run_batch`` does that from a table of set names:

.. code-block:: python

   import aiqclib as aq

   summary = aq.run_batch(
       "datasets.txt",
       mode="all",
       prepare_config="prepare_config.yaml",
       training_config="training_config.yaml",
       classification_config="classification_config.yaml",
       verbose=True,
   )

Running Without a Table
-----------------------

The table can be omitted entirely. Each phase then runs once with no set name,
leaving every configuration file to select its own set:

.. code-block:: python

   summary = aq.run_batch(
       mode="all",
       prepare_config="prepare_config.yaml",
       training_config="training_config.yaml",
       classification_config="classification_config.yaml",
   )

This is the whole batch for a project whose config files hold a single set
each, and otherwise a compact way of running the phases in order. The summary
records which set each file selected, so the run is still traceable; the
``name`` column is null, because no dataset name was given.

.. note::

   A configuration file holding several sets cannot select one on its own, so
   it raises. Name the set with a table, or pass ``set_name`` yourself through
   :func:`~aiqclib.interface.config.read_config`.

The Dataset Table
-----------------

The first column names each dataset; the other columns give the configuration
set to select for each phase. Columns are separated by any run of whitespace,
so a hand-aligned table works as written:

.. code-block:: text

   # NRT and CORA region pairs
   name       prepare_set_name       training_set_name       classification_set_name
   ar_ar      dataset_ar_ar_0001     training_ar_ar_0001     classification_ar_ar_0001
   ar_gl      dataset_ar_gl_0001     training_ar_gl_0001     classification_ar_gl_0001
   cora_ar    dataset_cora_ar_0001   training_cora_ar_0001   classification_cora_ar_0001
   bo_bo      dataset_bo_bo_0001     training_bo_bo_0001     classification_bo_bo_0001
   cora_bo    dataset_cora_bo_0001   training_cora_bo_0001   classification_cora_bo_0001

Blank lines and ``#`` comment lines are ignored. A ``.tsv`` or ``.csv``
extension is read by that delimiter instead, which is what you need if a value
ever contains a space. A Polars DataFrame can be passed directly in place of a
path.

.. note::

   The identifier column is the one called ``name``, or the first column if
   there is none, so an existing table does not have to be renamed to be used.

Modes
-----

``mode`` selects which phases run:

.. list-table::
   :header-rows: 1
   :widths: 20 36 44

   * - Mode
     - Runs
     - Required column
   * - ``"prepare"``
     - :func:`~aiqclib.interface.prepare.create_training_dataset`
     - ``prepare_set_name``
   * - ``"train"``
     - :func:`~aiqclib.interface.train.train_and_evaluate`
     - ``training_set_name``
   * - ``"classify"``
     - :func:`~aiqclib.interface.classify.classify_dataset`
     - ``classification_set_name``
   * - ``"all"``
     - all three, in that order, per dataset
     - all three columns

Only the columns and configuration files a mode needs are required, so a table
for ``mode="prepare"`` needs nothing but the name and prepare columns. The
current list is always available from ``aq.available_modes()``.

A blank cell skips that phase for that dataset, which is how a region that
takes part in some phases but not others is expressed.

Running Part of a Table
-----------------------

``names`` restricts a run to particular datasets, so a single failing region
can be rerun without repeating the rest:

.. code-block:: python

   aq.run_batch(
       "datasets.txt",
       mode="train",
       training_config="training_config.yaml",
       names=["bo_bo", "cora_bo"],
   )

Failures
--------

By default a failing dataset stops the batch and its exception propagates, so
a problem cannot pass unnoticed. For a long unattended run, set
``continue_on_error=True``: the failure is recorded and the remaining datasets
still run.

.. code-block:: python

   summary = aq.run_batch(
       "datasets.txt",
       mode="all",
       prepare_config="prepare_config.yaml",
       training_config="training_config.yaml",
       classification_config="classification_config.yaml",
       continue_on_error=True,
   )

.. important::

   With ``continue_on_error=True`` the batch always ends normally, so the
   summary is the only record of what failed. Check it before treating a run
   as successful.

The Summary
-----------

Every run returns a Polars DataFrame with one row per dataset and phase:

.. code-block:: text

   ┌───────┬──────────┬─────────────────────┬────────┬─────────┐
   │ name  ┆ phase    ┆ set_name            ┆ status ┆ seconds │
   ╞═══════╪══════════╪═════════════════════╪════════╪═════════╡
   │ bo_bo ┆ prepare  ┆ dataset_0001        ┆ ok     ┆ 0.32    │
   │ bo_bo ┆ train    ┆ training_0001       ┆ ok     ┆ 2.86    │
   │ bo_bo ┆ classify ┆ classification_0001 ┆ ok     ┆ 0.99    │
   └───────┴──────────┴─────────────────────┴────────┴─────────┘

``status`` is ``ok``, ``skipped`` (a blank cell) or ``failed``, and the
``error`` column holds the exception type and message of a failure:

.. code-block:: python

   failures = summary.filter(pl.col("status") == "failed")
   print(failures.select("name", "phase", "error"))

Progress
--------

``verbose=True`` reports each dataset and phase as it starts, and is passed on
to the workflows themselves, so their own step reporting appears too:

.. code-block:: text

   [aiqclib] batch: 5 datasets x 3 phase(s) [prepare, train, classify]
   [aiqclib] batch: bo_bo / prepare (dataset_bo_bo_0001)
   [aiqclib] prepare: dataset_bo_bo_0001
   [aiqclib]   [1/6]    0.0s  Reading input data
   ...
   [aiqclib] batch: 15 runs in 812.4s (14 ok, 1 failed)
