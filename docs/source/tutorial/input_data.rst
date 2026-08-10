Step 2: Preparing Input Data
============================

Before ``aiqclib`` can build a dataset, it needs a single Parquet file of
observation-level CTD data. There are two ways to get one:

*   **Download the example dataset** used throughout these tutorials. Start here
    if you are new to the library.
*   **Generate your own** from Copernicus NetCDF files with ``ctddump``, a
    companion command-line tool. Use this once you want to work with your own
    region or time span.

Either way you end up with a Parquet file in ``~/aiqc_project/input/``, which is
what :doc:`./preparation` consumes.

Setting Up the Project Directories
-----------------------------------

First, create the directories for your project. This structure is used
consistently throughout the tutorials:

.. code-block:: bash

   # Create a main project directory for all aiqclib outputs and configs
   mkdir -p ~/aiqc_project

   # Create subdirectories for configuration files and raw input data
   mkdir -p ~/aiqc_project/config
   mkdir -p ~/aiqc_project/input

Downloading the Example Dataset
--------------------------------

This tutorial uses the Copernicus Marine NRT CTD dataset, publicly available on
``Kaggle``. Both methods below place the data in ``~/aiqc_project/input/``.

.. tabs::

   .. tab:: Option 1: cURL (Recommended)

      The fastest way to get the data: it needs no tools or accounts beyond
      standard command-line utilities.

      1. **Download the zip file:**

         .. code-block:: bash

            curl -L -o ~/aiqc_project/input/data.zip \
              https://www.kaggle.com/api/v1/datasets/download/takaya88/copernicus-marine-nrt-ctd-data-for-aiqc

      2. **Unzip it into your input directory:**

         .. code-block:: bash

            unzip ~/aiqc_project/input/data.zip -d ~/aiqc_project/input/

   .. tab:: Option 2: Kaggle API

      Convenient if you already use the Kaggle client for other datasets.

      1. **Install and configure the Kaggle API:**

         .. code-block:: bash

            pip install kaggle

         Follow the official `Kaggle API authentication instructions <https://www.kaggle.com/docs/api#getting-started-installation-&-authentication>`_
         to obtain your ``kaggle.json`` file and place it in ``~/.kaggle/``.

      2. **Download and unzip in one command:**

         .. code-block:: bash

            kaggle datasets download -d takaya88/copernicus-marine-nrt-ctd-data-for-aiqc -p ~/aiqc_project/input --unzip

----------

Either way, you should now have a file named ``nrt_cora_bo_4.parquet`` inside
``~/aiqc_project/input/``. You can skip straight to :doc:`./preparation`.

Required Input Data Structure
-----------------------------

``aiqclib`` expects your input Parquet file to contain specific columns, which
identify unique profiles and observations. If your data already has them, you
are good to go; otherwise you may need to preprocess it.

The required columns are:

*   **platform_code**: A unique identifier for the measurement platform (e.g., buoy, ship).
*   **profile_no**: A unique, sequential number identifying each distinct "profile" (a set of measurements taken at a specific time and location) within a ``platform_code``.
*   **profile_timestamp**: The exact datetime of the profile. This column should be of a datetime type (e.g., Pandas/Polars datetime, or similar).
*   **longitude**: The longitude of the measurement profile.
*   **latitude**: The latitude of the measurement profile.
*   **observation_no**: A unique, sequential number identifying each individual observation (row) within a ``profile_no``.
*   **pres**: Pressure values for each observation.

Alongside these, you need the measurement columns you intend to model
(``temp``, ``psal``) and their QC flag columns (``temp_qc``, ``psal_qc``),
which supply the labels. ``pres`` is needed too, as an input feature and to
order observations within a profile, but it is not modelled — see
:ref:`choosing-targets`.

.. important::

   If your raw data lacks ``profile_no``, ``profile_timestamp``, or
   ``observation_no``, you will need to generate them. For detailed examples and
   helper code on how to perform these common data preprocessing steps (e.g.,
   converting float timestamps, generating unique IDs), please refer to the
   :doc:`../how-to/data_preprocessing_utilities` guide.

.. _which-qc-flags:

Which QC Flags Provide the Labels
----------------------------------

The ``temp_qc`` / ``psal_qc`` columns that these tutorials label from are
**near-real-time (NRT) QC** flags: the automated checks applied when
the data is first distributed. They are not delayed-mode (DMQC) flags, which
are assigned later, with expert review and the benefit of hindsight.

Neither route described on this page provides delayed-mode flags at all: the
example dataset ships the NRT flags only, and ``ctddump`` reads the NRT
products, so its output carries the NRT flags and nothing else. There is no
delayed-mode column to switch to in either file.

.. important::

   A model trained on these labels learns to **reproduce the NRT QC decisions**,
   including their mistakes. That is a legitimate goal (for instance, applying
   consistent NRT-quality screening to a new region), but it is a different task
   from predicting delayed-mode quality. Judge your evaluation metrics
   accordingly: agreement with the NRT flags is agreement with an automated
   system, not with ground truth.

Training on delayed-mode quality therefore requires data from a source that
carries such flags, which you would supply yourself. Once your input file has
them, point ``flag`` at those columns in the ``target_sets`` section of your
configuration; nothing else in the workflow changes:

.. code-block:: yaml

   target_sets:
     - name: target_set_1
       variables:
         - name: temp
           flag: temp_qc_dm        # a delayed-mode column from your own data
           pos_flag_values: [ 4, 6, 7 ]
           neg_flag_values: [ 1 ]

A related but separate module is available if what you want is to *compute* NRT
QC flags rather than learn from them: see the :doc:`../how-to/nrt_qc` guide,
which applies the standard RTQC tests directly and can compare its results
against the flags already in your file.

Generating Input Files with ``ctddump``
----------------------------------------

`ctddump <https://github.com/AIQC-Hub/ctddump>`_ is a companion command-line
tool that converts Copernicus CTD data from NetCDF to Parquet. Its output
schema is exactly what ``aiqclib`` expects, so a file it produces can be used as
input with no further preparation.

It supports the Near Real Time products for the Arctic (``nrt_ar``), Baltic
(``nrt_bo``), Mediterranean (``nrt_mo``) and Global (``nrt_gl``) seas, plus the
CORA reanalysis (``cora``, ``cora_legacy``).

Installing ctddump
~~~~~~~~~~~~~~~~~~~

The simplest option is a prebuilt binary, which bundles HDF5 and netCDF so
nothing else has to be installed. Download the archive for your platform from
the `latest release <https://github.com/AIQC-Hub/ctddump/releases/latest>`_ and
put the ``ctddump`` executable on your ``PATH``.

If you have a Rust toolchain and the HDF5 development headers, you can instead
install it from crates.io:

.. code-block:: bash

   # Ubuntu / Debian: sudo apt-get install libhdf5-dev libnetcdf-dev
   # macOS:           brew install hdf5
   cargo install ctddump

Building an Input File
~~~~~~~~~~~~~~~~~~~~~~~

Starting from a directory of NetCDF files, the pipeline below converts them,
merges them into one file, drops unusable profiles and removes duplicates. The
example uses the Baltic product; substitute the subcommand for your region.

.. code-block:: bash

   # 1. Convert every NetCDF file in the tree to Parquet (multi-threaded)
   ctddump batch convert nrt_bo --output ~/aiqc_project/work/pq /path/to/netcdf

   # 2. Merge them into one file, renumbering profile_no / observation_no
   ctddump concat convert ~/aiqc_project/work/pq ~/aiqc_project/work/merged.parquet

   # 3. Drop profiles whose temp, psal or pres is entirely missing
   ctddump dropna ~/aiqc_project/work/merged.parquet ~/aiqc_project/work/na.parquet

   # 4. Drop profiles flagged bad by the profile-level QC (time_qc / position_qc)
   ctddump dropqc ~/aiqc_project/work/na.parquet ~/aiqc_project/work/qc.parquet

   # 5. Mark and remove duplicate profiles
   ctddump markdup ~/aiqc_project/work/qc.parquet ~/aiqc_project/work/marked.parquet \
     ~/aiqc_project/work/duplicates.tsv
   ctddump dedup ~/aiqc_project/work/marked.parquet ~/aiqc_project/input/nrt_bo.parquet

The ``concat convert`` command already assigns ``profile_no`` and
``observation_no``, so the resulting file satisfies the required-column list
above as it stands. Point ``input_file_name`` in your configuration at it and
continue with :doc:`./preparation`.

To inspect what you produced before using it, ``ctddump report`` summarises a
Parquet file per platform, per profile, or as a whole:

.. code-block:: bash

   ctddump report parquet --level global --format text ~/aiqc_project/input/nrt_bo.parquet

.. note::

   ``ctddump`` writes QC flags as single-character strings (``"1"``, ``"4"``,
   ``""`` for missing) where other sources use integers. ``aiqclib`` reads both,
   and ``pos_flag_values`` / ``neg_flag_values`` may be written either way, so
   the same configuration works for both kinds of file. See
   :ref:`qc-flag-columns`.

.. important::

   The labels come from the QC flag columns, which here are the NRT flags of
   the source product (see `Which QC Flags Provide the Labels`_). The region and
   period you convert must therefore actually contain flagged-bad observations:
   if every ``temp_qc`` in your extract is ``1``, the preparation step will find
   no positive profiles and produce an empty training set. ``ctddump report``
   and the flag comparison report of the :doc:`../how-to/nrt_qc` module are both
   useful for checking this before you invest in a full run.

For the complete command reference, see the
`ctddump documentation <https://aiqc-hub.github.io/ctddump/>`_.

Next Steps
----------

With an input file in ``~/aiqc_project/input/``, you are ready to build a
training dataset.

Proceed to the next tutorial: :doc:`./preparation`.
