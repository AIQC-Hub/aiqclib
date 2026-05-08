Data Preprocessing Utilities
============================

This guide provides practical examples for preparing your raw data to meet the specific structural requirements of ``aiqclib``'s dataset preparation workflow. While ``aiqclib`` handles feature engineering and dataset splitting, it expects your input data to have certain key columns for identifying unique profiles and observations.

We'll focus on common transformations using the `Polars <https://pola.rs/>`_ library, which is a highly performant DataFrame library for Python.

Required Input Data Columns
---------------------------

For ``aiqclib`` to correctly process your data, your raw input Parquet file should contain the following columns:

*   **platform_code**: A unique string identifier for the measurement platform (e.g., a buoy ID, ship name).
*   **profile_no**: A unique, sequential integer number identifying each distinct "profile" within a ``platform_code``. A profile represents a single measurement event (e.g., all sensor readings at a specific time and location).
*   **profile_timestamp**: The exact datetime of the profile. This must be a proper datetime type (e.g., Polars ``Datetime`` with appropriate precision).
*   **longitude**: The longitude coordinate of the measurement profile (float).
*   **latitude**: The latitude coordinate of the measurement profile (float).
*   **observation_no**: A unique, sequential integer number identifying each individual observation (row) within a given ``profile_no``. This indicates the order of observations within a profile.
*   **pres**: Pressure values for each observation (float).

.. important::
   If your raw data is missing **profile_no**, **profile_timestamp**, or **observation_no**, or if these fields have invalid data types, the following steps will guide you through creating them.

.. important::
   Duplicate entries at the platform or profile level may result in incorrect datasets, even when following the examples on this page.

Example Data: Starting Point
----------------------------

Let's begin with an example Polars DataFrame that mimics raw data, notably missing the ``profile_no``, ``profile_timestamp`` (instead having a ``profile_time`` as a float), and ``observation_no`` columns.

.. code-block:: python

    import polars as pl
    from datetime import datetime
    import os

    # Define a consistent base path for saving/loading data
    project_root = os.path.expanduser("~/aiqc_project")
    input_dir = os.path.join(project_root, "input")
    os.makedirs(input_dir, exist_ok=True) # Ensure input directory exists

    # Example DataFrame (replace with your actual raw data loading)
    # This example assumes 'profile_time' is days since 1950-01-01
    df = pl.DataFrame({
        "platform_code": ["PLAT_A", "PLAT_A", "PLAT_A", "PLAT_B", "PLAT_A", "PLAT_B"],
        "profile_time": [14688.585416666667, 14688.585416666667, 14688.585416666667, 14690.0, 14688.585416666667, 14690.0],
        "longitude": [10.0, 10.0, 10.0, -1.5, 10.0, -1.5],
        "latitude": [45.0, 45.0, 45.0, 40.0, 45.0, 40.0],
        "pres": [1, 2, 3, 4, 5, 6],
        "some_other_feature": [100, 101, 102, 103, 104, 105], # Example additional feature
        "temp": [15.1, 14.9, 14.5, 20.0, 14.8, 19.5], # Example target variable
        "temp_qc": [3, 3, 1, 3, 3, 3] # Example QC flag for temp
    })

Step-by-Step Transformations
----------------------------

Follow these steps to transform your DataFrame into the structure expected by ``aiqclib``.

1. Create `profile_timestamp`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If your timestamp is a float representing days (e.g., days since a specific epoch like 1950-01-01), you'll need to convert it to a proper datetime column.

.. code-block:: python

    # Convert float datetime values to integer durations and then to a Polars datetime
    # Assuming 'profile_time' is in days since 1950-01-01
    df = df.with_columns(
        (
            # Start from the reference date (1950-01-01)
            pl.lit(datetime(1950, 1, 1)) +

            # Add whole days to the reference date
            pl.duration(days=pl.col("profile_time").floor()) +

            # Add fractional days, converted to seconds (1 day = 86400 seconds)
            pl.duration(seconds=(pl.col("profile_time") - pl.col("profile_time").floor()) * 86400)
        )
        .alias("profile_timestamp")  # Name the new column
        .cast(pl.Datetime("ms"))     # Store as a Polars datetime with millisecond precision
    )

2. Sort Rows
~~~~~~~~~~~~
Sorting the DataFrame is crucial before generating sequential ``profile_no`` and ``observation_no``. This ensures that observations belonging to the same profile are grouped together and are ordered correctly (e.g., by pressure).

.. code-block:: python

    df = df.sort(["platform_code", "profile_timestamp", "longitude", "latitude", "pres"])

3. Create `profile_key` (Helper Column)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A ``profile_key`` is a temporary helper column that uniquely identifies each distinct profile within a platform. This is a common pattern when ``profile_no`` is not directly available but can be inferred from a combination of other columns (e.g., ``platform_code``, ``profile_timestamp``, ``longitude``, ``latitude``).

.. code-block:: python

    df = df.with_columns(
        pl.format("{}|{}|{}|{}",
                  pl.col("platform_code"),
                  pl.col("profile_timestamp").cast(pl.Utf8), # Cast to string for formatting
                  pl.col("longitude"),
                  pl.col("latitude")
        ).alias("profile_key")
    )

4. Create `profile_no`
~~~~~~~~~~~~~~~~~~~~~~
Now, create the sequential ``profile_no`` within each ``platform_code`` by computing a dense rank of ``profile_key``. The ``rank("dense")`` ensures that the first distinct ``profile_key`` in a platform gets rank 1, the second gets rank 2, and so on.

.. code-block:: python

    df = df.with_columns(
        pl.col("profile_key")
          .rank("dense")
          .over("platform_code")
          .alias("profile_no")
    )

5. Create `observation_no`
~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``observation_no`` assigns a unique, sequential number to each observation *within* a given ``profile_no``. This is typically based on the order of records after sorting (e.g., by pressure depth). ``cum_count().over("profile_key")`` is used to count observations within each unique ``profile_key``. We add ``+1`` to make it 1-indexed.

.. code-block:: python

    df = df.with_columns(
        (pl.col("profile_key").cum_count().over("profile_key") + 1).alias("observation_no")
    )

6. Drop `profile_key` (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can optionally drop the helper ``profile_key`` column once ``profile_no`` and ``observation_no`` have been created, as it's no longer needed.

.. code-block:: python

    df = df.drop("profile_key")

Save the Preprocessed Data
--------------------------
Finally, save your preprocessed DataFrame as a Parquet file. This file will then be used as the ``input_file_name`` in your ``aiqclib`` ``prepare_config.yaml``.

.. code-block:: python

    # Define the output file path within your project's input directory
    output_file = os.path.join(input_dir, "nrt_cora_bo_preprocessed.parquet")

    # Write the DataFrame to a Parquet file
    df.write_parquet(output_file)

Next Steps
----------

With your raw data now structured correctly, you can proceed with the ``aiqclib`` dataset preparation workflow.

Return to the tutorial: :doc:`../tutorial/preparation`.
