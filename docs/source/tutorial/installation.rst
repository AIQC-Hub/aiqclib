Step 1: Installation
========================

The ``aiqclib`` library can be installed using several popular Python package managers. Standard installation methods using ``pip`` or ``conda`` are fully supported.

Standard Approaches
----------------------
You can use one of the following standard methods to install ``aiqclib``.

Using pip
~~~~~~~~~
You can install ``aiqclib`` directly from PyPI using ``pip``.

.. note::
   It is highly recommended to install the package inside a virtual environment (like ``venv`` or ``virtualenv``) to avoid conflicts with other projects or system packages. This is crucial for managing Python dependencies effectively.

.. code-block:: bash

   pip install aiqclib

Using conda or mamba
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``aiqclib`` package is available on the `conda-forge` channel, which is the recommended community-maintained channel for Conda packages. You can use either ``conda`` or ``mamba`` to install it.

.. code-block:: bash

   # Using conda (creates a new environment if ``aiqclib`` does not need to be found globally)
   conda install -c conda-forge aiqclib

   # Or using mamba (for a faster installation and better dependency resolution)
   mamba install -c conda-forge aiqclib

Using uv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you prefer to use ``uv`` for both creating virtual environments and installing packages, follow these steps. This method is an excellent pure-Python alternative for environment and package management.

**Step 1: Create and activate a virtual environment with uv**

``uv`` will create a new virtual environment in a ``.venv`` directory within your current working directory by default.

.. code-block:: bash

   # Create a virtual environment
   uv venv

   # Activate it (on Linux/macOS):
   source .venv/bin/activate
   # On Windows (in Command Prompt/PowerShell):
   .venv\Scripts\activate

**Step 2: Install aiqclib**

Once your ``uv``-managed virtual environment is activated, proceed with the installation:

.. code-block:: bash

   uv pip install aiqclib

Recommended Approach for Development: uv with mamba
---------------------------------------------------------
This method is highly recommended for all users, especially those managing multiple projects or complex dependencies.

.. tip::
   You can use `Miniforge <https://github.com/conda-forge/miniforge>`_ to install both conda and mamba together.

**Step 1: Install `uv` into your base mamba environment**

This makes the ``uv`` command available globally without cluttering your ``base`` environment.

.. code-block:: bash

   mamba activate base
   mamba install -n base -c conda-forge uv

**Step 2: Obtain the source code from GitHub**

.. code-block:: bash

   git clone https://github.com/AIQC-Hub/aiqclib.git

**Step 3: Navigate to the project root and create the virtual environment**

.. code-block:: bash

    cd ./aiqclib
    uv venv

**Step 4: Activate the uv virtual environment**

.. code-block:: bash

    source .venv/bin/activate

**Step 5: Install the dependencies**

.. code-block:: bash

    uv sync

**Step 6: Install the project**

This command installs the library in "editable" mode (``-e``).

.. code-block:: bash

    uv pip install -e .

Next Steps
----------
You have now successfully installed the ``aiqclib`` library! The next step in your journey is to understand how to prepare your raw data into a format suitable for model training.

Proceed to the next tutorial: :doc:`./preparation`.
