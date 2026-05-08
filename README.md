# aiqclib

[![PyPI - Version](https://img.shields.io/pypi/v/aiqclib)](https://pypi.org/project/aiqclib/)
[![Conda - Version](https://img.shields.io/conda/vn/conda-forge/aiqclib)](https://anaconda.org/conda-forge/aiqclib)
[![Check Package](https://github.com/AIQC-Hub/aiqclib/actions/workflows/check_package.yml/badge.svg)](https://github.com/AIQC-Hub/aiqclib/actions/workflows/check_package.yml)
[![Codecov](https://codecov.io/gh/AIQC-Hub/aiqclib/graph/badge.svg?token=N6P5V9KBNJ)](https://codecov.io/gh/AIQC-Hub/aiqclib)
[![CodeFactor](https://www.codefactor.io/repository/github/aiqc-hub/aiqclib/badge)](https://www.codefactor.io/repository/github/aiqc-hub/aiqclib)
[![DOI](https://zenodo.org/badge/1003136460.svg)](https://doi.org/10.5281/zenodo.16055323)

**aiqclib** is a Python library that provides a configuration-driven workflow for machine learning, simplifying dataset preparation, model training, and data classification. It is a core component of the AIQC project that aims to enhance anomaly detection in CTD (Conductivity, Temperature, Depth) data.

## ML Algorithms Supported by **aiqclib**

| Category | Algorithm | Short Name | Method |
| :--- | :--- | :--- | :--- |
| Tree-Based & Ensemble | **XGBoost** | XGB | Ensemble (Boosting) |
| | **Random Forest** | RF | Ensemble (Bagging) |
| | **Decision Tree** | DT | Tree |
| Linear & Geometric | **Logistic Regression** | Logit | Linear |
| | **Linear Discriminant Analysis** | LDA | Linear / Statistical |
| | **Support Vector Machine** | SVM | Geometric |
| Instance-Based (Distance-Based) | **K-Nearest Neighbors** | KNN | Distance-based |
| Probabilistic | **Gaussian Naive Bayes** | GNB | Probabilistic |
| Neural Network | **Multilayer Perceptron** | MLP | Neural Network |

## Installation

The package is available on PyPI and conda-forge.

**Using pip:**
```bash
pip install aiqclib
```

**Using conda:**
```bash
conda install -c conda-forge aiqclib
```

## Documentation

Project documentation is hosted on [Read the Docs](https://aiqclib.readthedocs.io/en/latest/index.html).

## Core Concepts

The library is designed around a three-stage workflow:

1.  **Dataset Preparation:** Prepare feature datasets from raw data and generate training, validation, and test data sets.
2.  **Training & Evaluation:** Train machine learning models and evaluate their performance using cross-validation.
3.  **Classification:** Apply a trained model to classify new, unseen data.

Each stage is controlled by a YAML configuration file, allowing you to define and reproduce your entire workflow with ease.

## Usage

The general workflow for any task in `aiqclib` follows these steps:

1.  **Generate a Configuration Template:** Create a starter YAML file for the task (e.g., `prepare`, `train`, `classify`).
2.  **Customize the Configuration:** Edit the YAML file to specify paths, dataset names, and other parameters.
3.  **Run the Task:** Load the configuration and execute the main function for the task.

### 1. Dataset Preparation

This workflow processes your input data and creates training, validation, and test sets.

**Step 1: Generate a configuration template.**

```python
import aiqclib as dm

dm.write_config_template(file_name="/path/to/prepare_config.yaml", stage="prepare")
```

**Step 2: Customize `prepare_config.yaml`.**
You must edit the file to set the correct input/output paths and define your dataset. See the [Configuration](#configuration) section for details.

**Step 3: Run the preparation process.**
```python
import aiqclib as dm

config = dm.read_config("/path/to/prepare_config.yaml")
dm.create_training_dataset(config)
```

This generates the following output folders:
- **summary**: Statistics of input data used for normalization.
- **select**: Profiles with bad observation flags (positive samples) and good profiles (negative samples).
- **locate**: Observation records for both positive and negative profiles.
- **extract**: Features extracted from the observation records.
- **training**: The final training, validation, and test datasets.

### 2. Model Training and Evaluation

This workflow uses the prepared dataset to train a model and evaluate its performance.

**Step 1: Generate a training configuration template.**

```python
import aiqclib as dm

dm.write_config_template(file_name="/path/to/training_config.yaml", stage="train")
```

**Step 2: Customize `training_config.yaml`.**
Edit the file to point to your prepared dataset and define training parameters.

**Step 3: Train and evaluate the model.**
```python
import aiqclib as dm

config = dm.read_config("/path/to/training_config.yaml")
dm.train_and_evaluate(config)
```

This generates the following output folders:
- **validate**: Results from the cross-validation process.
- **build**: The final trained models and their evaluation results on the test dataset.

### 3. Data Classification

This workflow applies a trained model to classify all observations in a dataset.

**Step 1: Generate a classification configuration template.**

```python
import aiqclib as dm

dm.write_config_template(file_name="/path/to/classification_config.yaml", stage="classify")
```

**Step 2: Customize `classification_config.yaml`.**
Edit the file to point to the input data and the trained model.

**Step 3: Run classification.**
```python
import aiqclib as aq

config = aq.read_config("/path/to/classification_config.yaml")
dm.classify_dataset(config)
```

This workflow processes a dataset using a trained model and generates:
- **classify**: The final classification results and a summary report.

## Configuration

Configuration is managed via YAML files. The `write_config_template` function provides a starting point that you must customize for each module.

### 1. Dataset Preparation (`stage="prepare"`)

The preparation config requires you to modify two key sections:

- **`path_info_sets`**: Defines the location of input and output data.
  ```yaml
  path_info_sets:
    - name: data_set_1
      common:
        base_path: /path/to/data # EDIT: Root output directory
      input:
        base_path: /path/to/input # EDIT: Directory with input files
        step_folder_name: ""
      split:
        step_folder_name: training
  ```

- **`data_sets`**: Defines a specific dataset to be processed.
  ```yaml
  data_sets:
    - name: dataset_0001  # EDIT: Your data set name
      dataset_folder_name: dataset_0001  # EDIT: Your output folder
      input_file_name: nrt_cora_bo_4.parquet # EDIT: Your input filename
  ```

### 2. Training and Evaluation (`stage="train"`)

The training config links the prepared data to the model training process.

- **`path_info_sets`**: Defines where to find the prepared dataset and where to save model artifacts.
  ```yaml
  path_info_sets:
    - name: data_set_1
      common:
        base_path: /path/to/data # EDIT: Root output directory
      input:
        step_folder_name: training
  ```

- **`training_sets`**: Links to a dataset prepared in the previous workflow.
  ```yaml
  training_sets:
    - name: training_0001  # EDIT: Your training name
      dataset_folder_name: dataset_0001  # EDIT: Your output folder
  ```

### 3. Classification (`stage="classify"`)

The classification config uses a trained model to classify new data.

- **`path_info_sets`**: Defines paths for raw data, models, and classification results.
  ```yaml
  path_info_sets:
    - name: data_set_1
      common:
        base_path: /path/to/data # EDIT: Root output directory
      input:
        base_path: /path/to/input # EDIT: Directory with input files
        step_folder_name: ""
      model:
        base_path: /path/to/model  # EDIT: Directory with model files
        step_folder_name: model
      concat:
        step_folder_name: classification # EDIT: Directory with classification results
  ```

- **`classification_sets`**: Defines a specific dataset to be classified.
  ```yaml
  classification_sets:
    - name: classification_0001  # EDIT: Your classification name
      dataset_folder_name: dataset_0001  # EDIT: Your output folder
      input_file_name: nrt_cora_bo_4.parquet   # EDIT: Your input filename
  ```

## Contributing & Development

We welcome contributions! Please use the following guidelines for development.

### Environment Setup

We recommend using **uv** for managing the development environment.

1.  **Install `uv` into your base conda/mamba environment.**
    This makes the `uv` command available globally without cluttering your `base` environment.

    ```bash
    # Using mamba (recommended)
    mamba activate base
    mamba install -n base -c conda-forge uv

    # Or using conda
    conda activate base
    conda install -n base -c conda-forge uv
    ```

2.  **Create and activate the project's virtual environment.**
    From the project's root directory, run the following:

    ```bash
    # Create the virtual environment in a .venv folder
    uv venv

    # Activate the virtual environment
    source .venv/bin/activate
    ```

3.  **Install the project and its dependencies.**
    This command installs the library in "editable" mode (`-e`) and pulls in all dependencies from `pyproject.toml`.

    ```bash
    uv sync
    uv pip install -e .
    ```

### Running Tests

With your environment activated, you can run the test suite using `pytest`.

```bash
uv run pytest -v
```

### Code Style (Linting & Formatting)

We use **Ruff** for linting and formatting.

**Linting:**
Check the library and test code for style issues.
```bash
# Lint the library source code
uv ruff check src

# Lint the test code
ruff check tests
```

**Formatting:**
Automatically format the code to match the project's style.
```bash
# Format the library source code
ruff format src

# Format the test code
ruff format tests
```

## Documentation (for Maintainers)

### Building Docs Locally

1.  **Update Docstrings (Requires Google Gemini API Key):**
    ```bash
    # Update docstrings for source files
    python ./docs/scripts/update_docstrings.py src docs/scripts/prompt_main.txt

    # Update docstrings for test files
    python ./docs/scripts/update_docstrings.py tests docs/scripts/prompt_unittest.txt
    ```

2.  **Review Docstrings:**
    Manually review all modified files. Remove generated headers/footers and correct any sections marked with "Issues:".

3.  **Update API Documents:**
    From the project root, run:
    ```bash
    uv run sphinx-apidoc -f --remove-old --module-first -o docs/source/api src/aiqclib
    ```

4.  **Build HTML:**
    From the project root, run:
    ```bash
    cd docs; uv run make html; cd ..
    ```
    You can view the generated site by opening `docs/build/html/index.html` in a browser.

## Deployment (for Maintainers)

### PyPI

The package is published to [PyPI](https://pypi.org/project/aiqclib/) automatically via a GitHub Action whenever a new release is created on GitHub.

### conda-forge (Automatic)

The conda-forge bot automatically creates a pull request and merges it into the main branch when a new version of the package is published on PyPI.

### conda-forge (Manual)

#### Bump version with new dependencies

When runtime dependencies change, the automated PR from the conda-forge bot may fail. In that case, you must manually update the feedstock by creating a pull request to the `conda-forge/aiqclib-feedstock` repository in this case.

1.  **Install build tools:**
    ```bash
    mamba install -c conda-forge conda-build conda-smithy grayskull
    ```
2. **Fork and clone** the `aiqclib-feedstock` repository.
3. **Sync with upstream** (e.g., add `conda-forge/aiqclib-feedstock` as a remote named `upstream` and `git rebase upstream/main`).
4. **Update the forked repo:**
    ```bash
    git checkout main                      # Go to your local main branch
    git fetch upstream                     # Get latest changes from original repo
    git rebase upstream/main               # Make your local main perfectly linear with original
    git push origin main --force           # Update your GitHub fork's main (optional but good practice)
    ```
5. **Create a new branch** (e.g., `git checkout -b update_vX.Y.Z`).
6. **Generate a strict recipe** (e.g., `grayskull pypi aiqclib --strict-conda-forge`).
7. **Review `recipes/meta.yaml`** and ensure it meets `conda-forge` standards.
8. **Rerender the feedstock** (e.g., `conda smithy rerender -c auto`).
9. **Commit, push, and open a pull request** to the `staged-recipes` repository.
10. **Merge it** after passing CI.

#### Initial upload
Submitting the package on `conda-forge` involves creating a pull request to the `conda-forge/staged-recipes` repository.

1.  **Fork and clone** the `staged-recipes` repository.
2.  **Configure upstream** the `git remote add upstream https://github.com/conda-forge/aiqclib-feedstock.git`
3.  **Create a new branch** (e.g., `git checkout -b aiqclib-recipe`).
4.  **Generate a strict recipe:** `grayskull pypi aiqclib --strict-conda-forge`.
5.  **Review `recipes/aiqclib/meta.yaml`** and ensure it meets `conda-forge` standards.
6.  **Commit, push, and open a pull request** to the `staged-recipes` repository.

### Anaconda.org (Manual)

Publishing to the `<username>` channel on [Anaconda.org](https://anaconda.org/takayasaito/aiqclib) is a manual process.

1.  **Install build tools:**
    ```bash
    mamba install -c conda-forge conda-build anaconda-client grayskull
    ```
conda-smithy
2.  **Generate Recipe:**
    From the project root, run `grayskull pypi aiqclib`. This creates `aiqclib/meta.yaml`.

3.  **Build Package:**
    `conda build aiqclib`

4.  **Upload Package:**
    ```bash
    anaconda login
    anaconda upload /path/to/your/conda-bld/noarch/aiqclib-*.conda
    ```
5.  **Cleanup:**
    Copy `aiqclib/meta.yaml` to `conda/meta.yaml` for version control and remove the temporary `aiqclib` directory.
