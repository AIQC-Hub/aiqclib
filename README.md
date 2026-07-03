# aiqclib

[![PyPI - Version](https://img.shields.io/pypi/v/aiqclib)](https://pypi.org/project/aiqclib/)
[![Check Package](https://github.com/AIQC-Hub/aiqclib/actions/workflows/check_package.yml/badge.svg)](https://github.com/AIQC-Hub/aiqclib/actions/workflows/check_package.yml)
[![codecov](https://codecov.io/gh/AIQC-Hub/aiqclib/graph/badge.svg?token=8PSXE3Z28Y)](https://codecov.io/gh/AIQC-Hub/aiqclib)
[![CodeFactor](https://www.codefactor.io/repository/github/aiqc-hub/aiqclib/badge)](https://www.codefactor.io/repository/github/aiqc-hub/aiqclib)
[![DOI](https://zenodo.org/badge/1232803553.svg)](https://doi.org/10.5281/zenodo.20083179)

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
import aiqclib as aq

aq.write_config_template(file_name="/path/to/prepare_config.yaml", stage="prepare")
```

**Step 2: Customize `prepare_config.yaml`.**
You must edit the file to set the correct input/output paths and define your dataset. See the [Configuration](#configuration) section for details.

**Step 3: Run the preparation process.**
```python
import aiqclib as aq

config = aq.read_config("/path/to/prepare_config.yaml")
aq.create_training_dataset(config)
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
import aiqclib as aq

aq.write_config_template(file_name="/path/to/training_config.yaml", stage="train")
```

**Step 2: Customize `training_config.yaml`.**
Edit the file to point to your prepared dataset and define training parameters.

**Step 3: Train and evaluate the model.**
```python
import aiqclib as aq

config = aq.read_config("/path/to/training_config.yaml")
aq.train_and_evaluate(config)
```

This generates the following output folders:
- **validate**: Results from the cross-validation process.
- **build**: The final trained models and their evaluation results on the test dataset.

### 3. Data Classification

This workflow applies a trained model to classify all observations in a dataset.

**Step 1: Generate a classification configuration template.**

```python
import aiqclib as aq

aq.write_config_template(file_name="/path/to/classification_config.yaml", stage="classify")
```

**Step 2: Customize `classification_config.yaml`.**
Edit the file to point to the input data and the trained model.

**Step 3: Run classification.**
```python
import aiqclib as aq

config = aq.read_config("/path/to/classification_config.yaml")
aq.classify_dataset(config)
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

We welcome contributions! Development setup (uv environment, test data),
running tests, and code style are documented in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Releasing & Deployment (for Maintainers)

The release process (versioning checklist), building the docs, and deployment
to PyPI, conda-forge, and Anaconda.org are documented in
[RELEASING.md](RELEASING.md).
