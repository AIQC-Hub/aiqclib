# Contributing & Development

We welcome contributions! Please use the following guidelines for development.
For the release process and deployment, see [RELEASING.md](RELEASING.md).

## Environment Setup

We recommend using **uv** for managing the development environment.

1.  **Install `uv`.**
    We recommend installing `uv` into your base conda/mamba environment so the `uv` command is available globally without cluttering `base`. If you don't use conda/mamba, you can install it with pip instead.

```bash
    # Using mamba (recommended)
    mamba activate base
    mamba install -n base -c conda-forge uv

    # Or using conda
    conda activate base
    conda install -n base -c conda-forge uv

    # Or using pip
    pip install uv
```

    Alternatively, the [standalone installer](https://docs.astral.sh/uv/getting-started/installation/) from Astral works on any platform without needing Python or conda preinstalled.

2.  **Create and activate the project's virtual environment.**
    From the project's root directory, run the following:

```bash
    # Create the virtual environment in a .venv folder
    uv venv

    # Activate the virtual environment
    source .venv/bin/activate
```

3.  **Install the project and its dependencies.**
    This command pulls in all dependencies from `pyproject.toml` and installs the library itself in "editable" mode, so no separate install step is needed.

```bash
    uv sync
```

4.  **Download the test data.**
    The test fixtures (~15 MB of parquet, joblib, and YAML files) are not stored in the repository. They live as a GitHub release asset and need to be downloaded once before tests can run:

```bash
    bash scripts/fetch_test_data.sh
```

    This places the fixtures under `tests/data/`. The script requires the [`gh` CLI](https://cli.github.com) (authenticated via `gh auth login`) and `unzip`. To pin a specific data version or pull from a fork, override the defaults via environment variables:

```bash
    TEST_DATA_VERSION=test-data-v1.0.2 bash scripts/fetch_test_data.sh
```

    You only need to re-run this when the test data version changes.

## Running Tests

With your environment activated and test data downloaded, you can run the test suite using `pytest`.

```bash
uv run pytest -v
```

## Code Style (Linting & Formatting)

We use **Ruff** for linting and formatting.

**Linting:**
Check the library and test code for style issues.
```bash
# Lint the library source code
uv run ruff check src

# Lint the test code
uv run ruff check tests
```

**Formatting:**
Automatically format the code to match the project's style.
```bash
# Format the library source code
uv run ruff format src

# Format the test code
uv run ruff format tests
```
