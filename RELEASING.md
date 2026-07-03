# Releasing (for Maintainers)

This document covers the release process, documentation builds, and deployment.
For general development setup, tests, and linting, see the
[Contributing & Development](README.md#contributing--development) section of the README.

## Release Checklist

Follow these steps in order when cutting a new version. Steps marked *if
necessary* only apply when the relevant code has changed since the last release.

1. **Lint and format** *(if necessary)* — run Ruff over any changed code:
   ```bash
   uv run ruff check src tests
   uv run ruff format src tests
   ```
2. **Update API documentation** *(if necessary)* — regenerate the Sphinx API
   docs when the public API has changed. See [Building Docs Locally](#building-docs-locally).
3. **Update `CHANGELOG.md`** — record the changes for the new version.
4. **Bump the version** — update `version` in `pyproject.toml`, then sync the
   lockfile so `uv.lock` records the new project version:
   ```bash
   uv sync
   ```
5. **Commit, then create the release** — once merged, creating a new release on
   GitHub triggers the PyPI publish (see [Deployment](#deployment)).

## Building Docs Locally

1. **Update API Documents:**
    From the project root, run:
    ```bash
    uv run sphinx-apidoc -f --remove-old --module-first -o docs/source/api src/aiqclib
    ```

2. **Build HTML:**
    From the project root, run:
    ```bash
    cd docs; uv run make html; cd ..
    ```
    You can view the generated site by opening `docs/build/html/index.html` in a browser.

## Deployment

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
