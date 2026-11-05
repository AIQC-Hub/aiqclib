# Design documents

Internal design and planning documents for `aiqclib` features. These are
**not** part of the Sphinx/Read the Docs build (that is `docs/source/` only);
they live here as a durable record of *why* and *how* a feature was built.

## NRT QC module (moved to `nrtqclib`)

The NRT QC pipeline was built here in v0.5.0 and moved to
[`nrtqclib`](https://github.com/AIQC-Hub/nrtqclib) in 0.15.0. Its design docs
(`NRTQC_doc.md`, `NRTQC_spec.md`, `NRTQC_plan.md`) now live in that repo's
`docs/design/`. The QC items themselves stay in `aiqclib`.

| File | What it is |
| --- | --- |
| [`NRTQCLIB_plan.md`](NRTQCLIB_plan.md) | Plan for moving the NRT QC pipeline, config, docs and tests into `nrtqclib`, which depends on `aiqclib`. |

## Profile-level mode

| File | What it is |
| --- | --- |
| [`PROFILE_spec.md`](PROFILE_spec.md) | Mode specification: profile labels (binary/proportion), feature levels, aggregation, config keys, outputs. |
| [`PROFILE_plan.md`](PROFILE_plan.md) | The 5-phase implementation plan followed to build the mode. |

## Signal and context feature groups

| File | What it is |
| --- | --- |
| [`FEATURES_spec.md`](FEATURES_spec.md) | Feature specification: the source table, what each class emits, every parameter and its default, the missing-input rules. |
| [`FEATURES_plan.md`](FEATURES_plan.md) | The 4-phase implementation plan. |

## GPU acceleration

| File | What it is |
| --- | --- |
| [`GPU_findings.md`](GPU_findings.md) | What was measured running on GPUs, under what conditions, and the reasoning behind what was adopted and rejected. Includes the one question still open: whether the SHAP values earn their cost. |
