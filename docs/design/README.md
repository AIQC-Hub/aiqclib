# Design documents

Internal design and planning documents for `aiqclib` features. These are
**not** part of the Sphinx/Read the Docs build (that is `docs/source/` only);
they live here as a durable record of *why* and *how* a feature was built.

## NRT QC module (v0.5.0)

| File | What it is |
| --- | --- |
| [`NRTQC_doc.md`](NRTQC_doc.md) | RTQC recommendation reference: the source material the module was based on. |
| [`NRTQC_spec.md`](NRTQC_spec.md) | Module specification: QC items, flag scheme, config layout, outputs. |
| [`NRTQC_plan.md`](NRTQC_plan.md) | The 8-phase implementation plan followed to build the module. |

## Profile-level mode

| File | What it is |
| --- | --- |
| [`PROFILE_spec.md`](PROFILE_spec.md) | Mode specification: profile labels (binary/proportion), feature levels, aggregation, config keys, outputs. |
| [`PROFILE_plan.md`](PROFILE_plan.md) | The 5-phase implementation plan followed to build the mode. |

## GPU acceleration

| File | What it is |
| --- | --- |
| [`GPU_findings.md`](GPU_findings.md) | What was measured running on GPUs, under what conditions, and the reasoning behind what was adopted and rejected. Includes the one question still open: whether the SHAP values earn their cost. |
