# NRT QC Module — Specification

Specification for the new **Near-Real Time Quality Control (NRT QC)** module of
`aiqclib`, derived from the RTQC recommendations in `NRTQC_doc.md`. Target data:
temperature and salinity profiles from CTD data across the **Arctic Sea, Baltic
Sea, and Mediterranean Sea**.

## 1. Purpose

The NRT QC module applies automated real-time QC tests to CTD profile data and:

1. Adds **one column per QC item** to the data (usable directly as training
   features by the Dataset Preparation / Classification modules).
2. Computes a **final NRT flag per variable** (`temp`, `psal`) by aggregating
   the individual item results.
3. Writes the **original parquet enriched with all QC-item columns and the
   final NRT flags** as its output.

The module is structurally similar to Dataset Preparation but much simpler:
read input → run configured QC items → aggregate flags → write output. Which
QC items run, their thresholds/regions, and the output column names are all
controlled by a YAML configuration file, following the existing
`*_sets` / `data_sets` config conventions.

## 2. Input

The standard `aiqclib` input parquet with the mandatory columns
(`REQUIRED_INPUT_COLUMNS`): `platform_code`, `profile_no`,
`profile_timestamp`, `longitude`, `latitude`, `observation_no`, `pres`,
plus the measured variables `temp` and `psal`. Existing QC/flag columns are
**not** required (NRT QC runs on unflagged data).

Rows are grouped into profiles by (`platform_code`, `profile_no`) and ordered
within a profile by `pres` (ascending; see RTQC8).

## 3. Flag scheme

Per-item and final flags use the standard IOC/Argo scheme (subset):

| Value | Meaning        |
|-------|----------------|
| 1     | good           |
| 3     | probably bad   |
| 4     | bad            |

Items produce flag `4` on failure (flag `1` otherwise) by default; the flag
value emitted on failure is configurable per item via `fail_flag` (e.g. to
soften a test to `3` where appropriate). Flag severity order for aggregation:
`1 < 3 < 4`.

## 4. QC items to implement

Ten items from `NRTQC_doc.md`, plus the temperature→salinity flag propagation
rule. "Level" states whether a failure flags a single observation (row), the
whole profile, or the profile's date/position metadata.

| # | Item | Doc ref | Level | Applies to |
|---|------|---------|-------|------------|
| 1 | Impossible date test | RTQC2 | profile | date |
| 2 | Impossible location test | RTQC3 | profile | position |
| 3 | Global range test | RTQC6 | observation | temp, psal |
| 4 | Regional range test | RTQC7 | observation | temp, psal |
| 5 | Pressure increasing test | RTQC8 | observation | pres (all variables) |
| 6 | Spike test | RTQC9 | observation | temp, psal |
| 7 | Gradient test | RTQC11 | observation | temp, psal |
| 8 | Digit rollover test | RTQC12 | observation | temp, psal |
| 9 | Stuck value test | RTQC13 | profile | temp, psal |
| 10 | Density inversion | RTQC14 | observation | temp + psal jointly |
| — | Temp→salinity propagation | §3 intro | observation | psal (from temp) |

### 4.1 RTQC2 — Impossible date test

`profile_timestamp` must be sensible: year > 1950 and not in the future
(timestamp ≤ processing time). Because the input column is already a parsed
datetime, structurally invalid dates (month 13, day 32, hour 24, …) cannot be
represented; they surface as **null** timestamps, which also fail the test.

- Fail → the profile's date is flagged: all rows of the profile get the fail
  flag in the item column.

### 4.2 RTQC3 — Impossible location test

- Latitude in range −90 to 90
- Longitude in range −180 to 180
- Null latitude/longitude fails.

Fail → the profile's position is flagged (all rows of the profile).

### 4.3 RTQC6 — Global range test

Gross filter on observed values:

- `temp` in range −2.5 °C to 40.0 °C
- `psal` in range 2.0 to 41.0

Fail → the individual value is flagged. Temp and psal are tested
independently; both may fail at the same depth.

> **Baltic note:** surface salinity in the northern Baltic (Bothnian Bay) can
> drop below the global minimum of 2.0. The regional range test (RTQC7) with a
> Baltic region must therefore be able to *relax* (not only restrict) the
> range, or the global bounds must be configurable. Ranges are config-driven
> (defaults above), so both options are available.

### 4.4 RTQC7 — Regional range test

Region-specific ranges for `temp` and `psal`. **One configuration file is
prepared per region** (Arctic, Baltic, Mediterranean), so region membership is
decided by which config is used — no polygon / point-in-polygon test is
needed. The item works exactly like the global range test but with the ranges
of the config's region; it is structurally the same check with different
bounds.

Reference values to seed each region's config:

| Region | temp (°C) | psal | Source |
|--------|-----------|------|--------|
| Mediterranean Sea | 10.0 to 40.0 | 2.0 to 40.0 | `NRTQC_doc.md` |
| Arctic Sea | −1.92 to 25.0 | 2.0 to 40.0 | `NRTQC_doc.md` |
| Baltic Sea | placeholder (permissive) | placeholder (permissive) | not in `NRTQC_doc.md` — to be tuned later |

All ranges are plain config values, so the Baltic entry can start permissive
and be tightened later without code changes. Fail → the individual value is
flagged.

### 4.5 RTQC8 — Pressure increasing test

Within a profile ordered from smallest to largest pressure, pressures must be
monotonically increasing:

- Constant-pressure run → all but the first of the consecutive constant
  pressures are flagged.
- Pressure reversal → all pressures in the reversed segment are flagged.

Fail → the affected rows are flagged for all variables (pressure is shared).

### 4.6 RTQC9 — Spike test

For each interior observation V2 with neighbours V1 (above) and V3 (below):

```
test_value = |V2 − (V3 + V1)/2| − |(V3 − V1)/2|
```

| Variable | pres < 500 db | pres ≥ 500 db |
|----------|---------------|----------------|
| temp     | > 6.0 °C      | > 2.0 °C       |
| psal     | > 0.9         | > 0.3          |

Fail → V2 is flagged. First/last observations of a profile are not testable
(flag stays 1).

### 4.7 RTQC11 — Gradient test

Same V1/V2/V3 stencil:

```
test_value = |V2 − (V3 + V1)/2|
```

| Variable | pres < 500 db | pres ≥ 500 db |
|----------|---------------|----------------|
| temp     | > 9.0 °C      | > 3.0 °C       |
| psal     | > 1.5         | > 0.5          |

Fail → V2 is flagged.

### 4.8 RTQC12 — Digit rollover test

Difference between vertically adjacent observations:

- `temp`: |Δ| > 10 °C
- `psal`: |Δ| > 5

Fail → the value is flagged.

### 4.9 RTQC13 — Stuck value test

All measurements of a variable in a profile are identical (profiles with a
single observation are exempt). Fail → **all** values of that variable in the
profile are flagged.

### 4.10 RTQC14 — Density inversion

Compute potential density σ₀ from `temp`, `psal`, `pres` per observation using
the UNESCO 1983 (EOS-80) algorithm. Compare consecutive levels in both
directions with a single configurable threshold Δσ (default 0.03 kg/m³).
Because configuration files are per-region, a region-specific threshold is
simply a different value in that region's config — no extra mechanism needed.

- Top→bottom: σ₀ at greater pressure < σ₀ at lesser pressure − Δσ → fail.
- Bottom→top: σ₀ at lesser pressure > σ₀ at greater pressure + Δσ → fail.

Fail → **both** `temp` and `psal` are flagged at the affected levels.

> **Implementation note:** implement EOS-80 σ₀ (Fofonoff & Millard 1983) as a
> small internal utility (pure numpy/polars expressions) rather than adding a
> `gsw`/TEOS-10 dependency; the doc explicitly prescribes the UNESCO 1983
> algorithm. The utility lives in `common/utils/` for potential reuse.

### 4.11 Temperature→salinity flag propagation

From the introduction of `NRTQC_doc.md`: when salinity is derived from
temperature and conductivity, a temperature flagged `4` (or `3`) forces the
salinity at the same observation to `4` (or `3`).

Applied as the **last step of aggregation**: if the final `temp` NRT flag of an
observation is worse than the final `psal` NRT flag, the `psal` flag is raised
to match. Recorded in its own item column so the propagation is traceable.
Like every other item, it runs only when listed in the active `qc_item_set` —
datasets with independently measured salinity simply omit it from their
configuration.

## 5. Excluded items (and why)

| Item | Reason |
|------|--------|
| RTQC1 Platform identification | GTS/Argo-specific (WMO/ptt matching) |
| RTQC4 Position on land | Requires external bathymetry (ETOPO2); candidate for a later version |
| RTQC5 Impossible speed | Argo/GTS drift-specific |
| RTQC10 Bottom spike | XBT only |
| RTQC15 Grey list | Argo DAC infrastructure |
| RTQC16 Sensor drift | Requires previous-profile history per platform |
| RTQC17 Frozen profile | Requires previous-profile history per platform |
| RTQC18 Deepest pressure | Requires instrument metadata (DEEPEST_PRESSURE) |

## 6. Output

The output parquet = **all original input columns** plus:

1. **One column per enabled QC item.** Naming pattern (configurable):
   - Variable-specific items: `{variable}_qc_{item}` — e.g.
     `temp_qc_global_range`, `psal_qc_spike`.
   - Profile-level, variable-independent items: `qc_{item}` — e.g.
     `qc_impossible_date`, `qc_impossible_location`, and the shared
     `qc_pressure_increasing`.
   - Values: the flag scheme of §3 (integer; 1 = pass). Never null for enabled
     items, so the columns can be consumed directly as model features.
2. **Final NRT flag per variable**: `temp_nrt_flag`, `psal_nrt_flag` — the
   worst (most severe) flag among all item columns applicable to that
   variable (variable-specific items + profile-level items), after the
   temp→salinity propagation of §4.11.

Item short names (used in column names and config): `impossible_date`,
`impossible_location`, `global_range`, `regional_range`,
`pressure_increasing`, `spike`, `gradient`, `digit_rollover`, `stuck_value`,
`density_inversion`, `temp_to_psal`.

## 7. Configuration

Follows the existing config conventions (named `*_sets` referenced from
`data_sets`), validated by a new NRT QC jsonschema. Sketch:

```yaml
path_info_sets:
  - name: nrt_qc_path_1
    common: { base_path: /path/to/data }
    input: { base_path: /path/to/input, step_folder_name: "" }
    output: { step_folder_name: nrt_qc }

qc_variable_sets:
  - name: qc_variable_set_1
    variables:
      - name: temp
      - name: psal

qc_item_sets:
  - name: qc_item_set_1
    items:
      - name: impossible_date          # enabled by listing it
      - name: impossible_location
      - name: global_range
        params:
          temp: { min: -2.5, max: 40.0 }
          psal: { min: 2.0, max: 41.0 }
      - name: regional_range           # this config file: Mediterranean
        params:
          temp: { min: 10.0, max: 40.0 }
          psal: { min: 2.0, max: 40.0 }
      - name: pressure_increasing
      - name: spike
        params:
          temp: { shallow: 6.0, deep: 2.0 }
          psal: { shallow: 0.9, deep: 0.3 }
          depth_threshold: 500          # db
      - name: gradient
        params:
          temp: { shallow: 9.0, deep: 3.0 }
          psal: { shallow: 1.5, deep: 0.5 }
          depth_threshold: 500
      - name: digit_rollover
        params: { temp: 10.0, psal: 5.0 }
      - name: stuck_value
      - name: density_inversion
        params: { threshold: 0.03 }
      - name: temp_to_psal
    # optional per-item overrides: fail_flag: 3, output column name, …

step_class_sets:
  - name: nrt_qc_step_set_1
    steps:
      input: InputDataSetAll
      qc: QCDataSetAll
      concat: ConcatDataSetAll

step_param_sets:
  - name: nrt_qc_param_set_1
    steps:
      input: { sub_steps: { rename_columns: false, filter_rows: false } }
      qc: { }
      concat: { }

data_sets:
  - name: nrt_qc_0001
    dataset_folder_name: nrt_qc_0001
    input_file_name: nrt_cora_bo_4.parquet
    path_info: nrt_qc_path_1
    qc_variable_set: qc_variable_set_1
    qc_item_set: qc_item_set_1
    step_class_set: nrt_qc_step_set_1
    step_param_set: nrt_qc_param_set_1
```

Defaults: every item ships built-in default thresholds (the values in §4);
`params` only overrides them. An item runs only if listed in the active
`qc_item_set`.

**One configuration file per region** (Arctic, Baltic, Mediterranean): the
files share the same structure and differ only in region-dependent `params`
(regional/global ranges, density-inversion threshold, `fail_flag` choices).

## 8. Module layout

New package `src/aiqclib/nrtqc/` mirroring the existing stage structure, but
with only three steps:

```
src/aiqclib/nrtqc/
├── step1_read_input/     # reuse prepare's input reading + validation
├── step2_run_qc/         # one class per QC item + a runner that applies
│                         #   the configured items and adds item columns
└── step3_concat_flags/   # aggregate item columns → {var}_nrt_flag,
                          #   apply temp→psal propagation, write parquet
```

- Each QC item is a small class with a common base (e.g. `QCItemBase`) exposing
  `apply(df) -> pl.DataFrame` that appends its column(s); items are registered
  by short name so the config can enable them by name.
- All computations are vectorised polars expressions over
  (`platform_code`, `profile_no`) windows — no per-profile Python loops.
- No stdout/stderr output from the library (consumers surface warnings).
- Interface: `aiqclib.interface` gets the matching high-level entry points
  (mirroring the existing per-module functions) plus a config template/schema
  for the new module.

## 9. Testing

- Unit tests per QC item (`tests/test_nrtqc_*.py`) with small synthetic
  profiles covering pass/fail/edge cases (profile boundaries, nulls, single
  observation profiles, constant/reversed pressure segments).
- One integration test running the full module on the existing CTD test
  fixture and checking output columns + final flags.

## 10. Resolved decisions

1. **Regional handling**: one configuration file per region (Arctic, Baltic,
   Mediterranean) — region membership is decided by config choice, not by
   coordinates, so RTQC7 needs no polygon test. Baltic ranges are not yet
   available; start with permissive placeholders and tune later (see §4.4,
   incl. the Baltic low-salinity conflict with the global range minimum).
2. **Density inversion threshold**: single configurable default (0.03 kg/m³);
   per-region values, when known, are simply set in each region's config.
3. **Fail flag severity**: default `4` for all items, with a per-item
   `fail_flag` config override so individual tests can be softened to `3`
   where appropriate.
4. **Temp→psal propagation**: controlled purely by the configuration —
   `temp_to_psal` is an ordinary item enabled by listing it in the
   `qc_item_set`. No platform-metadata conditioning.
