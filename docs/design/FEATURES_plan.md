# Signal and context feature groups: implementation plan

The four phases below each land on their own `feature/*` branch off `develop`,
green and usable on its own. The contract for every class is in
[`FEATURES_spec.md`](FEATURES_spec.md).

Ordering is physics first: the shared machinery and the quantities that other
groups depend on come before the groups that consume them.

## Phase 1: shared machinery, `derived_values`, `stratification`

Branch `feature/signal-utils-and-physics`.

1. `common/utils/profile_signal.py`: Savitzky-Golay coefficients and FIR
   application, neighbour differences, the shared RTQC9 spike stencil, centred
   rolling statistics, rolling and per-profile MAD and robust z.
2. `common/utils/seawater.py`: `gravity`, `depth_from_pressure`,
   `brunt_vaisala_squared`, in the module's existing domain-masked style.
3. `prepare/features/profile_feature_base.py`: the shared base for every new
   class, mirroring `QCItemFeatureBase`. Resolves `params` and `outputs`,
   computes over the sorted profile in `filtered_input`, joins onto the
   target's selected rows by the observation keys, and applies `scale_second`.
4. `prepare/features/derived_values.py` and
   `prepare/features/stratification.py`.
5. Schema: add `outputs` to `feature_param_sets[].params[]` in the dataset and
   classification schemas.
6. Registry and `tests/test_feature_levels.py`.

## Phase 2: point-level group

Branch `feature/point-level-features`.

`profile_smooth` (smoothed value, first and second derivative, residual, robust
z of the residual, curvature-to-residual ratio, spike index, and the window
fractions that depend on its own intermediates), `neighbor_diff` (differences at
configurable lags in both directions, plus the large-difference fraction), and
`regime_flags` (mixed layer, thermocline, halocline, normalized depth to the
peak gradient).

The window fractions live with the class that computes the quantity they count,
because `load_feature_class` instantiates every feature independently and there
is no way for one entry to read another's columns.

## Phase 3: window-level group and profile MAD

Branch `feature/window-level-features`.

`rolling_stats` for the plain window statistics and the local robust z, and
`mad` support in `profile_summary_stats` computed from `filtered_input` (see
the spec for why step 2 is left alone).

## Phase 4: `geo_context`, docs, changelog

Branch `feature/geo-context-and-docs`.

`geo_context` reading `bathymetry` and `coast_distance` as input columns, then
the user-facing documentation: one page per group under `docs/source/features/`,
a how-to with a worked configuration, the feature tables in the preparation and
classification configuration pages, and a note in the profile-level how-to that
all the new classes are observation level and so need `agg` there.

Templates in `common/config/yaml_templates.py` stay unchanged, so every existing
configuration keeps working and the new features are opt-in.

## Verification

Targeted test files per phase rather than the whole suite, plus `ruff`, a
zero-warning Sphinx build, and the repository dash check. Once per phase, an end
to end `create_training_dataset` and `train_and_evaluate` run with the phase's
features enabled, and the same configuration through the profile-level pipeline
with `agg` to prove the aggregation path.

## Release

The changelog entries are all `### Added`, so the release that carries this work
is a minor bump.
