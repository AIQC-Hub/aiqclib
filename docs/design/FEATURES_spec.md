# Signal and context feature groups: specification

Status: in progress (phase 1 of 4).

This document specifies the feature groups added to `aiqclib` from the
`features.pptx` proposal. It is the contract: what each feature class emits,
what every parameter means, what its default is, and how missing or
unusable inputs are handled.

## Source

The proposal is a single table with three columns (feature group, purpose,
example), transcribed verbatim below so the implementation can be checked
against it without opening the deck.

| Feature group | Purpose | Example |
| --- | --- | --- |
| Spatiotemporal | Geographic and regime context | Time, Position, Depth, Bathymetry (GEBCO), Coast distance (GSHHG), Normalized depth (depth/seafloor depth), Distance to bottom, Deep-stable-layer flag |
| Measured variables | Raw oceanographic signal | Temperature, Salinity, Density |
| Stratification | Physical stability constraints | Vertical density stability: N squared value, magnitude, unstable layer flag |
| Point-level | Local anomaly detection | Savitzky-Golay smoothed value, 1st derivative (gradient), 2nd derivative (curvature); regime flags (in halocline/thermocline, in mixed layer, normalized depth to peak gradient); neighbor differences at lags 1-5 (forward and backward); curvature-to-residual ratio; spike index (Argo-style); robust z-score of SG residual; residual from smoothed profile |
| Window-level (windows: 5, 11, 21, 41 points) | Multi-scale anomaly context | Mean, Median, MAD, Min, Max; local robust z-score; % of points flagged as outliers in residual; % of large neighbor differences; % of high-curvature points |
| Profile-level | Global profile baseline | Mean, Median, MAD |

The table names quantities, not numbers. Window sizes are the one exception,
and even those are given as an example. Every other threshold is a choice, so
the implementation makes each one a configuration parameter with a documented
default rather than a constant in the code.

## What already existed

| Proposal item | Covered by |
| --- | --- |
| Time | `day_of_year` |
| Position | `location` |
| Depth (as pressure) | `basic_values` on `pres` |
| Temperature, Salinity | `basic_values` |
| Neighbouring values at lags 1-5 | `flank_up` / `flank_down` (values, not differences) |
| Profile mean, median | `profile_summary_stats` |

Everything else is new.

## Design rules

1. **One feature class per group**, registered in `FEATURE_REGISTRY` under a
   short name, configured by one entry of `feature_param_sets[].params[]`.
2. **`outputs` selects the columns.** Every new class declares
   `supported_outputs`; the `outputs` key of the configuration entry picks a
   subset, and omitting it means all of them. A group can therefore be adopted
   one column at a time.
3. **Tunables live under `params`.** The schema already allows a free-form
   `params` object on a feature entry (it was added for the `qc_*` items), so
   the only schema change is the new `outputs` array.
4. **No new runtime dependencies.** Savitzky-Golay is a fixed-coefficient FIR
   filter, so its coefficients come from a cached numpy least-squares solve and
   the filter itself is a sum of shifted columns evaluated by polars.
5. **Profile awareness belongs in polars, physics belongs in numpy.** Pointwise
   equations of state stay in `common/utils/seawater.py` and remain
   elementwise; anything that looks at a neighbouring level is a polars
   expression over `PROFILE_KEYS` in `common/utils/profile_signal.py`.
6. **Missing inputs propagate as null**, never as a fabricated number. A value
   outside the domain of the equation of state is missing (the rule
   `seawater.py` already follows). A configured input column that is absent
   from the data is an error, matching `qc_position_on_land`, unless the entry
   sets `required: false`.
7. **Profile edges are null.** A centred window of width `w` needs `(w-1)/2`
   levels on each side. Rather than reflect, extend or shrink the window, the
   output is null there, which the models handle as missing and which cannot be
   mistaken for a measured value.

## Column naming

| Shape | Pattern | Example |
| --- | --- | --- |
| One column per variable | `{variable}_{output}` | `temp_sigma0`, `temp_residual` |
| Per variable and window | `{variable}_w{window}_{output}` | `temp_w11_mad` |
| Per variable, lag and direction | `{variable}_diff_{direction}_{lag}` | `temp_diff_up_3` |
| Not variable specific | `{output}` | `depth`, `distance_to_bottom` |

Two entries of the same feature class in one `feature_param_set` are legal and
useful (for example two smoothing windows). Because the window appears in the
column name, their outputs do not collide, which matters: the extract step
merges per-feature frames with `pl.concat(..., how="align_left")`.

## Shared utilities

### `common/utils/profile_signal.py`

All expression builders partition by `PROFILE_KEYS` and assume the frame is
sorted by `OBSERVATION_KEYS` (use `sort_profiles`).

| Function | What it gives |
| --- | --- |
| `savgol_coefficients(window, polyorder, deriv)` | Cached FIR coefficients for the centred Savitzky-Golay filter. |
| `fir_expr(column, coefficients)` | The filter applied as a sum of shifted columns. |
| `savgol_expr(column, window, polyorder, deriv)` | The two combined. Derivatives are per sample index; the feature classes convert to per-dbar with the local spacing. |
| `neighbor_diff_expr(column, lag, direction)` | `x[i] - x[i-lag]` (up) or `x[i] - x[i+lag]` (down). |
| `spike_index_expr(v1, v2, v3)` | The RTQC9 stencil value, shared with `QCSpike`. |
| `rolling_stat_expr(column, window, stat)` | Centred `mean`, `median`, `min`, `max`, `std`, `sum`. |
| `fraction_expr(condition, window)` | Centred rolling mean of a boolean, that is the fraction of points in the window satisfying it. |
| `with_rolling_mad`, `with_rolling_robust_z`, `with_profile_mad`, `with_profile_robust_z` | Two-stage statistics that polars cannot express as one window function. |

Robust z uses the conventional consistency factor 1.4826, and is null where the
deviation is zero, because a constant window gives no scale to compare against.

### `common/utils/seawater.py` additions

Elementwise, domain-masked, in the style the module already uses.

| Function | Notes |
| --- | --- |
| `gravity(lat)` | UNESCO 1983 formula. `gravity(0)` = 9.780318, `gravity(90)` = 9.832177 m/s squared. |
| `depth_from_pressure(p, lat)` | UNESCO 1983. Check value: `depth_from_pressure(10000, 30)` = 9712.653 m. |
| `brunt_vaisala_squared(sigma_theta, d_sigma_d_depth, lat)` | `g / (1000 + sigma_theta) * d_sigma_d_depth`, with depth positive downward so that a stable water column gives a positive result. The gradient is supplied by the caller, because it is a property of the profile rather than of the parcel. |

## Feature classes

### `derived_values` (phase 1, observation level)

The measured-variables row of the table, for the quantities that are computed
rather than measured.

| Output | Meaning |
| --- | --- |
| `sigma0` | Potential density anomaly, from `seawater.sigma0`. |
| `depth` | Depth in metres from pressure and latitude. |
| `potential_temperature` | Referenced to the surface. |

| Parameter | Default | Meaning |
| --- | --- | --- |
| `salinity_column` | `psal` | Input column holding practical salinity. |
| `temperature_column` | `temp` | Input column holding in-situ temperature. |
| `pressure_column` | `pres` | Input column holding pressure in decibars. |
| `latitude_column` | `latitude` | Input column holding latitude. |
| `prefer_input_columns` | `true` | When the input already carries a column of the output's name, use it instead of recomputing. |

### `stratification` (phase 1, observation level)

| Output | Meaning |
| --- | --- |
| `sigma0_gradient` | Centred difference of sigma-0 with respect to depth, kg/m^4. |
| `n2` | Brunt-Vaisala frequency squared, 1/s squared. |
| `n2_abs` | Its magnitude, which is what the table calls the value's "magnitude". |
| `unstable_flag` | 1 where `n2 < unstable_n2`, else 0; null where `n2` is null. |

| Parameter | Default | Meaning |
| --- | --- | --- |
| `salinity_column`, `temperature_column`, `pressure_column`, `latitude_column` | as above | |
| `sigma0_column` | none | Reuse a sigma-0 column already in the input instead of recomputing. |
| `unstable_n2` | `0.0` | The threshold below which a level counts as unstable. |
| `min_depth_separation` | `0.01` | Metres. A centred difference over a smaller separation is null rather than a division by nearly zero. |

### `profile_smooth` (phase 2, observation level)

Per variable in `col_names`.

| Output | Column | Meaning |
| --- | --- | --- |
| `smooth` | `{v}_smooth` | The Savitzky-Golay fit at the level. |
| `d1`, `d2` | `{v}_d1`, `{v}_d2` | Slope and curvature of the fit, per unit of `spacing_column`. |
| `residual` | `{v}_residual` | The measurement less the fit. |
| `robust_z` | `{v}_robust_z` | The residual standardised against the profile's own median and MAD. |
| `curvature_ratio` | `{v}_curvature_ratio` | `abs(d2) / abs(residual)`: high for a real sharp feature, low for a spike. |
| `spike_index` | `{v}_spike_index` | The RTQC9 stencil value, unthresholded. |
| `outlier_frac` | `{v}_w{n}_outlier_frac` | Fraction of the window whose `robust_z` exceeds `outlier_z`. |
| `high_curvature_frac` | `{v}_w{n}_high_curvature_frac` | The same for the robust score of the curvature. |

| Parameter | Default | Meaning |
| --- | --- | --- |
| `window` | `11` | Points in the smoothing window (odd). |
| `polyorder` | `2` | Degree of the fitted polynomial. |
| `spacing_column` | `pres` | Derivatives are per unit of this column; `null` keeps them per level. |
| `min_spacing` | `1e-6` | Below this step the derivative conversion is null. |
| `windows` | `[5, 11, 21, 41]` | Windows for the fraction outputs. |
| `outlier_z`, `high_curvature_z` | `3.0` | What counts as an outlier or high curvature. |
| `ratio_floor` | `1e-9` | Residual magnitude below which `curvature_ratio` is null. |

### `neighbor_diff` (phase 2, observation level)

| Output | Column | Meaning |
| --- | --- | --- |
| `diff` | `{v}_diff_{direction}_{lag}` | The level less its neighbour that many levels up or down. |
| `large_diff_frac` | `{v}_w{n}_large_diff_frac` | Fraction of the window whose reference difference is large. |

| Parameter | Default | Meaning |
| --- | --- | --- |
| `lags` | `[1, 2, 3, 4, 5]` | Neighbour distances. |
| `directions` | `["up", "down"]` | Shallower, deeper, or both. |
| `windows` | `[5, 11, 21, 41]` | Windows for the fraction. |
| `reference_lag`, `reference_direction` | `1`, `up` | The difference the fraction counts. |
| `large_diff_z` | `3.0` | Robust score above which a difference is large. |
| `large_diff_threshold` | none | An absolute magnitude instead, as one number or a per-variable mapping. |

### `regime_flags` (phase 2, observation level)

| Output | Column | Meaning |
| --- | --- | --- |
| `in_mixed_layer` | `in_mixed_layer` | 1 above the mixed layer depth. Not per variable. |
| `in_gradient_layer` | `{v}_in_gradient_layer` | 1 in the steepest part of the variable's gradient: the thermocline for temperature, the halocline for salinity. |
| `normalized_depth_to_peak_gradient` | `{v}_normalized_depth_to_peak_gradient` | Signed position relative to the steepest gradient, divided by the profile's pressure range. |

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mixed_layer_criterion` | `density` | Or `temperature`. |
| `density_threshold` | `0.03` | kg/m³ departure that ends the mixed layer. |
| `temperature_threshold` | `0.2` | °C departure, for the temperature criterion. |
| `reference_pressure` | `10.0` | dbar the criterion measures from, below the diurnal surface layer. |
| `gradient_percentile` | `0.9` | Quantile of gradient magnitude that counts as the steep part. |
| `min_pressure_separation` | `0.01` | Guards the gradient division. |

Two rules here are worth stating because they are judgement calls rather than
consequences of the arithmetic. A profile that never crosses the mixed layer
criterion is mixed all the way down, so every level gets 1 rather than null:
that is an answer, not a missing value. And `in_gradient_layer` additionally
requires a non-zero gradient, because a profile that is uniform over most of
its length has a zero percentile, and "at or above zero" would flag the flat
part as the steepest part of the water column.

### `rolling_stats` (phase 3, observation level)

Every output is produced per variable and per window, named
`{v}_w{n}_{output}`: `mean`, `median`, `mad`, `min`, `max`, `std` and
`robust_z` (the level measured against its own window's median and MAD).

| Parameter | Default | Meaning |
| --- | --- | --- |
| `windows` | `[5, 11, 21, 41]` | Window sizes, odd. |
| `min_samples` | none | Non-null values a window needs; the default is the full window, which makes profile edges null. |

### Profile MAD (phase 3)

`profile_summary_stats` accepts `mad` in its `summary_stats_names`
alongside the statistics the step 2 table carries. A name the table does not
have is computed per profile from the input instead of raising, which is how
`mad` is served without widening the table; an unknown name still raises,
naming both sets.

This is the one deliberate asymmetry in the feature set: a statistic computed
this way has no row in the summary table, so it cannot be normalised with
`auto_min_max` or `standard`.

### Later phases

`geo_context` (phase 4). Its parameter table is added to this document when
the phase lands.

## Deliberate omissions

- **No GEBCO or GSHHG reader.** `geo_context` reads `bathymetry` and
  `coast_distance` as input columns, exactly as `qc_position_on_land` reads the
  sea floor depth. Sampling the grids stays an upstream job, which keeps netCDF
  and shapefile libraries, and their data files, out of the library and its CI.
- **Profile MAD is not added to step 2.** The summary statistics table is a
  persisted artefact whose fixtures ship in a GitHub Release archive, so
  widening its schema means regenerating and re-releasing them. The statistic
  is computed in the feature class instead. Adding it to step 2 remains a
  reasonable follow-up, to be done together with a fixture release.
- **No automatic normalization for derived columns.** Data-derived
  normalization (`auto_min_max`, `standard`) is fitted from the step 2 summary
  table, which has rows only for raw input variables. The new columns therefore
  support `raw` (the default, and the right answer for z-scores, fractions and
  flags) and manual `min_max`.
