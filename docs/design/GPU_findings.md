# GPU acceleration: measurements and decisions

A durable record of what was measured when `aiqclib` was run on GPUs, what was
decided as a result, and what is still undecided. Measured on a shared server
with 2 × Tesla P100 (compute capability 6.0) between 2026-08-12 and 2026-08-15.

**What is documented elsewhere.** User-facing guidance lives in
`docs/source/how-to/gpu_acceleration.rst` (configuration, the device ceiling,
what is accelerated) and `docs/source/how-to/shap_values.rst` (what SHAP costs).
Machine-specific operational knowledge (rootless Docker, CDI, container
networking) is not `aiqclib`'s concern and lives in a separate repository. This
file holds what is specific to *this library*: the numbers, their conditions,
and the reasoning behind decisions that the public pages only state.

---

## 1. The headline measurement

`train_and_evaluate` over one region of production CTD data, same machine, same
data, `device` the only configured difference. CPU side pinned to `n_jobs: 20`.

| Step | GPU | CPU | Speedup |
|---|---|---|---|
| 1 Reading training sets | 11.1 s | 3.3 s | n/a |
| 2 Cross-validation (fitting only) | 2509 s | 2595 s | 1.03× |
| 3 Validation reports and plots | 62 s | 61 s | 1.00× |
| 4 Build, test and **SHAP** | **7604 s** | **17716 s** | **2.33×** |
| 5 Final model fit and write | 699 s | 684 s | 0.98× |
| **Whole phase** | **10885 s** | **21059 s** | **1.93×** |

**Fitting gained nothing.** Step 2 is fitting alone, since cross-validation forces
`enable_shap = False` (`train/step2_validate_model/validate_base.py:108`), and
step 5 is a single full-data fit. Both came out level with the CPU. The data is
not large enough for the GPU to repay the transfer on `fit`, which matches VRAM
sitting nearly unused during the run.

**Steps 3 and 5 are the internal control.** Step 3 touches no model at all.
Landing within 1-2% across the two runs is what makes the 2.33× on step 4
credible rather than an artefact of machine load.

**Conditions that bound the ratio.** The CPU side had 20 threads and CPU
TreeSHAP scales with them, so a different `n_jobs` moves the 2.33×. An earlier
"11269 s CPU" figure came from a different machine with `n_jobs: -1` and is not
comparable; it briefly suggested the GPU was worthless.

## 2. Where the time actually goes

Per-call profiling (wrapping `SklearnModelBase.build` / `test` /
`calculate_shap`), all on GPU:

| | large dataset (train) | small dataset (train) | small dataset (classify) |
|---|---|---|---|
| fit | 3636.8 s (33.9%) | 362.5 s (51.3%) | 0.0 s, loads models |
| test excl. SHAP | 131.1 s (1.2%) | 12.5 s (1.8%) | 0.3 s (0.6%) |
| **SHAP** | **6961.1 s (64.9%)** | **331.1 s (47.0%)** | **47.7 s (99.4%)** |

Two things follow.

**SHAP's share grows with data size.** Between the two datasets, fitting scaled
10.0× and SHAP scaled 21.0×. A small dataset therefore *understates* how much
SHAP matters; see §6.

**Classification is almost entirely SHAP**, because it does no fitting: it loads
a model, predicts, and explains. This also means classify's GPU speedup *is*
SHAP's speedup, ~2.3×, which is why a head-to-head classify benchmark was
closed as unnecessary rather than run.

### The cost is the algorithm, not the plumbing

Splitting `calculate_shap` three ways:

| | seconds | share |
|---|---|---|
| `explainer.shap_values` | 329.4 | **99.5%** |
| polars → pandas | 0.5 | 0.2% |
| output assembly | 1.3 | 0.4% |

**There is no optimisation available inside `aiqclib`.** A plausible theory,
that building `background_data` from the whole training set, which the tree
branch never uses, was a meaningful cost, was **wrong**: 0.5 s total. Its
speed also implies the conversion is near zero-copy, so it was probably not
the memory spike it was blamed for either.

It was a genuine defect nonetheless, and is fixed: `background_data` is now a
local function called only by the two branches that need one
(`common/base/scikit_learn_model_base.py:226`), so the tree path never builds
it. Expect no measurable speedup; that is the point of recording it here,
since the obvious-looking waste was not where the time went.

## 3. Reading the verbose log

`Progress.step()` prints **elapsed-at-start**, not the step's duration
(`common/utils/progress.py:117`). Every timing read from a verbose log is
therefore the *difference between consecutive lines*.

Taken literally, a classify log claims 3419.5 s for "Merging predictions with
input data". The merge takes **1.6 s**; the preceding step takes 3419.5 s. The
same trap reproduced on the small dataset: an apparent 51.1 s merge was 0.1 s,
with 49.1 s in "Classifying observations".

This misled an entire investigation once and is the first thing to check when a
step looks absurdly expensive.

## 4. Why SHAP runs on the GPU

Non-obvious, and documented wrongly in the public how-to for two days.

`calculate_shap` constructs `shap.TreeExplainer(self.model)` with **no
background data** (`common/base/scikit_learn_model_base.py:236-238`). With
`data is None`, shap resolves `feature_perturbation` to `tree_path_dependent`,
which enables the XGBoost fast path: values come from
`booster.predict(..., pred_contribs=True)` on the original booster, still
carrying `device: cuda`. **TreeSHAP therefore runs on the GPU.**

The warning that caused the misreading:

```
WARNING: Falling back to prediction using DMatrix due to mismatched devices.
```

"Falling back to `DMatrix`" is not "falling back to the CPU". The pandas input
lives in host memory, so XGBoost wraps it in a `DMatrix`, copies it to the
device, and predicts there. The warning announces the copy.

Confirmed independently: `TreeExplainer` with `device: cpu` versus `cuda` came
out **3.35×** in a separate synthetic run.

## 5. Decisions taken

### Rejected: `shap.GPUTreeExplainer`

Measured (100k × 30, 100 trees, depth 6, explaining 20k rows):

| Model | CPU `TreeExplainer` | `GPUTreeExplainer` | Ratio |
|---|---|---|---|
| RandomForest | 15.2 s | 0.5 s | **28.4×** |
| XGBoost `device: cuda` | 0.1 s | 0.2 s | **0.69×, slower** |

Not adopted, for two reasons. It is *slower* for XGBoost, which already reaches
the GPU, so it helps only algorithms that had no GPU path, and production is
XGBoost, with RandomForest occasional and exploratory. And its `_cext_gpu`
extension ships in no `shap` wheel, so using it means building `shap` against a
CUDA toolkit, which the deployment otherwise avoids entirely.

Documented for users in `gpu_acceleration.rst` under *SHAP for the Other Tree
Models*.

### Rejected: cuML for the other algorithms

RAPIDS removed Pascal support in 24.02 and requires compute capability 7.0+, so
cuML cannot run on the target hardware at all. Pinning back to 23.12 fails on
Python grounds (`requires-python >=3.12`).

Of the nine algorithms, only `LogisticRegression`, `LinearDiscriminantAnalysis`
and `GaussianNaiveBayes` have any GPU route on this hardware (scikit-learn's
Array API dispatch), and they are the cheapest algorithms in the suite. The four
worth accelerating (RandomForest, DecisionTree, SVM, KNN) are exactly the
unreachable ones.

**This changes if the hardware changes.** On Volta or newer, cuML would put
those four in reach and this decision should be revisited.

### Taken: report the SHAP cost

`common/utils/diagnostics.py:report_shap_cost`, called from `calculate_shap`,
fires once per run above 100,000 rows and names `calculate_shap: false`. Once
per run rather than per target because the message concerns the setting; a
module-level flag rather than `warnings`' own de-duplication, which does not
cover it, since the row count makes each message textually distinct.

It prints a `[aiqclib] note:` line through `progress.notice` rather than
raising a `UserWarning`. It was a warning first, and in practice that put a
paragraph of prose under a `site-packages/.../scikit_learn_model_base.py:130:
UserWarning:` header, which reads as a fault in the library rather than as the
cost of a setting the user chose. Nothing is wrong when it fires, so nothing
about it should look like it is.

The threshold is a heuristic chosen without data on typical row counts; the real
cost is rows × trees × depth². It is a named constant,
`SHAP_ROW_WARNING_THRESHOLD`.

### Taken: name the file in XGBoost's pickle-version warning

`common/utils/diagnostics.py:clarify_model_load_warnings`, wrapping the load in
`ModelBase.load_model`. XGBoost's own warning arrives as a `UserWarning`
attributed to a line in `pickle.py`, naming no file, which in a `run_batch` over
several datasets leaves nothing to act on.

**Measured, since the warning's own wording is misleading.** It says "generated
by an older version of XGBoost", which reads as a claim about the file. A save
and load matrix over the six releases in the uv cache (2.1.4, 3.0.2, 3.0.5,
3.1.1, 3.2.0, 3.4.0; 36 pairs, one small `XGBClassifier`) shows otherwise:

- **Every** mismatched pair warns, including patch-only gaps (3.0.2 vs 3.0.5),
  and it warns in both directions. Matching pairs never warn.
- Predictions from a model loaded into a **newer** XGBoost matched the training
  version exactly, in every such pair.
- Predictions from a model loaded into an **older** XGBoost did **not**: a
  3.2.0-trained model scored 0.240963 under 3.1.1 and newer, and 0.195341 under
  3.0.5 and older. No error, no failure, a ~19% shift in the score. The same
  split appeared for a 3.1.1-trained and a 3.4.0-trained model, so the boundary
  is the format change between 3.0.x and 3.1.x rather than anything about one
  model.

Repeated on a realistic model (200 trees, depth 6, 20 features, 2000 scored
rows), since a 3-tree toy could plausibly hide or exaggerate the effect:

| Model trained under | Used under | Scores | SHAP |
| --- | --- | --- | --- |
| 3.0.2 | 3.4.0 | bitwise identical, 0 label flips | max diff 1.9e-06 (float32 rounding) |
| 3.4.0 | 3.0.2 | max diff 0.076, 21/2000 labels flipped at 0.5 | not compared |

So the message must not repeat "older", and the direction is the part worth
telling the user: same version or newer is safe, older is not. Note that the
`SM 60` pin recommended for a P100 puts the *training* machine on the older
release, which is the safe direction, but only while the pin stays off the
classification machines.

### Taken: stamp the writing version into the model file

`common/utils/model_version.py`. Deciding the severity by direction needs the
version that wrote the file, and that is **not** recoverable from the loaded
model: `Booster.save_config()` reports the runtime version, not the file's. It
does sit in the pickle, as a UBJSON `version` triple inside the serialized
booster buffer, but only reachable by byte-scraping, which was rejected as too
fragile.

So `save_model` sets `_aiqclib_xgboost_version` on the estimator immediately
before `joblib.dump`, and it is pickled along with everything else.
Alternatives considered and rejected: wrapping the dump in a dict
(`{"model": ..., "version": ...}`) changes the file format, breaking anything
that loads these files with plain `joblib.load`, including users; a sidecar
file can be separated from the model it describes. An extra attribute leaves
the file a plain pickled estimator. Only XGBoost models are stamped, since they
are the only ones whose loader raises the warning this answers.

Three outcomes on load:

| Stamp | Direction | Reported as |
| --- | --- | --- |
| present | this environment same or newer | `[aiqclib] note:`, no action |
| present | this environment older | `UserWarning` naming both versions |
| absent (file predates this) | unknown | `UserWarning` saying so |

Each once per run, for the same reason as the SHAP notice: the cause is the
environment, not any one file. The three severities are tracked separately, so
a run loading a safe model and a risky one still hears about the risky one.

Verified end to end against real cross-version pickles rather than mocks, by
putting the 3.0.2 and 3.4.0 wheels from the `uv` cache on `PYTHONPATH` in turn:
3.0.2 to 3.4.0 produced the note and no warning, 3.4.0 to 3.0.2 produced the
warning, and an unstamped file produced the fallback.

Version strings are compared as integer tuples, not as text: XGBoost will reach
3.10, where `"3.10.0" < "3.9.0"` is true and would report a risky load as safe.
Anything that does not parse as a plain dotted number (a `.dev` build) falls
back to the unknown case rather than guessing.

## 6. Test datasets: fixture versus reference

- **Fixture, `bo_bo`** (~50 MB parquet): a train phase takes ~12 minutes
  instead of ~3 hours. Use for correctness and configuration iteration. Run
  `MODE=prepare` once before any training test.
- **Reference, `ar_ar`** (>600 MB): the **only** dataset any timing claim may
  be made on.

The distinction is not pedantry. Every GPU result here is size-dependent in the
direction that punishes a small file (fitting was already only 1.03× at 600 MB),
so a `bo_bo` run could show the GPU losing, which would be true of `bo_bo` and
false of the pipeline. The profile *shape* differs too: SHAP is 47% of the small
dataset and 65% of the large one.

**Naming hazard:** `0002` is overloaded. It was the set name of the CPU baseline
(`training_ar_ar_0002`) before it was the `bo_bo` dataset list. Say which is
meant.

## 7. Configuration specifics

Three that cost time to rediscover:

- **`device` must be inside `model_params`.** Keys directly under `model` are
  step parameters (like `calculate_shap`) and never reach the algorithm; a
  hyperparameter put there is silently ignored.
- **With `ModelSuite`, name XGBoost explicitly.** The scikit-learn algorithms
  reject `device` with `TypeError: unexpected keyword argument 'device'`:
  `model_params: { XGB: { device: cuda, tree_method: hist } }`.
- **The classify `model` step needs `use_dataset_folder=False`** to find the
  models that train wrote, because its `base_path` carries the dataset folder
  itself. A pre-flight check that flags a missing classify `model` `base_path`
  before the first training run is expected, not an error.

## 8. Open question

**Do the SHAP values get used downstream?** Everything measurable is settled:
SHAP is 47-65% of a training phase and 99.4% of classification, its cost is
irreducible (§2), and the GPU already gives it ~2.3×.

- If the values inform QC decisions or model interpretation, that share is
  simply their price, and the GPU is what makes it tolerable.
- If they are written and rarely read, `calculate_shap: false` is worth roughly
  2× on train and ~14× or more on classify, **a bigger lever than the GPU**,
  and it would remove most of the reason to use one.

No profile can answer this. It needs someone who knows what the outputs are for.

Note that `calculate_shap` already defaults to `False`
(`common/base/scikit_learn_model_base.py:55`), so this is a question about
project configurations, not about the library's default.

## 9. Deployment tooling

The image and run scripts used for these measurements are kept **outside this
repository** because they are site-specific: they pin `xgboost<3.3` for the
Pascal cards, bind-mount a project share at its own path, and carry flags
specific to one server's Docker installation. Capping `xgboost` in
`pyproject.toml` was considered and rejected; it would penalise every user with
newer hardware to accommodate two specific cards.

The profiler that produced §2 lives with that tooling. It wraps
`SklearnModelBase.build` / `test` / `calculate_shap` on the class, so it needs
no library change; note that `BuildModelSuite` deep-copies each method object
per target, which is why patching must be done on the class rather than on
instances.
