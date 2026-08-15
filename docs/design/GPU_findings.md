# GPU acceleration: measurements and decisions

A durable record of what was measured when `aiqclib` was run on GPUs, what was
decided as a result, and what is still undecided. Measured on a shared server
with 2 × Tesla P100 (compute capability 6.0) between 2026-08-12 and 2026-08-15.

**What is documented elsewhere.** User-facing guidance lives in
`docs/source/how-to/gpu_acceleration.rst` (configuration, the device ceiling,
what is accelerated) and `docs/source/how-to/shap_values.rst` (what SHAP costs).
Machine-specific operational knowledge — rootless Docker, CDI, container
networking — is not `aiqclib`'s concern and lives in a separate repository. This
file holds what is specific to *this library*: the numbers, their conditions,
and the reasoning behind decisions that the public pages only state.

---

## 1. The headline measurement

`train_and_evaluate` over one region of production CTD data, same machine, same
data, `device` the only configured difference. CPU side pinned to `n_jobs: 20`.

| Step | GPU | CPU | Speedup |
|---|---|---|---|
| 1 Reading training sets | 11.1 s | 3.3 s | — |
| 2 Cross-validation (fitting only) | 2509 s | 2595 s | 1.03× |
| 3 Validation reports and plots | 62 s | 61 s | 1.00× |
| 4 Build, test and **SHAP** | **7604 s** | **17716 s** | **2.33×** |
| 5 Final model fit and write | 699 s | 684 s | 0.98× |
| **Whole phase** | **10885 s** | **21059 s** | **1.93×** |

**Fitting gained nothing.** Step 2 is fitting alone — cross-validation forces
`enable_shap = False` (`train/step2_validate_model/validate_base.py:108`) — and
step 5 is a single full-data fit. Both came out level with the CPU. The data is
not large enough for the GPU to repay the transfer on `fit`, which matches VRAM
sitting nearly unused during the run.

**Steps 3 and 5 are the internal control.** Step 3 touches no model at all.
Landing within 1–2% across the two runs is what makes the 2.33× on step 4
credible rather than an artefact of machine load.

**Conditions that bound the ratio.** The CPU side had 20 threads and CPU
TreeSHAP scales with them, so a different `n_jobs` moves the 2.33×. An earlier
"11269 s CPU" figure came from a different machine with `n_jobs: -1` and is not
comparable — it briefly suggested the GPU was worthless.

## 2. Where the time actually goes

Per-call profiling (wrapping `SklearnModelBase.build` / `test` /
`calculate_shap`), all on GPU:

| | large dataset (train) | small dataset (train) | small dataset (classify) |
|---|---|---|---|
| fit | 3636.8 s (33.9%) | 362.5 s (51.3%) | 0.0 s — loads models |
| test excl. SHAP | 131.1 s (1.2%) | 12.5 s (1.8%) | 0.3 s (0.6%) |
| **SHAP** | **6961.1 s (64.9%)** | **331.1 s (47.0%)** | **47.7 s (99.4%)** |

Two things follow.

**SHAP's share grows with data size.** Between the two datasets, fitting scaled
10.0× and SHAP scaled 21.0×. A small dataset therefore *understates* how much
SHAP matters — see §6.

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

**There is no optimisation available inside `aiqclib`.** A plausible theory —
that `common/base/scikit_learn_model_base.py:228` building `background_data`
from the whole training set, which the tree branch at `:236` never uses, was a
meaningful cost — was **wrong**: 0.5 s total. It remains a real defect worth
cleaning up, but as hygiene, not performance. Its speed also implies the
conversion is near zero-copy, so it is probably not a memory problem either.

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
`booster.predict(..., pred_contribs=True)` on the original booster — still
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
| XGBoost `device: cuda` | 0.1 s | 0.2 s | **0.69× — slower** |

Not adopted, for two reasons. It is *slower* for XGBoost, which already reaches
the GPU, so it helps only algorithms that had no GPU path — and production is
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
and `GaussianNaiveBayes` have any GPU route on this hardware — scikit-learn's
Array API dispatch — and they are the cheapest algorithms in the suite. The four
worth accelerating (RandomForest, DecisionTree, SVM, KNN) are exactly the
unreachable ones.

**This changes if the hardware changes.** On Volta or newer, cuML would put
those four in reach and this decision should be revisited.

### Taken: warn about the SHAP cost

`common/utils/diagnostics.py:warn_shap_cost`, called from `calculate_shap`,
fires once per run above 100,000 rows and names `calculate_shap: false`. Once
per run rather than per target because the message concerns the setting;
`warnings`' own de-duplication does not cover it, since the row count makes each
message textually distinct.

The threshold is a heuristic chosen without data on typical row counts — real
cost is rows × trees × depth². It is a named constant,
`SHAP_ROW_WARNING_THRESHOLD`.

## 6. Test datasets: fixture versus reference

- **Fixture — `bo_bo`** (~50 MB parquet): a train phase takes ~12 minutes
  instead of ~3 hours. Use for correctness and configuration iteration. Run
  `MODE=prepare` once before any training test.
- **Reference — `ar_ar`** (>600 MB): the **only** dataset any timing claim may
  be made on.

The distinction is not pedantry. Every GPU result here is size-dependent in the
direction that punishes a small file — fitting was already only 1.03× at 600 MB
— so a `bo_bo` run could show the GPU losing, which would be true of `bo_bo` and
false of the pipeline. The profile *shape* differs too: SHAP is 47% of the small
dataset and 65% of the large one.

**Naming hazard:** `0002` is overloaded. It was the set name of the CPU baseline
(`training_ar_ar_0002`) before it was the `bo_bo` dataset list. Say which is
meant.

## 7. Configuration specifics

Three that cost time to rediscover:

- **`device` must be inside `model_params`.** Keys directly under `model` are
  step parameters (like `calculate_shap`) and never reach the algorithm — a
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
SHAP is 47–65% of a training phase and 99.4% of classification, its cost is
irreducible (§2), and the GPU already gives it ~2.3×.

- If the values inform QC decisions or model interpretation, that share is
  simply their price, and the GPU is what makes it tolerable.
- If they are written and rarely read, `calculate_shap: false` is worth roughly
  2× on train and ~14× or more on classify — **a bigger lever than the GPU**,
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
`pyproject.toml` was considered and rejected — it would penalise every user with
newer hardware to accommodate two specific cards.

The profiler that produced §2 lives with that tooling. It wraps
`SklearnModelBase.build` / `test` / `calculate_shap` on the class, so it needs
no library change; note that `BuildModelSuite` deep-copies each method object
per target, which is why patching must be done on the class rather than on
instances.
