================
GPU Acceleration
================

``aiqclib`` can train **XGBoost** models on an NVIDIA GPU. This requires no
code change and no extra dependency — only a configuration setting.

What Can Use a GPU
------------------

Only XGBoost — but that covers more than fitting, because everything XGBoost
does for a model runs on whichever device that model was given:

============================== ==================================================
Stage / component              Device
============================== ==================================================
Dataset preparation            CPU (polars / pandas)
**XGBoost** training           **GPU**, when ``device: cuda`` is set
**XGBoost** prediction         **GPU** — see :ref:`gpu-prediction-device`
**XGBoost** SHAP values        **GPU** — see :ref:`gpu-prediction-device`
The other eight algorithms     CPU — scikit-learn has no GPU backend
SHAP for those algorithms      CPU — but see :ref:`gpu-shap-other-trees`
============================== ==================================================

So a GPU accelerates one algorithm, across every stage that uses it. If your
pipeline is dominated by dataset preparation, or you train with ``ModelSuite``
across all nine algorithms, the overall speedup will be proportionally
smaller.

Which stage benefits most is not obvious in advance, and is often not the one
you set ``device: cuda`` for — see :ref:`gpu-worth-it`.

Requirements
------------

* An NVIDIA GPU and a working driver (check with ``nvidia-smi``).
* Nothing else. The ``xgboost`` wheel installed from PyPI bundles its own
  CUDA runtime, so a separate CUDA toolkit installation is **not** needed.

To confirm your installed build supports CUDA:

.. code-block:: python

    import xgboost as xgb
    print(xgb.build_info()["USE_CUDA"])   # True if GPU-capable

.. _gpu-compute-capability:

GPU Generation and the XGBoost Version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``USE_CUDA`` being ``True`` is necessary but not sufficient. Every NVIDIA GPU
has a *compute capability* (also written ``SM``) identifying its generation,
and a wheel carries compiled code only for the generations it was built for.
As new hardware appears, older ones are dropped. If your GPU predates the
wheel's oldest supported generation, training fails at ``fit`` time:

.. code-block:: text

    XGBoostError: This program was not compiled for SM 60
    : cudaErrorInvalidDevice: invalid device ordinal

Nothing earlier warns you — the import succeeds, ``build_info()`` reports
``USE_CUDA: True``, and the GPU appears in ``nvidia-smi``. Check your compute
capability with:

.. code-block:: bash

    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

A worked example: the Tesla P100 is compute capability **6.0** (Pascal), and
support for it ends partway through the 3.x series:

======================= ==================================
``xgboost`` release     Trains on a P100 (``SM 60``)
======================= ==================================
3.2.0                   yes — the last release that does
3.3.0                   no
3.4.0                   no
======================= ==================================

Since ``aiqclib`` requires only ``xgboost>=3.0.2``, a fresh install resolves to
the newest release — so a P100 that worked can stop working after an unrelated
dependency update.

If you rely on an older GPU, pin the version where you deploy rather than in
``aiqclib`` itself:

.. code-block:: bash

    pip install "aiqclib" "xgboost<3.3"

Pinning in your image or environment keeps the constraint attached to the
machine that needs it, instead of holding every other user back to an older
release. Where a CUDA toolkit is available, you can list a wheel's compiled
generations directly:

.. code-block:: bash

    cuobjdump --list-elf .../site-packages/xgboost/lib/libxgboost.so

Configuration
-------------

Set ``device: cuda`` in the ``model_params`` of the ``model`` step. Pairing it
with ``tree_method: hist`` is recommended, as that is the algorithm with a GPU
implementation.

Single Algorithm
^^^^^^^^^^^^^^^^

.. code-block:: yaml
   :emphasize-lines: 6, 12

    step_class_sets:
      - name: training_step_set_1
        steps:
          input: InputTrainingSetA
          validate: KFoldValidation
          model: XGBoost
          build: BuildModel

    step_param_sets:
      - name: training_param_set_1
        steps:
          input: { }
          validate: { }
          model: { model_params: { device: cuda, tree_method: hist } }
          build: { }

.. warning::
   ``device`` must go inside ``model_params``, not directly under ``model``.
   Keys placed directly under ``model`` are step parameters (such as
   ``calculate_shap``) and never reach the algorithm — a hyperparameter put
   there is silently ignored.

Model Suite
^^^^^^^^^^^

With ``ModelSuite``, ``device: cuda`` must **not** be applied to every model:
the scikit-learn algorithms reject it with
``TypeError: unexpected keyword argument 'device'``. Give it to XGBoost alone
by nesting it under the model's name:

.. code-block:: yaml
   :emphasize-lines: 8

    step_param_sets:
      - name: training_param_set_1
        steps:
          input: { }
          validate: { }
          model: {
                   methods: [ DT, XGB, RF ],
                   model_params: { XGB: { device: cuda, tree_method: hist } }
                 }
          build: { }

Only the named model receives that section; ``DT`` and ``RF`` keep their
defaults. See :doc:`algorithm_selection` for how per-model parameters combine
with shared ones.

.. note::
   In earlier versions every method in the suite had to be given its own
   entry, because a model that was not named received the whole
   ``model_params`` dictionary — including the other models' sections — and
   failed to construct. Naming only the models you want to configure now works
   as expected.

Saved Models Remain Portable
----------------------------

A model trained on a GPU records ``device: cuda`` in the saved ``.joblib``
file. Loading it on a machine with no GPU is safe: XGBoost detects the absence
and falls back to CPU, emitting

.. code-block:: text

    WARNING: No visible GPU is found, setting device to CPU.

Predictions are unaffected. You can therefore train on a GPU server and run
classification on CPU-only machines with the same model files.

.. _gpu-prediction-device:

The ``DMatrix`` Fallback Warning
--------------------------------

When predicting with a GPU-trained model you will see:

.. code-block:: text

    WARNING: Falling back to prediction using DMatrix due to mismatched
    devices. XGBoost is running on: cuda:0, while the input data is on: cpu.

This is expected and harmless, and — despite how it reads — it does **not**
mean the work moves to the CPU. ``aiqclib`` passes feature data as a pandas
DataFrame, which lives in host memory; XGBoost wraps it in a ``DMatrix``,
copies it to the device, and predicts there. The warning reports that copy,
not a change of device. The only cost is the transfer.

This matters most for SHAP values, where it is easy to assume the opposite.
``aiqclib`` constructs ``shap.TreeExplainer(model)`` with no background
dataset, so SHAP resolves ``feature_perturbation`` to ``tree_path_dependent``
and takes its XGBoost fast path, computing the values with
``booster.predict(..., pred_contribs=True)``. That call reaches the same
booster, still carrying ``device: cuda``, so **TreeSHAP runs on the GPU** —
and it can benefit far more than fitting does.

.. _gpu-worth-it:

Is It Worth It?
---------------

Not automatically, and the gain may not come from the stage you expect.
Moving data to the GPU has a fixed overhead that only pays off once the
dataset is large enough, and on small datasets GPU training is no faster than
CPU, or slower.

A measured example — one region of CTD data, on one machine, with ``device``
the only difference between the two runs and the CPU run pinned to
``n_jobs: 20``:

======================================== ========== ========== =========
Step of ``train_and_evaluate``           GPU        CPU        Speedup
======================================== ========== ========== =========
Cross-validation (fitting only)          2509s      2595s      1.03x
Validation reports and plots             62s        61s        1.00x
Build, test and SHAP                     7604s      17716s     2.33x
Final model fit and write                699s       684s       0.98x
**Whole train phase**                    **10885s** **21059s** **1.93x**
======================================== ========== ========== =========

Before reading that as an argument for a GPU, note what it implies: if SHAP is
the cost, then *not computing it* is the larger lever. ``calculate_shap:
false`` removes most of that runtime outright, where the GPU halves it — see
:ref:`shap-cost` for what it costs in each phase.

The phase came out roughly twice as fast, but **fitting gained nothing**.
Cross-validation and the final model are fitting alone, and both were level
with the CPU; the dataset was simply not large enough for the GPU to pay off
there. The entire saving sits in the one step that also computes SHAP values.
The same run with ``enable_shap: false`` would have shown no useful difference
at all.

Two things follow. Benchmark the workflow you actually run, rather than a bare
``fit``, or you will measure the part that did not change. And read a
comparison's thread count before trusting the ratio — the CPU side of this one
had 20 threads, and TreeSHAP on CPU scales with them, so a different
``n_jobs`` moves the 2.33x.

Timing the whole phase both ways on your own data settles it:

.. code-block:: python

    import time
    import aiqclib as aq

    config = aq.read_config(str(config_file))

    for device in ("cpu", "cuda"):
        params = {"tree_method": "hist"}
        if device == "cuda":
            params["device"] = "cuda"
        config.data["step_param_set"]["steps"]["model"]["model_params"] = params

        start = time.time()
        aq.train_and_evaluate(config)
        print(f"{device}: {time.time() - start:.1f}s")

Also check that your data fits in GPU memory (``nvidia-smi`` reports the
total), remembering that the histogram algorithm needs working space beyond
the dataset itself.

.. _gpu-shap-other-trees:

SHAP for the Other Tree Models
------------------------------

``RandomForest`` and ``DecisionTree`` compute their SHAP values on the CPU.
``shap`` does ship a GPU implementation, ``GPUTreeExplainer``, and on those
models it is dramatically faster — a measured **28x** on a 100-tree, depth-6
forest explaining 20,000 rows (15.2s to 0.5s), agreeing with the CPU result to
within floating-point noise.

``aiqclib`` does not use it, for two reasons worth knowing before you reach
for it yourself.

**It does not help where the time goes.** For XGBoost — the algorithm whose
SHAP values dominate a typical run — ``GPUTreeExplainer`` is *slower* than the
ordinary ``TreeExplainer``, 0.2s against 0.1s in that same benchmark. The
plain explainer already reaches the GPU (see
:ref:`gpu-prediction-device`), so the specialised one only adds marshalling.
It accelerates exactly the algorithms that had no GPU path, and no others.

**It is not in the published wheel.** ``GPUTreeExplainer`` calls a compiled
extension, ``_cext_gpu``, that no PyPI release of ``shap`` contains. Calling
it on a normal install fails:

.. code-block:: text

    ImportError: cannot import name '_cext_gpu' from 'shap'

Building it requires the CUDA *toolkit* — ``nvcc``, not merely a driver —
which is the one thing the rest of this page does not otherwise need, since
the ``xgboost`` wheel bundles its own CUDA runtime.

If your workflow leans on the scikit-learn tree models and you want it anyway,
build ``shap`` from source in an image based on an NVIDIA CUDA *devel* tag.
Four traps, each of which produces a working install that fails only when the
explainer is finally called:

* **Use CUDA 12, not 13.** ``shap`` compiles for ``sm_60`` upward, and CUDA 13
  dropped Pascal — a 13.x toolkit fails on the architecture an older card
  needs. The same ceiling described in :ref:`gpu-compute-capability`, reached
  from the other direction.
* **The trigger is version-specific.** From 0.52 the build is CMake-based and
  switched on by ``SHAP_ENABLE_CUDA=1``. Earlier releases ignore that variable
  and always attempt CUDA — but catch any compile error, warn
  ``Could not compile cuda extensions``, and retry without it. The install
  then *succeeds*, having produced a CPU-only ``shap``. Capture the build log
  and search it for that warning rather than trusting the exit status.
* **Verify from outside the source tree.** Running ``import shap`` with the
  checkout as the working directory picks up the source package rather than
  the installed one, which never holds the compiled extension — an
  ``ImportError`` that looks like a failed build but is not.
* **Check what landed**, rather than that the build finished:

  .. code-block:: python

      import glob, os, shap
      print(sorted(os.path.basename(p)
                   for p in glob.glob(os.path.join(os.path.dirname(shap.__file__), "*.so"))))

  ``_cext`` alone means the GPU extension was skipped.

Then use it directly on the models that benefit, leaving XGBoost on the
ordinary explainer:

.. code-block:: python

    import shap

    explainer = shap.explainers.GPUTree(random_forest_model)
    shap_values = explainer.shap_values(x_test)

Running on a Remote GPU Server
------------------------------

Inside a container, the GPU is not visible by default. The container runtime
must expose the host driver — for Docker and Podman this is the `NVIDIA
Container Toolkit
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/>`_,
after which the GPU is requested per container (``--gpus all`` for Docker).

Because the ``xgboost`` wheel bundles its CUDA runtime, the image needs no
CUDA base layer — an ordinary Python base image plus ``uv sync`` is enough.
The host driver still has to be present and recent enough for the CUDA version
the wheel was built against, which ``xgboost.build_info()["CUDA_VERSION"]``
reports.

Verify inside the container before running a full workflow. Fit a small model
rather than only checking that the GPU is visible — a driver that is present
and a build that reports ``USE_CUDA`` still leave
:ref:`compute capability <gpu-compute-capability>` untested:

.. code-block:: bash

    nvidia-smi        # driver visible? note the compute capability

.. code-block:: python

    import json
    import numpy as np
    import xgboost as xgb

    X = np.random.rand(10_000, 20).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)

    booster = xgb.train(
        {"device": "cuda", "tree_method": "hist", "objective": "binary:logistic"},
        xgb.DMatrix(X, label=y),
        num_boost_round=10,
    )
    config = json.loads(booster.save_config())
    print(config["learner"]["generic_param"]["device"])   # expect cuda:0

Print the resolved device rather than trusting the run to have used the GPU:
when XGBoost cannot reach one it falls back to the CPU and completes normally,
so a successful fit alone proves nothing.
