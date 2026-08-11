================
GPU Acceleration
================

``aiqclib`` can train **XGBoost** models on an NVIDIA GPU. This requires no
code change and no extra dependency — only a configuration setting.

What Can Use a GPU
------------------

Only XGBoost. This is worth being clear about before investing in a GPU
setup:

============================== =========================================
Stage / component              Device
============================== =========================================
Dataset preparation            CPU (polars / pandas)
**XGBoost** training           **GPU**, when ``device: cuda`` is set
The other eight algorithms     CPU — scikit-learn has no GPU backend
SHAP value calculation         CPU
Classification (prediction)    CPU — see :ref:`gpu-prediction-device`
============================== =========================================

So a GPU accelerates the model-fitting step of a single algorithm. If your
pipeline is dominated by dataset preparation, or you train with
``ModelSuite`` across all nine algorithms, the overall speedup will be
proportionally smaller.

Requirements
------------

* An NVIDIA GPU and a working driver (check with ``nvidia-smi``).
* Nothing else. The ``xgboost`` wheel installed from PyPI bundles its own
  CUDA runtime, so a separate CUDA toolkit installation is **not** needed.

To confirm your installed build supports CUDA:

.. code-block:: python

    import xgboost as xgb
    print(xgb.build_info()["USE_CUDA"])   # True if GPU-capable

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

Prediction Stays on the CPU
---------------------------

When predicting with a GPU-trained model you will see:

.. code-block:: text

    WARNING: Falling back to prediction using DMatrix due to mismatched
    devices. XGBoost is running on: cuda:0, while the input data is on: cpu.

This is expected and harmless. ``aiqclib`` passes feature data to XGBoost as a
pandas DataFrame, which lives in host memory, so prediction runs on the CPU.
The GPU accelerates **training only**.

Is It Worth It?
---------------

Not automatically. Moving data to the GPU has a fixed overhead that only pays
off once the dataset is large enough, and on small datasets GPU training is
*slower* than CPU. Benchmark with your own data before committing to a GPU
workflow:

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

Verify inside the container before running a full workflow:

.. code-block:: bash

    nvidia-smi                                     # driver visible?
    uv run python -c "import xgboost; print(xgboost.build_info()['USE_CUDA'])"
