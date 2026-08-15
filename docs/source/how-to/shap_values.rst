===========
SHAP Values
===========

``aiqclib`` integrates `SHAP <https://shap.readthedocs.io>`_ (SHapley Additive exPlanations) to easily identify exactly why a model flagged a specific data point (e.g., "temperature is abnormally high for this specific depth").

Setting this configuration enables ``aiqclib`` to generate SHAP values during the testing and classification phases, but intentionally disables it during the validation (k-fold) phase to save computational time.

Configuration
-------------

To enable SHAP value creation, set the ``calculate_shap`` key in the ``model`` step within ``step_param_sets`` to ``True``.

Training Configuration Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml
   :emphasize-lines: 6

   step_param_sets:
     - name: training_param_set_1
       steps:
         input: { }
         validate: { }
         model: { calculate_shap: True }
         build: { }

Classification Configuration Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml
   :emphasize-lines: 9

   step_param_sets:
     - name: classify_param_set_1
       steps:
         input: { }
         summary: { }
         select: { }
         locate: { }
         extract: { }
         model: { calculate_shap: True }
         classify: { }
         concat: { }

.. note::
   The same configuration works when using the ``ModelSuite`` class for evaluating multiple algorithms.

.. _shap-cost:

What It Costs
-------------

SHAP is off by default, and switching it on is normally the single largest
change you can make to how long a run takes. Nothing in the output attributes
the time to it, so the phase simply takes much longer with no indication of
which setting is responsible.

Measured on production CTD data, with the values computed on a GPU:

============================================ ==========================
Phase                                        Share of the time in SHAP
============================================ ==========================
Training (small dataset)                     ~47%
Training (large dataset)                     ~65%
**Classification**                           **~99%**
============================================ ==========================

Classification is almost entirely SHAP because it does no fitting at all — it
loads a model, predicts, and explains, and the prediction is a rounding error
beside the explanation.

Two consequences worth planning around:

* **The cost grows faster than your data.** TreeSHAP scales with rows ×
  trees × depth², so a dataset an order of magnitude larger can cost more than
  an order of magnitude more. Between two real datasets differing about
  tenfold in fitting time, SHAP time differed twentyfold.
* **Turning it off is a bigger lever than any hardware.** ``calculate_shap:
  false`` removes roughly half of a training phase and nearly all of a
  classification phase. For XGBoost, ``device: cuda`` computes SHAP on the GPU
  and roughly halves its cost (see :doc:`gpu_acceleration`), which is
  worthwhile but much smaller.

Those figures are for XGBoost, which uses the fastest explainer available. The
algorithms served by ``shap.KernelExplainer`` below — SVM, KNN, Gaussian Naive
Bayes and the multi-layer perceptron — are slower by a wide margin, because
that explainer has no shortcut and must re-query the model thousands of times
per explanation. Enabling SHAP for a ``ModelSuite`` that includes them costs
far more than these numbers suggest, and the cost is dominated by those
methods rather than shared evenly across the suite.

``aiqclib`` warns once per run when SHAP is computed over more than 100,000
rows, naming the setting that disables it. Treat the threshold as a rough
signal rather than a boundary: the real cost is rows × trees × depth², so a
deep forest can be slow well below it and a shallow one comfortable well
above. Cross-validation is exempt regardless — SHAP is switched off for the
k-fold phase so that validation does not pay this cost once per fold.

The decision is yours and depends on use, not on speed: if the values inform QC
decisions or model interpretation, this is simply their price. If they are
written and rarely read, they are the first thing to switch off.

Importing SHAP Values
---------------------

When enabled, ``aiqclib`` writes per-instance SHAP values to a Parquet file during the testing and classification phases. Import that file with ``read_shap_scores`` for visualization and evaluation:

.. code-block:: python

   import aiqclib as aq

   shap = aq.read_shap_scores("classify_shap_values_temp.parquet")

The file has three metadata columns — ``label``, ``predicted_label`` and ``score`` — followed by one ``<feature>_shap`` column per feature. By default the ``_shap`` suffix is stripped so each feature column is named by its feature (``temp_mean_shap`` becomes ``temp_mean``); the metadata columns are returned unchanged. Pass ``strip_suffix=False`` to keep the original names.

The result is a `Polars <https://pola.rs/>`_ DataFrame, ready for SHAP plots such as mean-importance bar charts, summary plots, and dependence plots. For example, to obtain the SHAP value matrix and the feature names:

.. code-block:: python

   features = [c for c in shap.columns
               if c not in ("label", "predicted_label", "score")]
   values = shap.select(features).to_numpy()

.. note::
   The importer reads Parquet, TSV, and CSV (optionally gzipped); the format is inferred from the file extension.

SHAP Explainers
---------------

Different SHAP explainers are automatically selected based on the specified ML algorithm to optimize performance.

The "Fast & Exact" Group (shap.TreeExplainer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  - **Models**: XGBoost, Random Forest, Decision Tree.
  - **How it works**: SHAP has a highly optimized, C++ backed explainer specifically for tree-based models. It calculates exact Shapley values, and is fast relative to the model-agnostic explainer below — though in absolute terms it is still usually the most expensive part of a run (see :ref:`shap-cost`).

The "Fast & Linear" Group (shap.LinearExplainer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  - **Models**: Logistic Regression, Linear Discriminant Analysis.
  - **How it works**: SHAP can exactly compute feature contributions for linear models by looking at the model's coefficients and the data distribution.

The "Slow & Model-Agnostic" Group (shap.KernelExplainer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  - **Models**: SVM, K-Nearest Neighbors (KNN), Gaussian Naive Bayes (GNB), Multi-layer Perceptron (MLP).
  - **How it works**: Because these models have complex, non-linear, or instance-based internal structures without specialized SHAP math, SHAP must treat them as "Black Boxes." It perturbs the input data hundreds or thousands of times, asks the model for predictions, and solves a regression problem to estimate the SHAP values.