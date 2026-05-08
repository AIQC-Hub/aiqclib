===========
SHAP Values
===========

``aiqclib`` integrates `SHAP <https://shap.readthedocs.io>`_ (SHapley Additive exPlanations) to easily identify exactly why a model flagged a specific data point (e.g., "temperature is abnormally high for this specific depth").

Setting this configuration enables ``aiqclib`` to generate SHAP values during the testing and classification phases, but intentionally disables it during the validation (k-fold) phase to save computational time.

Configuration
-------------

To enable SHAP value creation, set the ``calculate_shap`` key in the ``model`` step within ``step_param_sets`` to ``True``.

Training Configuration Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

SHAP Explainers
---------------

Different SHAP explainers are automatically selected based on the specified ML algorithm to optimize performance.

The "Fast & Exact" Group (shap.TreeExplainer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  - **Models**: XGBoost, Random Forest, Decision Tree.
  - **How it works**: SHAP has a highly optimized, C++ backed explainer specifically for tree-based models. It calculates exact Shapley values incredibly fast.

The "Fast & Linear" Group (shap.LinearExplainer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  - **Models**: Logistic Regression, Linear Discriminant Analysis.
  - **How it works**: SHAP can exactly compute feature contributions for linear models by looking at the model's coefficients and the data distribution.

The "Slow & Model-Agnostic" Group (shap.KernelExplainer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  - **Models**: SVM, K-Nearest Neighbors (KNN), Gaussian Naive Bayes (GNB), Multi-layer Perceptron (MLP).
  - **How it works**: Because these models have complex, non-linear, or instance-based internal structures without specialized SHAP math, SHAP must treat them as "Black Boxes." It perturbs the input data hundreds or thousands of times, asks the model for predictions, and solves a regression problem to estimate the SHAP values.
