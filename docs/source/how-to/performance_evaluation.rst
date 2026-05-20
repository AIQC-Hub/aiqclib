Performance Evaluation
======================

Each phase that runs a model writes a per-target **model-scores** file: a
long-format table with one row per scored prediction. These files are intended
for performance evaluation — computing ROC curves, Precision-Recall curves,
confusion matrices at a chosen threshold, AUC, and so on — including in
external tools such as R.

Schema
------

.. list-table::
   :header-rows: 1
   :widths: 15 12 73

   * - Column
     - Type
     - Description
   * - ``method``
     - string
     - The model that produced the row, as a lowercase short name (for
       example ``xgb``, ``dt``, ``rf``, ``logit``, ``lda``, ``svm``, ``knn``,
       ``gnb``, ``mlp``).
   * - ``k``
     - integer
     - Fold index. For validation this ranges over the cross-validation folds
       (``0 .. K-1``); for the build/test and classify phases there is a single
       pass, so ``k`` is ``0``.
   * - ``label``
     - integer
     - Ground-truth label (``0`` or ``1``).
   * - ``score``
     - float
     - Model probability for the positive class, in ``[0, 1]``.

The target variable (``temp``, ``psal``, ``pres``) is encoded in the *file
name*, not in a column — each file holds a single target.

.. note::

   These files intentionally do **not** contain a ``predicted_label`` column.
   A predicted label is simply ``score >= threshold``, so storing it would bake
   in a single threshold and make the file less useful for sweeping the
   threshold — which is exactly what ROC and Precision-Recall analysis does.
   Derive labels yourself at whatever threshold you need:

   * Python / Polars: ``predicted = (df["score"] >= t).cast(pl.Int64)``
   * R: ``predicted <- as.integer(df$score >= t)``

   See :doc:`prediction_threshold` for the threshold used when the library
   itself generates predicted labels.

File names by phase
-------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Phase
     - Single model
     - Suite (multiple methods)
   * - Validation (step 2)
     - ``model_scores_{target}.parquet``
     - ``model_scores_{method}_{target}.parquet``
   * - Build / test (step 4)
     - ``test_model_scores_{target}.parquet``
     - ``test_model_scores_{target}.parquet``
   * - Classify (step 6)
     - ``classify_model_scores_{target}.parquet``
     - ``classify_model_scores_{target}.parquet``

In the suite build/classify files, the ``method`` column distinguishes the
methods within a single per-target file. In the suite validation files, each
method has its own file, so the ``method`` column is constant within a file.

Example: ROC curve in R
-----------------------

.. code-block:: r

   library(arrow)
   library(pROC)

   df <- read_parquet("test_model_scores_temp.parquet")
   roc_obj <- roc(df$label, df$score)   # uses scores directly, all thresholds
   plot(roc_obj)
   auc(roc_obj)

   # Confusion matrix at a chosen operating point:
   t <- 0.7
   predicted <- as.integer(df$score >= t)
   table(actual = df$label, predicted = predicted)

Because the files store raw ``score`` values rather than thresholded labels,
the same file supports any threshold you choose without rerunning the
pipeline.
