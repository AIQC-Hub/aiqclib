===================
Algorithm Selection
===================

The library supports multiple machine learning algorithms spanning different logical categories, from tree-based ensembles to distance-based and neural methods.

Available Algorithms
--------------------

Except for `XGBoost <https://xgboost.readthedocs.io>`_, all methods use the implementation provided by `scikit-learn <https://scikit-learn.org>`_.

================================ ================================= ============================ ============ ======================
Category                         Algorithm                         Class Name                   Short Name   Method
================================ ================================= ============================ ============ ======================
Tree-Based & Ensemble            **XGBoost**                       XGBoost                      XGB          Ensemble (Boosting)
\                                **Random Forest**                 RandomForest                 RF           Ensemble (Bagging)
\                                **Decision Tree**                 DecisionTree                 DT           Tree
Linear & Geometric               **Logistic Regression**           LogisticRegression           Logit        Linear
\                                **Linear Discriminant Analysis**  LinearDiscriminantAnalysis   LDA          Linear / Statistical
\                                **Support Vector Machine**        SupportVectorMachine         SVM          Geometric
Instance-Based                   **K-Nearest Neighbors**           KNearestNeighbors            KNN          Distance-based
Probabilistic                    **Gaussian Naive Bayes**          GaussianNaiveBayes           GNB          Probabilistic
Neural Network                   **Multilayer Perceptron**         MultilayerPerceptron         MLP          Neural Network
================================ ================================= ============================ ============ ======================

Configuration
-------------

To select an algorithm, set the ``model`` key in ``step_class_sets`` to the algorithm's class name (e.g., ``XGBoost``).

To customize the hyperparameters for your selected algorithm, add them to the ``model`` step within ``step_param_sets``.

Training Configuration Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml
   :emphasize-lines: 6, 14

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
          model: { learning_rate: 0.01 }
          build: { }

Classification Configuration Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml
   :emphasize-lines: 9

    step_class_sets:
      - name: data_set_step_set_1
        steps:
          input: InputDataSetAll
          summary: SummaryDataSetAll
          select: SelectDataSetAll
          locate: LocateDataSetAll
          extract: ExtractDataSetAll
          model: XGB
          classify: ClassifyAll
          concat: ConcatDataSetAll

Imputation
-----------------
As non-tree-based machine learning methods do not accept NaN values, missing values are automatically imputed using ``SimpleImputer(strategy="median")`` provided by `scikit-learn <https://scikit-learn.org>`_ during the training phase.

During the classification phase, instances containing NaN values in their features are handled such that non-tree-based models output a class value of 0 and a score of 0.

Model Suite Class
-----------------

``aiqclib`` provides a model suite class that performs training and classification with multiple algorithms simultaneously. To select a set of algorithms, set the ``model`` key in ``step_class_sets`` to ``ModelSuite``.Then, specify a list of actual algorithms and their parameters using the ``methods`` and ``model_params`` keys within ``step_param_sets``, respectively.

.. note::
  Both ``methods`` and ``model_params`` keys accept "Class name" and "Short name" shown in the table above (e.g., ``XGBoost`` and ``XGB``).

In addition, the ``ModelSuite`` class requires specific counterpart classes for training and classification to correctly handle multiple outputs. These are:

* ``KFoldValidationSuite`` for k-fold validation
* ``BuildModelSuite`` for model building
* ``ClassifyAllSuite`` for classification
* ``ConcatDataSetSuite`` for the final result concatenation

Training Configuration Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml
   :emphasize-lines: 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22

    step_class_sets:
      - name: training_step_set_1
        steps:
          input: InputTrainingSetA
          validate: KFoldValidationSuite
          model: ModelSuite
          build: BuildModelSuite

    step_param_sets:
      - name: training_param_set_1
        steps:
          input: { }
          validate: { }
          model: {
                   calculate_shap: True,
                   methods: [ DT, XGB, RF ],
                   model_params: {
                     DT:  { },             # Default (you still need to set an empty dictionary)
                     XGB: { scale_pos_weight: 200 , n_jobs: 30 },
                     RF:  { n_jobs: 30 }   # Number of parallel jobs
                   }
                 }
          build: { }

Classification Configuration Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml
   :emphasize-lines: 9, 10, 11, 21

    step_class_sets:
      - name: data_set_step_set_1
        steps:
          input: InputDataSetAll
          summary: SummaryDataSetAll
          select: SelectDataSetAll
          locate: LocateDataSetAll
          extract: ExtractDataSetAll
          model: ModelSuite
          classify: ClassifyAllSuite
          concat: ConcatDataSetSuite

    step_param_sets:
      - name: training_param_set_1
        steps:
          input: { }
          summary: { }
          select: { }
          locate: { }
          extract: { }
          model: { methods: [ DT, XGB, RF ] }
          classify: { }
          concat: { }

Default Parameters
------------------

If no specific parameters are provided in ``step_param_sets``, the algorithms will initialize with the following default parameters based on their Scikit-Learn or XGBoost implementations.

Decision Tree (DT)
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``criterion``
     - ``"gini"``
     - The function to measure the quality of a split.
   * - ``splitter``
     - ``"best"``
     - The strategy used to choose the split at each node.
   * - ``max_depth``
     - ``10``
     - The maximum depth of the tree.
   * - ``min_samples_split``
     - ``10``
     - The minimum number of samples required to split an internal node.
   * - ``min_samples_leaf``
     - ``5``
     - The minimum number of samples required to be at a leaf node.
   * - ``max_features``
     - ``None``
     - The number of features to consider when looking for the best split.
   * - ``random_state``
     - ``None``
     - Controls the randomness of the estimator for reproducibility.
   * - ``class_weight``
     - ``"balanced"``
     - Weights associated with classes (e.g., ``"balanced"``).
   * - ``ccp_alpha``
     - ``0.001``
     - Complexity parameter used for Minimal Cost-Complexity Pruning.

Random Forest (RF)
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``n_estimators``
     - ``100``
     - The number of trees in the forest.
   * - ``criterion``
     - ``"gini"``
     - The function to measure the quality of a split.
   * - ``max_depth``
     - ``10``
     - The maximum depth of the trees.
   * - ``min_samples_split``
     - ``10``
     - The minimum number of samples required to split an internal node.
   * - ``min_samples_leaf``
     - ``5``
     - The minimum number of samples required to be at a leaf node.
   * - ``max_features``
     - ``"sqrt"``
     - The number of features to consider when looking for the best split.
   * - ``bootstrap``
     - ``True``
     - Whether bootstrap samples are used when building trees.
   * - ``n_jobs``
     - ``-1``
     - The number of jobs to run in parallel (``-1`` means using all processors).
   * - ``random_state``
     - ``None``
     - Controls both the randomness of the bootstrapping and feature sampling.
   * - ``class_weight``
     - ``"balanced_subsample"``
     - Weights associated with classes (e.g., ``"balanced"``).

XGBoost (XGB)
^^^^^^^^^^^^^

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``n_estimators``
     - ``100``
     - Number of boosting rounds (trees to build).
   * - ``max_depth``
     - ``10``
     - Maximum tree depth for base learners.
   * - ``learning_rate``
     - ``0.1``
     - Boosting learning rate (step size shrinkage).
   * - ``eval_metric``
     - ``"logloss"``
     - Evaluation metric for validation data.
   * - ``scale_pos_weight``
     - ``1``
     - Multiplier for the gradient of positive samples (e.g., set to sum(negative cases) / sum(positive cases)).
   * - ``n_jobs``
     - ``-1``
     - Number of parallel threads used to run XGBoost.

Logistic Regression (Logit)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``penalty``
     - ``"l2"``
     - Specifies the norm of the penalty used in regularization.
   * - ``C``
     - ``1.0``
     - Inverse of regularization strength; smaller values specify stronger regularization.
   * - ``solver``
     - ``"lbfgs"``
     - Algorithm to use in the optimization problem.
   * - ``max_iter``
     - ``200``
     - Maximum number of iterations taken for the solvers to converge.

Linear Discriminant Analysis (LDA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``solver``
     - ``"svd"``
     - Solver to use (Singular Value Decomposition).
   * - ``shrinkage``
     - ``None``
     - Shrinkage parameter, used to improve estimation of covariance matrices.
   * - ``priors``
     - ``None``
     - The class prior probabilities.
   * - ``n_components``
     - ``None``
     - Number of components for dimensionality reduction.
   * - ``store_covariance``
     - ``False``
     - If True, explicitly computes the empirical class covariance matrix.
   * - ``tol``
     - ``1.0e-4``
     - Absolute threshold for a singular value of X to be considered significant.

Support Vector Machine (SVM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``C``
     - ``1.0``
     - Regularization parameter. The strength of the regularization is inversely proportional to C.
   * - ``kernel``
     - ``"linear"``
     - Specifies the kernel type to be used in the algorithm.
   * - ``probability``
     - ``True``
     - Whether to enable probability estimates (required for ROC/PR curves).
   * - ``tol``
     - ``1e-3``
     - Tolerance for stopping criterion.
   * - ``max_iter``
     - ``200``
     - Hard limit on iterations within solver (``-1`` for no limit).
   * - ``random_state``
     - ``None``
     - Controls the pseudo random number generation for probability estimates.
   * - ``class_weight``
     - ``"balanced"``
     - Weights associated with classes (e.g., ``"balanced"``).

Gaussian Naive Bayes (GNB)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``priors``
     - ``None``
     - Prior probabilities of the classes. If specified, priors are not adjusted according to the data.
   * - ``var_smoothing``
     - ``1e-9``
     - Portion of the largest variance of all features added to variances for calculation stability.

K-Nearest Neighbors (KNN)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``n_neighbors``
     - ``5``
     - Number of neighbors to use by default for queries.
   * - ``weights``
     - ``"uniform"``
     - Weight function used in prediction (all points in neighborhood are weighted equally).
   * - ``algorithm``
     - ``"auto"``
     - Algorithm used to compute the nearest neighbors.
   * - ``leaf_size``
     - ``30``
     - Leaf size passed to BallTree or KDTree (affects memory and speed).
   * - ``p``
     - ``2``
     - Power parameter for the Minkowski metric (``2`` corresponds to Euclidean distance).
   * - ``metric``
     - ``"minkowski"``
     - The distance metric to use for the tree.
   * - ``n_jobs``
     - ``-1``
     - The number of parallel jobs to run for neighbors search.

Multilayer Perceptron (MLP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``hidden_layer_sizes``
     - ``(50,)``
     - The ith element represents the number of neurons in the ith hidden layer.
   * - ``activation``
     - ``"relu"``
     - Activation function for the hidden layer.
   * - ``solver``
     - ``"adam"``
     - The solver for weight optimization.
   * - ``alpha``
     - ``0.0001``
     - L2 penalty (regularization term) parameter.
   * - ``batch_size``
     - ``"auto"``
     - Size of minibatches for stochastic optimizers.
   * - ``learning_rate``
     - ``"constant"``
     - Learning rate schedule for weight updates.
   * - ``learning_rate_init``
     - ``0.001``
     - The initial learning rate used.
   * - ``max_iter``
     - ``100``
     - Maximum number of iterations/epochs.
   * - ``shuffle``
     - ``True``
     - Whether to shuffle samples in each iteration.
   * - ``random_state``
     - ``None``
     - Determines random number generation for weights and bias initialization.
   * - ``tol``
     - ``1e-3``
     - Tolerance for the optimization.
   * - ``early_stopping``
     - ``True``
     - Whether to use early stopping to terminate training when validation score is not improving.
   * - ``n_iter_no_change``
     - ``5``
     - Number of iterations to stop training when validation score stops improving.