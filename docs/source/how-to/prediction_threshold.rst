Prediction Threshold
=====================

By default, a predicted label is ``1`` when the model's positive-class score is
at least ``0.5``, and ``0`` otherwise. This threshold is configurable.

Configuration
-------------

Set it under the ``model`` step parameters:

.. code-block:: yaml

   step_param_set:
     steps:
       model:
         predicted_label_threshold: 0.5    # default if omitted

* **Default:** ``0.5``, applied silently when the field is absent — existing
  configurations need no change.
* **Range:** any float in ``[0, 1]``. A higher threshold makes positive
  predictions more conservative (fewer ``1``\ s); a lower threshold makes them
  more liberal.
* **Scope:** applies to every phase — validation, build/test, and classify —
  wherever a model converts scores into labels. It governs the
  ``predicted_label`` column of the user-facing **prediction** files and any
  threshold-dependent metrics in the reports.

How it is applied
-----------------

The threshold is **read from the configuration each time a model wrapper is
constructed**, and it is applied at prediction time as
``predicted_label = (score >= threshold)``.

There are three consequences worth understanding.

.. warning::

   **The threshold is not stored in the saved model file.** Model files
   (``*.joblib``) contain only the trained estimator, not the threshold. When a
   saved model is loaded for the classify phase, the threshold comes from the
   *classify* configuration in effect at that time — not from whatever was used
   during training.

**It is config-driven, not a runtime property.**
   Setting the threshold on a model object in code has no lasting effect if that
   object is later reconstructed from configuration (as happens, for example,
   when models are reloaded per target). Change the threshold by changing the
   configuration.

**Keep it consistent across phases.**
   Because each phase reads its own configuration, using different thresholds in
   the training and classify configurations will produce labels at different
   operating points for the *same* model. Unless that is intended, set the same
   ``predicted_label_threshold`` in every phase's configuration.

Relationship to model-scores files
-----------------------------------

The model-scores files described in :doc:`performance_evaluation` are
unaffected by this setting — they store raw ``score`` values, so you can
evaluate performance across all thresholds regardless of which one is
configured for label generation. The threshold only affects where the library
draws the line when it must emit a concrete ``predicted_label``.
