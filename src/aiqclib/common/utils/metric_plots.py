"""
This module provides functions for generating and saving performance metric plots,
specifically Receiver Operating Characteristic (ROC) curves and Precision-Recall (PR) curves.
It supports plotting for individual models across multiple cross-validation folds (with
mean and standard deviation) or comparing multiple models/methods on a single plot.
Plots are saved as SVG files.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def create_metric_plots(model) -> None:
    """
    Create and save ROC and Precision-Recall plots as an SVG file for a single model.

    Generates a figure with two subplots (ROC on left, PR on right) based on
    the data in ``model.contingency_tables``. If the contingency table contains
    multiple unique 'k' values (folds), it plots individual fold curves and then
    the mean curve with a shaded confidence band (standard deviation).

    The output file path is determined by ``model.output_file_names['metric_plot']``.

    :param model: An object containing evaluation results and output configuration.
                  It is expected to have the following attributes:

                  - ``contingency_tables`` (dict[str, polars.DataFrame]): A dictionary
                    where keys are target names and values are Polars DataFrames. Each
                    DataFrame must contain at least 'k' (fold identifier), 'label'
                    (true binary labels), and 'score' (prediction probabilities/scores)
                    columns.
                  - ``output_file_names`` (dict[str, dict[str, str]]): A dictionary
                    containing output file paths. Specifically,
                    ``output_file_names['metric_plot'][target_name]`` should provide
                    the full path where the plot for a given target will be saved.

    :type model: object
    :raises ValueError: If ``model.contingency_tables`` is empty.
    :return: None
    :rtype: None
    """
    if not model.contingency_tables:
        raise ValueError("Member variable 'contingency_tables' must not be empty.")

    for target_name, df in model.contingency_tables.items():
        output_path = model.output_file_names["metric_plot"][target_name]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        unique_k = df["k"].unique().sort()
        has_folds = len(unique_k) > 1

        plt.rcParams.update({"font.size": 14})
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 6))

        # --- ROC Curve Setup ---
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        # --- Precision-Recall Curve Setup ---
        # We interpolate over recall to average precision
        mean_recall = np.linspace(0, 1, 100)
        precisions = []

        # Loop through folds (or single run)
        for k in unique_k:
            fold_data = df.filter(pl.col("k") == k)
            y_true = fold_data["label"].to_numpy()
            y_score = fold_data["score"].to_numpy()

            if len(np.unique(y_true)) < 2:
                continue  # Skip folds with only one class

            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)

            # PR
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            # Reverse to ensure increasing recall for interpolation
            prec = prec[::-1]
            rec = rec[::-1]
            interp_prec = np.interp(mean_recall, rec, prec)
            precisions.append(interp_prec)

            if has_folds:
                ax_roc.plot(
                    fpr,
                    tpr,
                    lw=1,
                    alpha=0.3,
                    label=f"ROC fold {k} (AUC = {roc_auc:.2f})",
                )
                ax_pr.plot(rec, prec, lw=1, alpha=0.3)

        # --- Plot Mean ROC ---
        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)

            label_roc = (
                f"Mean ROC (AUC = {mean_auc:.2f} $\\pm$ {std_auc:.2f})"
                if has_folds
                else f"ROC (AUC = {mean_auc:.2f})"
            )

            ax_roc.plot(mean_fpr, mean_tpr, color="b", label=label_roc, lw=2, alpha=0.8)

            if has_folds:
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                ax_roc.fill_between(
                    mean_fpr,
                    tprs_lower,
                    tprs_upper,
                    color="grey",
                    alpha=0.2,
                    label=r"$\pm$ 1 std. dev.",
                )

        # ROC Formatting
        ax_roc.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
        ax_roc.set_xlim([-0.05, 1.05])
        ax_roc.set_ylim([-0.05, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"ROC Curve - {target_name}")
        ax_roc.legend(loc="lower right", fontsize="small")
        ax_roc.grid(True, alpha=0.3)

        # --- Plot Mean PR ---
        if precisions:
            mean_precision = np.mean(precisions, axis=0)
            mean_ap = auc(mean_recall, mean_precision)

            label_pr = (
                f"Mean PR (AP = {mean_ap:.2f})"
                if has_folds
                else f"PR (AP = {mean_ap:.2f})"
            )

            ax_pr.plot(
                mean_recall,
                mean_precision,
                color="b",
                label=label_pr,
                lw=2,
                alpha=0.8,
            )

            if has_folds:
                std_prec = np.std(precisions, axis=0)
                prec_upper = np.minimum(mean_precision + std_prec, 1)
                prec_lower = np.maximum(mean_precision - std_prec, 0)
                ax_pr.fill_between(
                    mean_recall,
                    prec_lower,
                    prec_upper,
                    color="grey",
                    alpha=0.2,
                    label=r"$\pm$ 1 std. dev.",
                )

        # PR Formatting
        ax_pr.set_xlim([-0.05, 1.05])
        ax_pr.set_ylim([-0.05, 1.05])
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"Precision-Recall Curve - {target_name}")
        ax_pr.legend(loc="lower left", fontsize="small")
        ax_pr.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, format="svg")
        plt.close(fig)


def create_multi_method_metric_plots(model) -> None:
    """
    Create and save ROC and Precision-Recall plots for multiple methods overlaid
    on the same figure. Assumes the contingency tables have a 'method' column.

    The output file path is determined by ``model.output_file_names['metric_plot']``.

    :param model: An object containing evaluation results and output configuration.
                  It is expected to have the following attributes:

                  - ``contingency_tables`` (dict[str, polars.DataFrame]): A dictionary
                    where keys are target names and values are Polars DataFrames. Each
                    DataFrame must contain at least 'method' (method identifier), 'label'
                    (true binary labels), and 'score' (prediction probabilities/scores)
                    columns. It aggregates results across all folds/runs for each method.
                  - ``output_file_names`` (dict[str, dict[str, str]]): A dictionary
                    containing output file paths. Specifically,
                    ``output_file_names['metric_plot'][target_name]`` should provide
                    the full path where the plot for a given target will be saved.

    :type model: object
    :raises ValueError: If ``model.contingency_tables`` is empty.
    :return: None
    :rtype: None

    .. admonition:: Code Issue
       :class: warning

       The calculation of Average Precision (AP) for the Precision-Recall curve using
       ``pr_auc = auc(rec[::-1], prec[::-1])`` is incorrect.
       The ``sklearn.metrics.precision_recall_curve`` function returns `recall` values
       that are already in increasing order. Therefore, `auc(rec, prec)` should be used
       directly to calculate the Area Under the Curve for the Precision-Recall plot.
       Reversing `rec` and `prec` before passing them to `auc` when `rec` is already
       increasing will lead to an incorrect AP value.
    """
    if not model.contingency_tables:
        raise ValueError("Member variable 'contingency_tables' must not be empty.")

    for target_name, df in model.contingency_tables.items():
        output_path = model.output_file_names["metric_plot"][target_name]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.rcParams.update({"font.size": 14})
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 7))

        # Retrieve unique methods from the aggregated dataframe
        methods = df["method"].unique().to_list()

        for method in methods:
            method_data = df.filter(pl.col("method") == method)
            y_true = method_data["label"].to_numpy()
            y_score = method_data["score"].to_numpy()

            if len(np.unique(y_true)) < 2:
                continue  # Skip methods where evaluation lacks distinct classes

            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, lw=2, label=f"{method} (AUC = {roc_auc:.2f})")

            # PR
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(rec[::-1], prec[::-1])
            ax_pr.plot(rec, prec, lw=2, label=f"{method} (AP = {pr_auc:.2f})")

        # ROC Formatting
        ax_roc.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", alpha=0.5)
        ax_roc.set_xlim([-0.05, 1.05])
        ax_roc.set_ylim([-0.05, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"ROC Curve - {target_name}")
        ax_roc.legend(loc="lower right", fontsize="small")
        ax_roc.grid(True, alpha=0.3)

        # PR Formatting
        ax_pr.set_xlim([-0.05, 1.05])
        ax_pr.set_ylim([-0.05, 1.05])
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"Precision-Recall Curve - {target_name}")
        ax_pr.legend(loc="lower left", fontsize="small")
        ax_pr.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, format="svg")
        plt.close(fig)
