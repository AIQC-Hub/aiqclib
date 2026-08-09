"""
This module provides the CompareFlagsBase class, the NRT QC module's
optional step 4: comparing existing NRT QC flags with the newly computed
ones.

For every configured variable that carries an existing flag column (the
``flag`` entry of its ``qc_variable_set`` definition, e.g. ``temp_qc``),
a summary report is built with three sections:

1. ``contingency`` — the cross-tabulation of existing flag value and new
   NRT flag value with counts and percentages. Works with any existing
   flag scheme; no value mapping is required.
2. ``agreement`` — binary agreement metrics (accuracy, precision, recall
   plus the underlying confusion counts), reported only when
   ``pos_flag_values`` / ``neg_flag_values`` are configured for the
   variable, using the same convention as the other modules.
3. ``item_breakdown`` — per enabled QC item, how many observations the
   item flagged within each existing flag value, showing which items
   drive agreement or disagreement.

Variables without a configured flag are silently skipped; a configured
flag column that is absent from the data raises an error (no silent skip
on typos).
"""

import os
from typing import Dict, List, Optional

import polars as pl
from polars.exceptions import ColumnNotFoundError

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase
from aiqclib.common.utils.qc_flags import (
    FLAG_GOOD,
    flag_as_int,
    normalize_flag_values,
)

#: Column schema of the comparison report (long format, one section column).
REPORT_SCHEMA: Dict = {
    "section": pl.Utf8,
    "item": pl.Utf8,
    "existing_flag": pl.Int64,
    "new_flag": pl.Int64,
    "count": pl.Int64,
    "percent": pl.Float64,
    "metric": pl.Utf8,
    "value": pl.Float64,
}


class CompareFlagsBase(DataSetBase):
    """
    Base class for the existing-vs-new flag comparison (step ``"compare"``).

    Takes the final frame produced by step 3 (:attr:`merged_data`, holding
    both the existing flag columns from the input and the newly computed
    ``{variable}_nrt_flag`` columns) and builds one report per comparable
    variable, stored in :attr:`reports` and written as TSV files.
    """

    def __init__(
        self,
        config: ConfigBase,
        merged_data: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the comparison step with the configuration and data.

        :param config: An NRT QC configuration object.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param merged_data: The final frame produced by step 3.
        :type merged_data: Optional[pl.DataFrame]
        """
        super().__init__(step_name="compare", config=config)

        #: The default per-variable report file name pattern.
        self.default_file_name: str = "nrt_qc_flag_comparison_{target_name}.tsv"

        #: A dictionary mapping variable names to report file paths.
        self.output_file_names: Dict[str, str] = self.config.get_target_file_names(
            step_name="compare", default_file_name=self.default_file_name
        )

        self.merged_data: Optional[pl.DataFrame] = merged_data
        #: A dictionary mapping variable names to comparison reports.
        self.reports: Dict[str, pl.DataFrame] = {}

    def compare_targets(self) -> None:
        """
        Build the comparison report for every comparable variable.

        Variables without a configured existing flag are skipped silently
        (there is nothing to compare against).

        :raises ValueError: If :attr:`merged_data` is empty.
        :raises polars.exceptions.ColumnNotFoundError: If a configured flag
            column is absent from the data.
        """
        if self.merged_data is None:
            raise ValueError("Member variable 'merged_data' must not be empty.")

        for target_name in self.config.get_target_names():
            flag_column = self.config.get_variable_flag(target_name)
            if flag_column is None:
                continue
            self.reports[target_name] = self.compare(target_name, flag_column)

    def compare(self, target_name: str, flag_column: str) -> pl.DataFrame:
        """
        Build the comparison report for a single variable.

        :param target_name: The variable to compare.
        :type target_name: str
        :param flag_column: The variable's existing flag column.
        :type flag_column: str
        :return: The report frame (see :data:`REPORT_SCHEMA`).
        :rtype: pl.DataFrame
        :raises polars.exceptions.ColumnNotFoundError: If the existing flag
            column or the new NRT flag column is absent from the data.
        """
        new_column = f"{target_name}_nrt_flag"
        for column in (flag_column, new_column):
            if column not in self.merged_data.columns:
                raise ColumnNotFoundError(
                    f"Column '{column}' not found in the NRT QC output; "
                    f"cannot compare flags for variable '{target_name}'."
                )

        # Existing flags may be strings, integers or floats depending on the
        # input source; unparseable values become null and appear as their own
        # contingency row.
        work = self.merged_data.with_columns(
            flag_as_int(flag_column).alias("_existing"),
            pl.col(new_column).cast(pl.Int64).alias("_new"),
        )

        rows: List[Dict] = []
        rows += self._contingency_rows(work)
        rows += self._agreement_rows(work, target_name)
        rows += self._item_breakdown_rows(work, target_name)
        return pl.DataFrame(rows, schema=REPORT_SCHEMA)

    @staticmethod
    def _contingency_rows(work: pl.DataFrame) -> List[Dict]:
        """
        Cross-tabulate existing vs new flag values.

        :param work: The frame with ``_existing`` / ``_new`` columns.
        :type work: pl.DataFrame
        :return: One ``contingency`` report row per flag combination.
        :rtype: List[Dict]
        """
        total = work.height
        contingency = (
            work.group_by(["_existing", "_new"])
            .len()
            .sort(["_existing", "_new"], nulls_last=True)
        )
        return [
            {
                "section": "contingency",
                "existing_flag": row["_existing"],
                "new_flag": row["_new"],
                "count": row["len"],
                "percent": 100.0 * row["len"] / total,
            }
            for row in contingency.iter_rows(named=True)
        ]

    def _agreement_rows(self, work: pl.DataFrame, target_name: str) -> List[Dict]:
        """
        Binary agreement metrics against the existing flags as reference.

        Existing flags are binarised with the variable's
        ``pos_flag_values`` / ``neg_flag_values`` (rows with other values
        are excluded); new flags count as positive when worse than good.
        Reported only when both value lists are configured.

        :param work: The frame with ``_existing`` / ``_new`` columns.
        :type work: pl.DataFrame
        :param target_name: The variable being compared.
        :type target_name: str
        :return: One ``agreement`` report row per metric (or none).
        :rtype: List[Dict]
        """
        target_value = self.config.get_target_dict().get(target_name, {})
        pos_values = target_value.get("pos_flag_values")
        neg_values = target_value.get("neg_flag_values")
        if not pos_values or not neg_values:
            return []

        pos_values = normalize_flag_values(pos_values)
        neg_values = normalize_flag_values(neg_values)

        subset = work.filter(pl.col("_existing").is_in(pos_values + neg_values))
        actual = pl.col("_existing").is_in(pos_values)
        predicted = pl.col("_new") > FLAG_GOOD

        counts = subset.select(
            (actual & predicted).sum().alias("tp"),
            (~actual & predicted).sum().alias("fp"),
            (actual & ~predicted).sum().alias("fn"),
            (~actual & ~predicted).sum().alias("tn"),
        ).row(0, named=True)
        n_used = subset.height

        def _ratio(numerator: int, denominator: int) -> Optional[float]:
            return numerator / denominator if denominator else None

        metrics = {
            "n_used": float(n_used),
            "tp": float(counts["tp"]),
            "fp": float(counts["fp"]),
            "fn": float(counts["fn"]),
            "tn": float(counts["tn"]),
            "accuracy": _ratio(counts["tp"] + counts["tn"], n_used),
            "precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": _ratio(counts["tp"], counts["tp"] + counts["fn"]),
        }
        return [
            {"section": "agreement", "metric": name, "value": value}
            for name, value in metrics.items()
        ]

    def _item_breakdown_rows(self, work: pl.DataFrame, target_name: str) -> List[Dict]:
        """
        Count per-item failures within each existing flag value.

        :param work: The frame with the ``_existing`` column.
        :type work: pl.DataFrame
        :param target_name: The variable being compared.
        :type target_name: str
        :return: One ``item_breakdown`` row per item and existing value.
        :rtype: List[Dict]
        """
        rows: List[Dict] = []
        for item_name in self.config.get_qc_item_names():
            for candidate in (
                f"{target_name}_qc_{item_name}",
                f"qc_{item_name}",
            ):
                if candidate not in work.columns:
                    continue
                breakdown = (
                    work.group_by("_existing")
                    .agg(
                        (pl.col(candidate) > FLAG_GOOD).sum().alias("count"),
                        pl.len().alias("group_total"),
                    )
                    .sort("_existing", nulls_last=True)
                )
                rows += [
                    {
                        "section": "item_breakdown",
                        "item": item_name,
                        "existing_flag": row["_existing"],
                        "count": row["count"],
                        "percent": 100.0 * row["count"] / row["group_total"],
                    }
                    for row in breakdown.iter_rows(named=True)
                ]
        return rows

    def write_reports(self) -> None:
        """
        Write each variable's comparison report to a TSV file.

        Writes one file per entry in :attr:`reports`; when no variable had
        an existing flag configured there is nothing to write and the
        method returns without error.
        """
        for target_name, report in self.reports.items():
            output_path = self.output_file_names[target_name]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            report.write_csv(output_path, separator="\t")
