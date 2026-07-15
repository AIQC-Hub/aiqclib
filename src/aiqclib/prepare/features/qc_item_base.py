"""
This module defines QCItemFeatureBase, the common base class for the NRT QC
item feature classes (RTQC tests from the NRT QC recommendation document).

Each QC item is an ordinary feature class registered in the feature registry,
so its flag columns can be used both by the NRT QC module and as training
features in the dataset preparation pipeline. Subclasses implement
:meth:`compute_flags`, which produces one flag column per checked variable
(or a single profile-level column) for every observation of the input frame,
using vectorised polars expressions.
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import polars as pl

from aiqclib.common.base.feature_base import FeatureBase
from aiqclib.common.utils.qc_flags import FLAG_BAD, FLAG_GOOD

#: Columns identifying a single observation across the pipeline.
OBSERVATION_KEYS: List[str] = ["platform_code", "profile_no", "observation_no"]

#: Columns identifying a profile.
PROFILE_KEYS: List[str] = ["platform_code", "profile_no"]


class QCItemFeatureBase(FeatureBase):
    """
    Abstract base class for NRT QC item feature classes.

    Subclasses define :attr:`item_name`, :attr:`default_params`, and
    :meth:`compute_flags`. The base class resolves parameters
    (configuration ``params`` override :attr:`default_params` per key),
    the ``fail_flag`` emitted on failure (default 4, bad data), and the
    dual output mode:

    - Inside the dataset preparation pipeline (``selected_rows`` given),
      :attr:`features` holds ``row_id`` plus the flag columns for the
      selected rows, like any other feature class.
    - Inside the NRT QC module (no ``selected_rows``), :attr:`features`
      holds the observation keys plus the flag columns for every row of
      :attr:`filtered_input`.

    Flag columns follow the naming scheme ``{variable}_qc_{item_name}``
    for variable-specific items and ``qc_{item_name}`` for profile-level
    items, and are never null: 1 (good) or the item's fail flag.
    """

    #: Short item name used in output column names (e.g. "global_range").
    item_name: str = ""

    #: Built-in default parameters (the values from the NRT QC spec).
    #: Parameters supplied via ``feature_info["params"]`` override these
    #: per key.
    default_params: Dict = {}

    #: Parameter keys that are settings rather than variable entries;
    #: used when deriving the checked variables from :attr:`params`.
    scalar_param_names: Tuple[str, ...] = ()

    def __init__(
        self,
        target_name: Optional[str] = None,
        feature_info: Optional[Dict] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
        filtered_input: Optional[pl.DataFrame] = None,
        selected_rows: Optional[Dict[str, pl.DataFrame]] = None,
        summary_stats: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the QC item with resolved parameters and fail flag.

        :param target_name: The name of the target variable used to index
                            :attr:`selected_rows` in the preparation pipeline.
                            Defaults to None.
        :type target_name: Optional[str]
        :param feature_info: A dictionary describing the item configuration.
                             Recognised keys: ``params`` (per-item parameters,
                             overriding :attr:`default_params`), ``fail_flag``
                             (flag emitted on failure, default 4), and
                             ``col_names`` (explicit list of variables to
                             check). Defaults to None.
        :type feature_info: Optional[Dict]
        :param selected_profiles: (Unused by QC items) A Polars DataFrame of
                                  pre-selected profiles. Defaults to None.
        :type selected_profiles: Optional[pl.DataFrame]
        :param filtered_input: The observations to run the QC item on.
                               Defaults to None.
        :type filtered_input: Optional[pl.DataFrame]
        :param selected_rows: A dictionary of target-specific DataFrames; when
                              provided, :attr:`features` is narrowed to these
                              rows. Defaults to None.
        :type selected_rows: Optional[Dict[str, pl.DataFrame]]
        :param summary_stats: (Unused by QC items) A Polars DataFrame of
                              summary statistics. Defaults to None.
        :type summary_stats: Optional[pl.DataFrame]
        """
        super().__init__(
            target_name=target_name,
            feature_info=feature_info,
            selected_profiles=selected_profiles,
            filtered_input=filtered_input,
            selected_rows=selected_rows,
            summary_stats=summary_stats,
        )
        info = self.feature_info or {}
        #: Resolved parameters: configuration values override the defaults.
        self.params: Dict = {**self.default_params, **(info.get("params") or {})}
        #: The flag emitted when an observation fails this item.
        self.fail_flag: int = info.get("fail_flag", FLAG_BAD)

    def get_variables(self) -> List[str]:
        """
        Return the variables this item produces flag columns for.

        An explicit ``col_names`` list in :attr:`feature_info` wins;
        otherwise the variables are the parameter keys that are not
        settings (see :attr:`scalar_param_names`).

        :return: Variable names, in configuration order.
        :rtype: List[str]
        """
        info = self.feature_info or {}
        if info.get("col_names"):
            return list(info["col_names"])
        return [k for k in self.params if k not in self.scalar_param_names]

    def flag_column_name(self, variable: Optional[str] = None) -> str:
        """
        Return the output column name for this item.

        :param variable: The checked variable, or :obj:`None` for
                         profile-level items.
        :type variable: Optional[str]
        :return: ``{variable}_qc_{item_name}`` or ``qc_{item_name}``.
        :rtype: str
        """
        if variable is None:
            return f"qc_{self.item_name}"
        return f"{variable}_qc_{self.item_name}"

    def _flag_expr(self, fail: pl.Expr, alias: str) -> pl.Expr:
        """
        Turn a boolean failure condition into a flag column expression.

        Null conditions (e.g. from comparisons on missing values) count as
        a pass, so the resulting column is never null.

        :param fail: A boolean expression that is True where the item fails.
        :type fail: pl.Expr
        :param alias: The output column name.
        :type alias: str
        :return: An Int64 flag expression (1 or :attr:`fail_flag`).
        :rtype: pl.Expr
        """
        return (
            pl.when(fail.fill_null(False))
            .then(pl.lit(self.fail_flag, dtype=pl.Int64))
            .otherwise(pl.lit(FLAG_GOOD, dtype=pl.Int64))
            .alias(alias)
        )

    def _select_flags(
        self, df: pl.DataFrame, flag_exprs: List[pl.Expr]
    ) -> pl.DataFrame:
        """
        Build the standard flag frame: observation keys plus flag columns.

        :param df: The input observations.
        :type df: pl.DataFrame
        :param flag_exprs: The flag column expressions of this item.
        :type flag_exprs: List[pl.Expr]
        :return: One row per input row with keys and flag columns.
        :rtype: pl.DataFrame
        """
        return df.select([*OBSERVATION_KEYS, *flag_exprs])

    @abstractmethod
    def compute_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute this item's flag column(s) for every row of ``df``.

        Must return the observation key columns plus one flag column per
        checked variable (or a single profile-level flag column), with the
        same number of rows as ``df``.

        :param df: The observations to check.
        :type df: pl.DataFrame
        :return: Observation keys plus flag column(s).
        :rtype: pl.DataFrame
        """
        pass  # pragma: no cover

    def extract_features(self) -> None:
        """
        Compute the flag columns and store them in :attr:`features`.

        With :attr:`selected_rows` present (preparation pipeline), the flags
        are joined onto the target's selected rows and keyed by ``row_id``;
        otherwise (NRT QC module) the full flag frame is kept.
        """
        flags = self.compute_flags(self.filtered_input)

        if self.selected_rows is not None and self.target_name is not None:
            self.features = (
                self.selected_rows[self.target_name]
                .select(["row_id", *OBSERVATION_KEYS])
                .join(flags, on=OBSERVATION_KEYS, maintain_order="left")
                .drop(OBSERVATION_KEYS)
            )
        else:
            self.features = flags

    def scale_first(self) -> None:
        """
        (No-op) QC flags are not scaled before extraction.
        """
        pass  # pragma: no cover

    def scale_second(self) -> None:
        """
        (No-op) QC flags are kept as raw flag values. A mapping to [0, 1]
        can be added here if the flags are later used as model features.
        """
        pass  # pragma: no cover


class QCNeighborStencilBase(QCItemFeatureBase):
    """
    Base class for QC items using the V1/V2/V3 neighbour stencil.

    The spike (RTQC9) and gradient (RTQC11) tests both evaluate a test value
    from a measurement V2 and its vertical neighbours V1 (above) and V3
    (below), compared against a depth-dependent threshold: ``shallow`` for
    pressures less than ``depth_threshold`` (500 db by default) and ``deep``
    otherwise. Neighbours are taken in observation order within each profile;
    the first and last observations of a profile have no complete stencil and
    always pass, as do stencils involving null values.

    Subclasses implement :meth:`test_value_expr` with their formula.
    """

    scalar_param_names: Tuple[str, ...] = ("depth_threshold",)

    #: Column defining the vertical order of observations within a profile.
    order_column: str = "observation_no"

    @abstractmethod
    def test_value_expr(self, v1: pl.Expr, v2: pl.Expr, v3: pl.Expr) -> pl.Expr:
        """
        Return the item's test value for the V1/V2/V3 stencil.

        :param v1: The neighbouring value above V2.
        :type v1: pl.Expr
        :param v2: The value being tested.
        :type v2: pl.Expr
        :param v3: The neighbouring value below V2.
        :type v3: pl.Expr
        :return: The test value expression compared against the threshold.
        :rtype: pl.Expr
        """
        pass  # pragma: no cover

    def _stencil_thresholds(self, variable: str) -> Dict:
        """
        Return the validated {shallow, deep} thresholds for a variable.

        :param variable: The variable to look up.
        :type variable: str
        :return: The thresholds dictionary with ``shallow`` and ``deep`` keys.
        :rtype: Dict
        :raises ValueError: If the variable has no complete threshold entry.
        """
        thresholds = self.params.get(variable)
        if not isinstance(thresholds, dict) or not {"shallow", "deep"} <= set(
            thresholds
        ):
            raise ValueError(
                f"QC item '{self.item_name}' requires 'shallow' and 'deep' "
                f"params for variable '{variable}'."
            )
        return thresholds

    def compute_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Flag values whose stencil test value exceeds the depth-dependent
        threshold.

        :param df: The observations to check.
        :type df: pl.DataFrame
        :return: Observation keys plus one flag column per variable.
        :rtype: pl.DataFrame
        """
        depth_threshold = self.params["depth_threshold"]

        exprs: List[pl.Expr] = []
        for variable in self.get_variables():
            thresholds = self._stencil_thresholds(variable)
            v2 = pl.col(variable)
            v1 = v2.shift(1).over(PROFILE_KEYS, order_by=self.order_column)
            v3 = v2.shift(-1).over(PROFILE_KEYS, order_by=self.order_column)

            threshold = (
                pl.when(pl.col("pres") < depth_threshold)
                .then(pl.lit(thresholds["shallow"]))
                .otherwise(pl.lit(thresholds["deep"]))
            )
            fail = self.test_value_expr(v1, v2, v3) > threshold
            exprs.append(self._flag_expr(fail, self.flag_column_name(variable)))

        return self._select_flags(df, exprs)
