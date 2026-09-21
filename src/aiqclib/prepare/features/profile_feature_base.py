"""
This module defines ProfileFeatureBase, the common base class for the
feature classes that read a whole profile rather than a single observation.

Everything derived from the shape of a profile shares the same skeleton:
take the observations of the selected profiles, compute new columns along
each profile in vertical order, then narrow the result to the rows the
target actually trains on. The base class owns that skeleton, so a subclass
writes only :meth:`compute_columns`.

Two conventions run through all of them:

- **Parameters over constants.** The feature proposal names quantities, not
  thresholds, so every number a subclass would otherwise hard-code is a key
  of :attr:`~ProfileFeatureBase.default_params`, overridable per key from
  the ``params`` entry of the feature configuration.
- **Columns are opt-in.** A subclass declares
  :attr:`~ProfileFeatureBase.supported_outputs`, and the ``outputs`` entry
  of the configuration picks the subset to emit. Omitting ``outputs`` means
  all of them. A group of twenty columns is therefore adoptable one column
  at a time, which matters because these classes can otherwise widen a
  training frame considerably.
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import polars as pl

from aiqclib.common.base.feature_base import FeatureBase
from aiqclib.common.constants import OBSERVATION_KEYS
from aiqclib.common.utils.normalization import is_scaling_type, scale_flat_columns
from aiqclib.common.utils.profile_signal import sort_profiles


class ProfileFeatureBase(FeatureBase):
    """
    Abstract base class for profile-shaped feature classes.

    Subclasses define :attr:`feature_name`, :attr:`supported_outputs`,
    :attr:`default_params` and :meth:`compute_columns`. The base class
    resolves the configuration, runs the computation over the profiles in
    vertical order, and joins the result onto the target's selected rows by
    the observation keys, so :attr:`features` ends up keyed by ``row_id``
    like every other feature class.

    These classes are observation level. In the profile-level pipeline they
    must be aggregated through the extract step's ``agg`` mechanism, and
    using one directly there is an error rather than a silent mismatch.
    """

    #: The registry name of the feature, used in error messages.
    feature_name: str = ""

    #: Every column this class can emit, as short output names.
    supported_outputs: Tuple[str, ...] = ()

    #: Built-in default parameters. Values from the configuration's
    #: ``params`` entry override these per key.
    default_params: Dict = {}

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
        Initialize the feature with its resolved parameters and outputs.

        :param target_name: The name of the target variable used to index
                            :attr:`selected_rows`.
        :type target_name: Optional[str]
        :param feature_info: The feature configuration entry. Recognised
                             keys: ``params`` (overriding
                             :attr:`default_params` per key), ``outputs``
                             (the columns to emit), ``col_names`` (the
                             variables to compute them for) and
                             ``stats_set`` / ``stats`` (normalization).
        :type feature_info: Optional[Dict]
        :param selected_profiles: A Polars DataFrame of selected profiles.
        :type selected_profiles: Optional[pl.DataFrame]
        :param filtered_input: The observations of the selected profiles.
        :type filtered_input: Optional[pl.DataFrame]
        :param selected_rows: A dictionary of target-specific DataFrames.
        :type selected_rows: Optional[Dict[str, pl.DataFrame]]
        :param summary_stats: A Polars DataFrame of summary statistics.
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
        #: The output columns to emit, in declaration order.
        self.outputs: List[str] = self._resolve_outputs(info.get("outputs"))

    def _resolve_outputs(self, requested: Optional[List[str]]) -> List[str]:
        """
        Validate the requested outputs against the supported ones.

        :param requested: The ``outputs`` entry of the configuration, or
                          :obj:`None` to emit everything supported.
        :type requested: Optional[List[str]]
        :return: The outputs to emit, in :attr:`supported_outputs` order.
        :rtype: List[str]
        :raises ValueError: If an output name is not supported, or the list
                            is given but empty.
        """
        if requested is None:
            return list(self.supported_outputs)

        unknown = [name for name in requested if name not in self.supported_outputs]
        if unknown:
            raise ValueError(
                f"Unknown output(s) {unknown} for feature "
                f"'{self.feature_name}'. Valid outputs: "
                f"{list(self.supported_outputs)}."
            )
        if not requested:
            raise ValueError(
                f"Feature '{self.feature_name}' was given an empty 'outputs' "
                f"list, so it would produce no columns. Remove the feature "
                f"instead, or name the outputs to keep."
            )
        return [name for name in self.supported_outputs if name in requested]

    def get_variables(self) -> List[str]:
        """
        Return the input variables to compute the outputs for.

        :return: The ``col_names`` entry of the configuration.
        :rtype: List[str]
        """
        return list((self.feature_info or {}).get("col_names") or [])

    def require_columns(self, df: pl.DataFrame, columns: List[str]) -> None:
        """
        Raise unless every named column is present in the input.

        A configured column that is absent is a configuration error, not a
        reason to emit nulls: silently producing a column of nulls would
        look like missing data rather than a typo.

        :param df: The input observations.
        :type df: pl.DataFrame
        :param columns: The column names the feature needs.
        :type columns: List[str]
        :raises ValueError: If any named column is missing.
        """
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(
                f"Feature '{self.feature_name}' needs the column(s) "
                f"{missing}, which are not in the input. Point the feature's "
                f"params at the columns that hold them, or remove the "
                f"feature from the feature set."
            )

    def output_column_name(self, output: str, variable: Optional[str] = None) -> str:
        """
        Return the emitted column name for an output.

        :param output: The short output name.
        :type output: str
        :param variable: The variable it was computed for, or :obj:`None`
                         for an output that is not variable specific.
        :type variable: Optional[str]
        :return: ``{variable}_{output}`` or ``{output}``.
        :rtype: str
        """
        if variable is None:
            return output
        return f"{variable}_{output}"

    @abstractmethod
    def compute_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute this feature's columns for every row of ``df``.

        ``df`` holds the observations of the selected profiles, sorted into
        vertical order within each profile. The result must have the same
        rows, and must consist of the observation key columns plus one
        column per emitted output.

        :param df: The observations of the selected profiles, sorted.
        :type df: pl.DataFrame
        :return: Observation keys plus the emitted columns.
        :rtype: pl.DataFrame
        """
        pass  # pragma: no cover

    def extract_features(self) -> None:
        """
        Compute the columns and narrow them to the target's selected rows.

        :raises ValueError: If the selected rows are profile-level, since
                            these features are observation-level and must be
                            aggregated by the extract step's ``agg``
                            mechanism instead.
        """
        computed = self.compute_columns(sort_profiles(self.filtered_input))

        if self.selected_rows is None or self.target_name is None:
            self.features = computed
            return

        selected = self.selected_rows[self.target_name]
        if "observation_no" not in selected.columns:
            raise ValueError(
                f"Feature '{self.feature_name}' is observation-level and "
                f"cannot be used directly at profile level; add an 'agg' "
                f"list to its feature parameters so it is aggregated per "
                f"profile, or remove it from the feature set."
            )

        self.features = (
            selected.select(["row_id", *OBSERVATION_KEYS])
            .join(computed, on=OBSERVATION_KEYS, maintain_order="left")
            .drop(OBSERVATION_KEYS)
        )

    def scale_first(self) -> None:
        """
        Pre-extraction scaling, unused by these features.

        The emitted columns are derived quantities rather than the raw
        variables the stats sets describe, so any scaling happens after
        they exist (see :meth:`scale_second`).
        """
        pass  # pragma: no cover

    def scale_second(self) -> None:
        """
        Scale the emitted columns from an explicit ``min_max`` stats set.

        Data-derived normalization (``auto_min_max``, ``standard``) is
        fitted from the step 2 summary table, which has rows only for the
        raw input variables, so it cannot cover these columns. They are
        meant to be used ``raw`` (the z-scores, fractions and flags already
        are comparable across datasets) or with explicit bounds.
        """
        stats_type = self.feature_info.get("stats_set", {}).get("type", "raw")
        if is_scaling_type(stats_type) and self.feature_info.get("stats"):
            self.features = scale_flat_columns(
                self.features, self.feature_info["stats"], stats_type
            )
