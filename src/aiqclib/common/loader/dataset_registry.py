"""
Module providing registry dictionaries that map dataset class names (str) to their
corresponding Python classes. These registries enable dynamic loading of
the correct class during each preparation step in the pipeline.
"""

from typing import Dict, Type

from aiqclib.prepare.step1_read_input.dataset_a import InputDataSetA
from aiqclib.prepare.step1_read_input.input_base import InputDataSetBase
from aiqclib.prepare.step2_calc_stats.dataset_a import SummaryDataSetA
from aiqclib.prepare.step2_calc_stats.summary_base import SummaryStatsBase
from aiqclib.prepare.step3_select_profiles.dataset_a import SelectDataSetA
from aiqclib.prepare.step3_select_profiles.dataset_all import SelectDataSetAll
from aiqclib.prepare.step3_select_profiles.select_base import ProfileSelectionBase
from aiqclib.prepare.step4_select_rows.dataset_a import LocateDataSetA
from aiqclib.prepare.step4_select_rows.dataset_all import LocateDataSetAll
from aiqclib.prepare.step4_select_rows.locate_base import LocatePositionBase
from aiqclib.prepare.step5_extract_features.dataset_a import ExtractDataSetA
from aiqclib.prepare.step5_extract_features.extract_base import ExtractFeatureBase
from aiqclib.prepare.step6_split_dataset.dataset_a import SplitDataSetA
from aiqclib.prepare.step6_split_dataset.dataset_all import SplitDataSetAll
from aiqclib.prepare.step6_split_dataset.split_base import SplitDataSetBase

#: A registry mapping class names (used in YAML config) to their corresponding
#: Python classes for step1_read_input tasks.
#:
#: :type: Dict[str, Type[InputDataSetBase]]
INPUT_DATASET_REGISTRY: Dict[str, Type[InputDataSetBase]] = {
    "InputDataSetA": InputDataSetA,
}

#: A registry mapping class names (used in YAML config) to their corresponding
#: Python classes for step2_calc_stats tasks.
#:
#: :type: Dict[str, Type[SummaryStatsBase]]
SUMMARY_DATASET_REGISTRY: Dict[str, Type[SummaryStatsBase]] = {
    "SummaryDataSetA": SummaryDataSetA,
}

#: A registry mapping class names (used in YAML config) to their corresponding
#: Python classes for step3_select_profiles tasks.
#:
#: :type: Dict[str, Type[ProfileSelectionBase]]
SELECT_DATASET_REGISTRY: Dict[str, Type[ProfileSelectionBase]] = {
    "SelectDataSetA": SelectDataSetA,
    "SelectDataSetAll": SelectDataSetAll,
}

#: A registry mapping class names (used in YAML config) to their corresponding
#: Python classes for step4_select_rows tasks.
#:
#: :type: Dict[str, Type[LocatePositionBase]]
LOCATE_DATASET_REGISTRY: Dict[str, Type[LocatePositionBase]] = {
    "LocateDataSetA": LocateDataSetA,
    "LocateDataSetAll": LocateDataSetAll,
}

#: A registry mapping class names (used in YAML config) to their corresponding
#: Python classes for step5_extract_features tasks.
#:
#: :type: Dict[str, Type[ExtractFeatureBase]]
EXTRACT_DATASET_REGISTRY: Dict[str, Type[ExtractFeatureBase]] = {
    "ExtractDataSetA": ExtractDataSetA,
}

#: A registry mapping class names (used in YAML config) to their corresponding
#: Python classes for step6_split_dataset tasks.
#:
#: :type: Dict[str, Type[SplitDataSetBase]]
SPLIT_DATASET_REGISTRY: Dict[str, Type[SplitDataSetBase]] = {
    "SplitDataSetA": SplitDataSetA,
    "SplitDataSetAll": SplitDataSetAll,
}
