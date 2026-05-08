"""
Module providing registry dictionaries that map dataset class names (str) to their
corresponding Python classes. These registries enable dynamic loading of
the correct class during each preparation step in the pipeline.
"""

from typing import Dict, Type

from aiqclib.classify.step1_read_input.dataset_all import InputDataSetAll
from aiqclib.classify.step2_calc_stats.dataset_all import SummaryDataSetAll
from aiqclib.classify.step3_select_profiles.dataset_all import SelectDataSetAll
from aiqclib.classify.step4_select_rows.dataset_all import LocateDataSetAll
from aiqclib.classify.step5_extract_features.dataset_all import ExtractDataSetAll
from aiqclib.classify.step6_classify_dataset.dataset_all import ClassifyAll
from aiqclib.classify.step6_classify_dataset.dataset_all_suite import ClassifyAllSuite
from aiqclib.classify.step7_concat_datasets.concat_base import ConcatDatasetsBase
from aiqclib.classify.step7_concat_datasets.dataset_all import ConcatDataSetAll
from aiqclib.classify.step7_concat_datasets.dataset_suite import ConcatDataSetSuite
from aiqclib.prepare.step1_read_input.input_base import InputDataSetBase
from aiqclib.prepare.step2_calc_stats.summary_base import SummaryStatsBase
from aiqclib.prepare.step3_select_profiles.select_base import ProfileSelectionBase
from aiqclib.prepare.step4_select_rows.locate_base import LocatePositionBase
from aiqclib.prepare.step5_extract_features.extract_base import ExtractFeatureBase
from aiqclib.train.step4_build_model.build_model_base import BuildModelBase

#: A registry mapping class names (as strings, typically from YAML configuration)
#: to their corresponding Python classes for step1_read_input tasks in the
#: classification pipeline.
#:
#: :type: Dict[str, Type[InputDataSetBase]]
INPUT_CLASSIFY_REGISTRY: Dict[str, Type[InputDataSetBase]] = {
    "InputDataSetAll": InputDataSetAll,
}

#: A registry mapping class names (as strings, typically from YAML configuration)
#: to their corresponding Python classes for step2_calc_stats tasks in the
#: classification pipeline.
#:
#: :type: Dict[str, Type[SummaryStatsBase]]
SUMMARY_CLASSIFY_REGISTRY: Dict[str, Type[SummaryStatsBase]] = {
    "SummaryDataSetAll": SummaryDataSetAll,
}

#: A registry mapping class names (as strings, typically from YAML configuration)
#: to their corresponding Python classes for step3_select_profiles tasks in the
#: classification pipeline.
#:
#: :type: Dict[str, Type[ProfileSelectionBase]]
SELECT_CLASSIFY_REGISTRY: Dict[str, Type[ProfileSelectionBase]] = {
    "SelectDataSetAll": SelectDataSetAll,
}

#: A registry mapping class names (as strings, typically from YAML configuration)
#: to their corresponding Python classes for step4_select_rows tasks in the
#: classification pipeline.
#:
#: :type: Dict[str, Type[LocatePositionBase]]
LOCATE_CLASSIFY_REGISTRY: Dict[str, Type[LocatePositionBase]] = {
    "LocateDataSetAll": LocateDataSetAll,
}

#: A registry mapping class names (as strings, typically from YAML configuration)
#: to their corresponding Python classes for step5_extract_features tasks in the
#: classification pipeline.
#:
#: :type: Dict[str, Type[ExtractFeatureBase]]
EXTRACT_CLASSIFY_REGISTRY: Dict[str, Type[ExtractFeatureBase]] = {
    "ExtractDataSetAll": ExtractDataSetAll,
}

#: A registry mapping class names (as strings, typically from YAML configuration)
#: to their corresponding Python classes for step6_classify_dataset tasks in the
#: classification pipeline.
#:
#: :type: Dict[str, Type[BuildModelBase]]
CLASSIFY_CLASSIFY_REGISTRY: Dict[str, Type[BuildModelBase]] = {
    "ClassifyAll": ClassifyAll,
    "ClassifyAllSuite": ClassifyAllSuite,
}

#: A registry mapping class names (as strings, typically from YAML configuration)
#: to their corresponding Python classes for step7_concat_datasets tasks in the
#: classification pipeline.
#:
#: :type: Dict[str, Type[ConcatDatasetsBase]]
CLASSIFY_CONCAT_REGISTRY: Dict[str, Type[ConcatDatasetsBase]] = {
    "ConcatDataSetAll": ConcatDataSetAll,
    "ConcatDataSetSuite": ConcatDataSetSuite,
}
