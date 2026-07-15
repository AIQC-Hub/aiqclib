"""
Module providing registry dictionaries that map NRT QC class names (str) to
their corresponding Python classes. These registries enable dynamic loading
of the correct class during each step of the NRT QC pipeline.
"""

from typing import Dict, Type

from aiqclib.nrtqc.step1_read_input.dataset_all import InputDataSetAll
from aiqclib.nrtqc.step2_run_qc.dataset_all import QCDataSetAll
from aiqclib.nrtqc.step2_run_qc.qc_base import QCDataSetBase
from aiqclib.nrtqc.step3_concat_flags.concat_base import ConcatFlagsBase
from aiqclib.nrtqc.step3_concat_flags.dataset_all import ConcatDataSetAll
from aiqclib.prepare.step1_read_input.input_base import InputDataSetBase

#: A registry mapping class names (as strings, typically from YAML
#: configuration) to their corresponding Python classes for
#: step1_read_input tasks in the NRT QC pipeline.
#:
#: :type: Dict[str, Type[InputDataSetBase]]
INPUT_NRTQC_REGISTRY: Dict[str, Type[InputDataSetBase]] = {
    "InputDataSetAll": InputDataSetAll,
}

#: A registry mapping class names (as strings, typically from YAML
#: configuration) to their corresponding Python classes for step2_run_qc
#: tasks in the NRT QC pipeline.
#:
#: :type: Dict[str, Type[QCDataSetBase]]
QC_NRTQC_REGISTRY: Dict[str, Type[QCDataSetBase]] = {
    "QCDataSetAll": QCDataSetAll,
}

#: A registry mapping class names (as strings, typically from YAML
#: configuration) to their corresponding Python classes for
#: step3_concat_flags tasks in the NRT QC pipeline.
#:
#: :type: Dict[str, Type[ConcatFlagsBase]]
CONCAT_NRTQC_REGISTRY: Dict[str, Type[ConcatFlagsBase]] = {
    "ConcatDataSetAll": ConcatDataSetAll,
}
