"""
Deprecation notice for the NRT QC module, which is moving to ``nrtqclib``.

The NRT QC pipeline, its configuration and its batch phase move to the
separate ``nrtqclib`` package (which depends on ``aiqclib`` for the QC items).
The ``aiqclib`` copies keep working until they are removed in 0.15.0; each
public entry point calls :func:`warn_moved` so callers see the change coming.
"""

import warnings

#: The ``aiqclib`` release that removes the NRT QC module.
REMOVAL_VERSION: str = "0.15.0"


def warn_moved(name: str, replacement: str, stacklevel: int = 3) -> None:
    """
    Emit a :class:`DeprecationWarning` for an NRT QC entry point.

    The default ``stacklevel`` attributes the warning to the code calling the
    entry point, so Python's default filters show it to a user script but not
    when one ``aiqclib`` function calls another.

    :param name: The deprecated entry point, as a user would write it.
    :type name: str
    :param replacement: The ``nrtqclib`` equivalent.
    :type replacement: str
    :param stacklevel: Passed to :func:`warnings.warn`; 3 points at the
                       caller of the function calling this one.
    :type stacklevel: int
    :return: None
    :rtype: None
    """
    warnings.warn(
        f"{name} is moving to the nrtqclib package and will be removed from "
        f"aiqclib in {REMOVAL_VERSION}. Install nrtqclib and use "
        f"{replacement} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
