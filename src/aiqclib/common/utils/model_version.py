"""
Which library version wrote a model file, and whether the difference matters.

Model files are joblib pickles of a fitted estimator. XGBoost makes no
compatibility promise about those, so it warns whenever one is unpickled by a
version other than the one that wrote it. That warning is unhelpful in two
ways. It arrives from inside :mod:`pickle` without naming a file, so a run
loading one model per target has nothing to act on; and its wording ("an older
version of XGBoost") reads as a claim about the file, when in fact it fires in
both directions and on any difference at all, patch releases included.

The difference is worth knowing about, but only in one direction. Measured
across six releases (2.1.4 to 3.4.0, every pair), on a 200-tree model scored
over 2,000 rows:

* Loading into a **newer** XGBoost than trained the model reproduced the
  trained predictions bitwise, with SHAP values agreeing to within float32
  rounding.
* Loading into an **older** one did not, and did not fail either: scores moved
  by up to 0.076 and 21 of the 2,000 labels flipped at a 0.5 threshold.

So the two cases deserve different treatment, and telling them apart needs the
version that wrote the file. That is not recoverable from the model itself:
``Booster.save_config`` reports the running version after loading, not the
file's. :func:`stamp_model_version` therefore records it on the estimator at
save time, where it is pickled along with everything else, and
:func:`load_model_file` reads it back:

* Stamp present and this environment is the same or newer: a notice, since the
  predictions are the ones the training run produced.
* Stamp present and this environment is older: a warning naming both versions.
* No stamp, meaning a file written before this was added: the warning without
  the versions, since the direction cannot be checked.

Each is reported once per run. The cause is the environment rather than any one
file, and a run classifying several datasets would otherwise repeat the same
paragraph for every model it loads.
"""

import warnings
from typing import Any, Optional, Tuple

from joblib import load

from aiqclib.common.utils.progress import notice

#: Attribute the writing XGBoost version is recorded under. Set on the fitted
#: estimator rather than wrapped around it, so a model file stays a plain
#: pickled estimator that ``joblib.load`` returns directly. Private by name
#: because it is this library's bookkeeping, not part of XGBoost's API.
STAMP_ATTRIBUTE: str = "_aiqclib_xgboost_version"

#: Fragments identifying XGBoost's own complaint about unpickling a model that
#: a different version serialized. Matched on the text because there is nothing
#: else to match on: the message is routed out of the C++ logger by
#: ``xgboost.core._log_callback``, which raises it as a plain ``UserWarning``
#: from inside :mod:`pickle`. Both fragments must be present, so an unrelated
#: XGBoost warning is passed through untouched.
_XGBOOST_SERIALIZATION_MARKERS: Tuple[str, ...] = (
    "serialized model",
    "older version of xgboost",
)

#: Severities already reported in this process, one entry per kind. Kept apart
#: so a run loading a safe model and a risky one still hears about the risky
#: one, rather than having the first load silence the second.
_reported: set = set()


def _installed_xgboost() -> Optional[str]:
    """
    Return the installed XGBoost version, or None if it cannot be determined.

    :return: A version string such as ``"3.4.0"``.
    :rtype: Optional[str]
    """
    try:
        import xgboost

        return str(xgboost.__version__)
    except Exception:  # pragma: no cover - xgboost is a hard dependency
        return None


def _is_xgboost_model(model: Any) -> bool:
    """
    Report whether an estimator comes from XGBoost.

    Checked by module rather than by importing XGBoost and testing the class,
    so nothing is imported for the sake of a model that is not one.

    :param model: The fitted estimator about to be saved.
    :type model: Any
    :return: True for ``xgboost.*`` classes.
    :rtype: bool
    """
    return type(model).__module__.split(".")[0] == "xgboost"


def _as_tuple(version: Optional[str]) -> Optional[Tuple[int, ...]]:
    """
    Parse a version string into comparable integers.

    Anything that is not a plain dotted number (a release candidate, a
    development build) yields None, so the caller falls back to the message
    that makes no ordering claim rather than guessing at one.

    :param version: A version string such as ``"3.4.0"``.
    :type version: Optional[str]
    :return: The numeric components, or None if they cannot be read.
    :rtype: Optional[Tuple[int, ...]]
    """
    if not version:
        return None
    parts = version.split(".")
    if not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def stamp_model_version(model: Any) -> Optional[str]:
    """
    Record the writing XGBoost version on an estimator about to be saved.

    Called from :meth:`ModelBase.save_model` immediately before the dump, so
    the value is pickled with the model and is available when it is loaded
    again, possibly in another environment. Only XGBoost models are stamped:
    they are the ones whose loader raises a version warning this can answer.

    Refreshed on every save rather than written once, so re-saving a model
    under a different XGBoost records the version that actually wrote the file.

    :param model: The fitted estimator about to be serialized.
    :type model: Any
    :return: The version recorded, or None when nothing was stamped.
    :rtype: Optional[str]
    """
    if model is None or not _is_xgboost_model(model):
        return None

    version = _installed_xgboost()
    if version is None:
        return None

    try:
        setattr(model, STAMP_ATTRIBUTE, version)
    except Exception:  # pragma: no cover - estimators accept attributes
        return None
    return version


def read_model_stamp(model: Any) -> Optional[str]:
    """
    Return the XGBoost version recorded on a loaded model, if any.

    :param model: The estimator just deserialized.
    :type model: Any
    :return: The version string written at save time, or None for a file
             written before stamping existed.
    :rtype: Optional[str]
    """
    version = getattr(model, STAMP_ATTRIBUTE, None)
    return str(version) if version else None


def _is_xgboost_serialization_warning(message: str) -> bool:
    """
    Recognize XGBoost's warning about a model pickled by another version.

    :param message: The warning text as rendered by :class:`str`.
    :type message: str
    :return: True when every marker in
             :data:`_XGBOOST_SERIALIZATION_MARKERS` is present.
    :rtype: bool
    """
    text = message.lower()
    return all(marker in text for marker in _XGBOOST_SERIALIZATION_MARKERS)


def report_model_version(file_name: str, saved_version: Optional[str]) -> str:
    """
    Report a version difference at the severity the direction deserves.

    See the module docstring for the measurements behind the split. In short,
    an environment at least as new as the one that trained a model reproduces
    its predictions, and one older than it can change them silently.

    :param file_name: The model file that was loaded.
    :type file_name: str
    :param saved_version: The version recorded at save time, or None when the
                          file carries no stamp.
    :type saved_version: Optional[str]
    :return: What was reported: ``"note"``, ``"warning"``, ``"unknown"``, or
             ``"repeat"`` when this severity was already reported.
    :rtype: str
    """
    installed = _installed_xgboost()
    saved = _as_tuple(saved_version)
    running = _as_tuple(installed)

    if saved is None or running is None:
        return _report_unknown(file_name)

    if saved <= running:
        return _report_safe(file_name, saved_version, installed)

    return _report_risky(file_name, saved_version, installed)


def _report_safe(file_name: str, saved: Optional[str], installed: str) -> str:
    """
    Note a version difference in the direction that preserves predictions.

    A notice rather than a warning: XGBoost raised one, but this library has
    checked the direction it could not, and the answer is that the run is fine.

    :param file_name: The model file that was loaded.
    :type file_name: str
    :param saved: The version that wrote the file.
    :type saved: Optional[str]
    :param installed: The version loading it.
    :type installed: str
    :return: ``"note"``, or ``"repeat"`` if already reported.
    :rtype: str
    """
    if "note" in _reported:
        return "repeat"
    _reported.add("note")

    notice(
        f"The model in '{file_name}' was trained under XGBoost {saved} and "
        f"this environment has XGBoost {installed}. XGBoost warns about any "
        f"difference between the two, because model files are pickles and it "
        f"promises nothing about those. Loading into a newer version is the "
        f"safe direction: in testing it reproduced the trained predictions "
        f"exactly. Nothing to do. Reported once per run."
    )
    return "note"


def _report_risky(file_name: str, saved: Optional[str], installed: str) -> str:
    """
    Warn about a version difference that can change predictions in silence.

    :param file_name: The model file that was loaded.
    :type file_name: str
    :param saved: The version that wrote the file.
    :type saved: Optional[str]
    :param installed: The older version loading it.
    :type installed: str
    :return: ``"warning"``, or ``"repeat"`` if already reported.
    :rtype: str
    """
    if "warning" in _reported:
        return "repeat"
    _reported.add("warning")

    warnings.warn(
        f"The model in '{file_name}' was trained under XGBoost {saved}, but "
        f"this environment has the older XGBoost {installed}. Loading a model "
        f"into an XGBoost older than the one that trained it can change its "
        f"predictions without failing: in testing it moved scores by up to "
        f"0.076 and flipped 21 of 2,000 labels at a 0.5 threshold. Upgrade "
        f"this environment to at least XGBoost {saved}, or retrain with the "
        f"installed version. Reported once per run: the cause is the "
        f"environment, not this one file.",
        UserWarning,
        stacklevel=4,
    )
    return "warning"


def _report_unknown(file_name: str) -> str:
    """
    Warn about a version difference whose direction cannot be established.

    Reached for a model file written before stamping existed, and for a
    version string that does not parse as a plain dotted number. Both leave
    the ordering unknown, and the direction is the whole question, so this
    says what to check rather than guessing.

    :param file_name: The model file that was loaded.
    :type file_name: str
    :return: ``"unknown"``, or ``"repeat"`` if already reported.
    :rtype: str
    """
    if "unknown" in _reported:
        return "repeat"
    _reported.add("unknown")

    installed = _installed_xgboost() or "the installed XGBoost"
    warnings.warn(
        f"The model in '{file_name}' was written by a different XGBoost "
        f"version from the XGBoost {installed} in this environment, and the "
        f"file does not record which one, so the difference cannot be judged "
        f"here. It matters in one direction only: a model loaded into an "
        f"XGBoost older than the one that trained it can have its predictions "
        f"changed silently, while a newer one reproduces them. Models saved "
        f"from this version onwards record the version that wrote them; to "
        f"settle it for this file, retrain the target with the installed "
        f"version. Reported once per run.",
        UserWarning,
        stacklevel=4,
    )
    return "unknown"


def load_model_file(file_name: str) -> Any:
    """
    Deserialize a model file, answering XGBoost's version warning if it fires.

    The warning has to be caught here rather than filtered by the caller: it is
    raised during deserialization, when the file name is still known, and the
    stamp that decides its severity is only readable once the model is back.
    Every other warning raised by the load is re-emitted unchanged, so this
    narrows the output rather than filtering it.

    Warnings raised before an exception are dropped along with the block: a
    failed load has more urgent news than a version remark.

    .. note::

       :func:`warnings.catch_warnings` replaces the global filter state for
       the duration, so this is not safe to run in parallel with other threads
       that depend on it. Model files are loaded one at a time.

    :param file_name: The model file to read.
    :type file_name: str
    :return: The deserialized estimator.
    :rtype: Any
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = load(file_name)

    mismatch = False
    for entry in caught:
        if _is_xgboost_serialization_warning(str(entry.message)):
            mismatch = True
            continue
        warnings.warn_explicit(
            entry.message, entry.category, entry.filename, entry.lineno
        )

    if mismatch:
        report_model_version(file_name, read_model_stamp(model))

    return model
