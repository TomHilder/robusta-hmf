# warnings.py

"""Warning classes issued by robusta-hmf, and where they point.

All derive from :class:`RobustaWarning`, so every warning the library issues
can be filtered or escalated at once, e.g.
``warnings.simplefilter("error", RobustaWarning)``.
"""

import os
import sys

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def user_stacklevel() -> int:
    """``stacklevel`` for ``warnings.warn`` that points at the first caller outside the package.

    Call it from the function that calls ``warnings.warn``. The library reaches
    a warning through call chains of different depth (dense or sharded frame,
    ``fit`` or ``infer``), so a fixed ``stacklevel`` would point inside it.
    """
    level, frame = 1, sys._getframe(1)
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE_DIR):
        frame = frame.f_back
        level += 1
    return level


class RobustaWarning(UserWarning):
    """Base class for warnings issued by robusta-hmf."""


class DataWarning(RobustaWarning):
    """The input data were modified before fitting, e.g. non-finite entries
    treated as missing."""


class UnderdeterminedWarning(RobustaWarning):
    """Some rows or columns have fewer usable weights than the rank, so their
    factors are not determined by the data."""


class ConvergenceWarning(RobustaWarning):
    """The fit did not converge, or the loss behaved in a way the optimiser
    should not allow."""
