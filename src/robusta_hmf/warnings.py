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
    """Base class for warnings issued by robusta-hmf.

    Filter on it to act on all of them at once, e.g.
    ``warnings.simplefilter("error", RobustaWarning)`` turns every
    robusta-hmf warning into an exception.
    """


class DataWarning(RobustaWarning):
    """The input data were modified before use.

    Issued by :meth:`~robusta_hmf.Robusta.fit`, :meth:`~robusta_hmf.Robusta.infer`
    and :meth:`~robusta_hmf.Robusta.score` (or ``prepare()`` on a frame) when
    entries of ``Y`` or ``W`` are NaN or infinite: those entries are treated
    as missing (``Y = W = 0``), and the message says how many there were.
    Pass ``nan_error=True`` to raise ``ValueError`` instead.
    """


class UnderdeterminedWarning(RobustaWarning):
    """Some rows or columns have fewer usable weights than the rank.

    Their factors are not determined by the data: a row of ``A`` (or of
    ``G``) has ``rank`` unknowns and needs at least that many non-zero
    weights. Without ``als_ridge`` these rows and columns are excluded from
    the fit and returned as NaN (see
    :attr:`FitInfo.exclusions <robusta_hmf.FitInfo>`); with ``als_ridge``
    they are kept, determined only by the ridge, and the warning names them.
    """


class ConvergenceWarning(RobustaWarning):
    """The fit did not converge, or its loss rose when it should not.

    Issued when a fit or inference reaches ``max_iter`` before the
    convergence test passes, and (for ALS without a ridge) when the loss
    rises by more than floating-point round-off allows, which in exact
    arithmetic it never does. A rise usually means a badly conditioned
    least-squares solve: try float64, a smaller rank, or check ``W``.
    """
