# warnings.py

"""Warning classes issued by robusta-hmf.

All derive from :class:`RobustaWarning`, so every warning the library issues
can be filtered or escalated at once, e.g.
``warnings.simplefilter("error", RobustaWarning)``.
"""


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
