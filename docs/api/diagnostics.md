# Diagnostics, warnings and errors

`model.fit_info` records how the last `fit()` ran. Warnings all derive from
`RobustaWarning`, so `warnings.simplefilter("error", RobustaWarning)` turns every one of
them into an error.

## FitInfo

::: robusta_hmf.FitInfo

## ExclusionEvent

::: robusta_hmf.ExclusionEvent

## Warnings

::: robusta_hmf.RobustaWarning

::: robusta_hmf.DataWarning

::: robusta_hmf.UnderdeterminedWarning

::: robusta_hmf.ConvergenceWarning

## Errors

::: robusta_hmf.PrecisionError
