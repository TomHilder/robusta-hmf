# Sharded fitting

Row-chunked, multi-device fitting for data larger than one GPU. Pass a `ShardedFrame` to
`Robusta(override_frame=...)`; everything else works as for a dense fit.

## ShardedFrame

::: robusta_hmf.ShardedFrame
    options:
      members:
        - prepare
        - n_devices

## ShardedData

::: robusta_hmf.ShardedData

## RowPlan

::: robusta_hmf.RowPlan

## GramInitialiser

::: robusta_hmf.GramInitialiser
    options:
      members: false
