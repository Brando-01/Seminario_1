from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(frame: pd.DataFrame, time_column: str, train_fraction: float, validation_fraction: float, train_before=None, test_from=None) -> TemporalSplit:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test partition")
    ordered = frame.sort_values(time_column, kind="stable").reset_index(drop=True)
    if ordered[time_column].isna().any() or ordered[time_column].nunique() < 3:
        raise ValueError("At least three distinct nonmissing timestamps are required")
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))
    times = ordered[time_column]
    train_cut = times.iloc[train_end] if train_before is None else train_before
    test_cut = times.iloc[validation_end] if test_from is None else test_from
    train_end = int(times.searchsorted(train_cut, side="left"))
    validation_end = int(times.searchsorted(test_cut, side="left"))
    if not 0 < train_end < validation_end < len(ordered):
        raise ValueError("Temporal boundaries must produce three nonempty partitions")
    return TemporalSplit(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        test=ordered.iloc[validation_end:].copy(),
    )
