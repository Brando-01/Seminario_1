from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DatasetBundle:
    """Dataset plus the semantic columns needed by the experiment."""

    name: str
    frame: pd.DataFrame
    target_column: str
    time_column: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        required = {self.target_column, self.time_column}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if self.frame[self.target_column].isna().any():
            raise ValueError("Target column contains missing values")

