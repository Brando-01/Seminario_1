from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TabularPreprocessor:
    """Leakage-safe numeric scaling and categorical frequency encoding."""

    excluded_columns: set[str]
    missing_threshold: float = 0.95
    feature_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    frequency_maps: dict[str, dict[str, float]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        candidates = [c for c in frame.columns if c not in self.excluded_columns]
        self.feature_columns = [c for c in candidates if frame[c].isna().mean() <= self.missing_threshold]
        self.numeric_columns = [c for c in self.feature_columns if pd.api.types.is_numeric_dtype(frame[c])]
        self.categorical_columns = [c for c in self.feature_columns if c not in self.numeric_columns]
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.medians[column] = median
            self.means[column] = mean
            self.stds[column] = std if std > 1e-12 else 1.0
        for column in self.categorical_columns:
            values = frame[column].astype("string").fillna("__MISSING__")
            self.frequency_maps[column] = values.value_counts(normalize=True).to_dict()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            raise RuntimeError("Preprocessor must be fitted before transform")
        arrays: list[np.ndarray] = []
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(self.medians[column])
            arrays.append(((values.to_numpy(dtype=np.float32) - self.means[column]) / self.stds[column]).reshape(-1, 1))
        for column in self.categorical_columns:
            values = frame[column].astype("string").fillna("__MISSING__")
            encoded = values.map(self.frequency_maps[column]).fillna(0.0).to_numpy(dtype=np.float32)
            arrays.append(encoded.reshape(-1, 1))
        if not arrays:
            return np.empty((len(frame), 0), dtype=np.float32)
        return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

