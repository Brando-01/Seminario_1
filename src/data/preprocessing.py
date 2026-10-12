from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TabularPreprocessor:
    """Leakage-safe numeric scaling and categorical frequency encoding."""

    excluded_columns: set[str]
    missing_threshold: float = 0.95
    log_columns: tuple = ()
    clip_value: float | None = None
    add_missing_indicators: bool = False
    categorical_encoding: str = "frequency"
    max_categories: int = 16
    feature_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    frequency_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    missing_indicator_columns: list[str] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        if self.categorical_encoding not in ("frequency", "onehot_frequency") or self.max_categories < 1:
            raise ValueError("Supported categorical encodings: frequency, onehot_frequency; positive max_categories required")
        candidates = [c for c in frame.columns if c not in self.excluded_columns]
        self.feature_columns = [c for c in candidates if frame[c].isna().mean() <= self.missing_threshold]
        self.numeric_columns = [c for c in self.feature_columns if pd.api.types.is_numeric_dtype(frame[c])]
        self.categorical_columns = [c for c in self.feature_columns if c not in self.numeric_columns]
        self.missing_indicator_columns = []
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if self.add_missing_indicators and values.isna().any():
                self.missing_indicator_columns.append(column)
            if column in self.log_columns:
                values = np.sign(values) * np.log1p(np.abs(values))
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
            self.category_levels[column] = values.value_counts().head(self.max_categories).index.tolist()
            if self.add_missing_indicators and frame[column].isna().any():
                self.missing_indicator_columns.append(column)
        return self

    @property
    def output_columns(self) -> list[str]:
        columns = self.numeric_columns.copy()
        for column in self.categorical_columns:
            columns.append(column)
            if self.categorical_encoding == "onehot_frequency":
                columns.extend(f"{column}__category_{index}" for index in range(len(self.category_levels[column])))
        columns.extend(f"{column}__missing" for column in getattr(self, "missing_indicator_columns", []))
        return columns

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            raise RuntimeError("Preprocessor must be fitted before transform")
        arrays: list[np.ndarray] = []
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if column in self.log_columns:
                values = np.sign(values) * np.log1p(np.abs(values))
            values = values.fillna(self.medians[column])
            scaled = (values.to_numpy(dtype=np.float32) - self.means[column]) / self.stds[column]
            if self.clip_value is not None:
                scaled = np.clip(scaled, -self.clip_value, self.clip_value)
            arrays.append(scaled.reshape(-1, 1))
        for column in self.categorical_columns:
            values = frame[column].astype("string").fillna("__MISSING__")
            encoded = values.map(self.frequency_maps[column]).fillna(0.0).to_numpy(dtype=np.float32)
            arrays.append(encoded.reshape(-1, 1))
            if self.categorical_encoding == "onehot_frequency":
                for level in self.category_levels[column]:
                    arrays.append(values.eq(level).to_numpy(dtype=np.float32).reshape(-1, 1))
        for column in getattr(self, "missing_indicator_columns", []):
            values = frame[column]
            if column in self.numeric_columns:
                values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
            arrays.append(values.isna().to_numpy(dtype=np.float32).reshape(-1, 1))
        if not arrays:
            return np.empty((len(frame), 0), dtype=np.float32)
        return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)
