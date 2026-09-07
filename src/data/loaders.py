from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import DatasetBundle


BALANCE_COLUMNS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def add_paysim_causal_features(frame: pd.DataFrame, time_column: str = "step") -> pd.DataFrame:
    """Create row features using only the current transaction and earlier rows."""
    enriched = frame.sort_values(time_column, kind="stable").reset_index(drop=True).copy()
    amount = pd.to_numeric(enriched["amount"], errors="coerce").fillna(0.0).astype("float64")
    enriched["amount_log1p"] = np.log1p(amount.clip(lower=0))
    enriched["destination_is_merchant"] = enriched["nameDest"].astype("string").str.startswith("M", na=False).astype("int8")
    for role, entity_column in (("origin", "nameOrig"), ("destination", "nameDest")):
        grouped = enriched.groupby(entity_column, sort=False, dropna=False)
        # Aggregate history by complete timestamp: ties cannot observe one another.
        buckets = enriched.assign(_amount=amount).groupby([entity_column, time_column], sort=True, dropna=False).agg(_sum=("_amount", "sum"), _count=("_amount", "size")).reset_index()
        accounts = buckets.groupby(entity_column, sort=False, dropna=False)
        buckets["_previous_count"] = accounts["_count"].cumsum() - buckets["_count"]
        buckets["_previous_sum"] = accounts["_sum"].cumsum() - buckets["_sum"]
        buckets["_recency"] = accounts[time_column].diff().fillna(-1)
        history = enriched[[entity_column, time_column]].merge(buckets, on=[entity_column, time_column], how="left", validate="many_to_one")
        previous_count = history["_previous_count"].astype("int64")
        previous_sum = history["_previous_sum"].astype("float64")
        previous_mean = previous_sum.div(previous_count.where(previous_count > 0)).fillna(0.0)
        time_since_previous = history["_recency"]
        enriched[f"{role}_previous_count_log1p"] = np.log1p(previous_count).astype("float32")
        enriched[f"{role}_previous_amount_sum_log1p"] = np.log1p(previous_sum.clip(lower=0)).astype("float32")
        enriched[f"{role}_previous_amount_mean_log1p"] = np.log1p(previous_mean.clip(lower=0)).astype("float32")
        enriched[f"{role}_time_since_previous"] = time_since_previous.astype("float32")
        enriched[f"{role}_is_new"] = (previous_count == 0).astype("int8")
        enriched[f"{role}_amount_vs_previous_mean"] = np.log1p(amount).sub(np.log1p(previous_mean.clip(lower=0))).astype("float32")
    return enriched


def add_paysim_balance_consistency_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add domain features without retaining the four raw balance columns."""
    enriched = frame.copy()
    amount = pd.to_numeric(enriched["amount"], errors="coerce").fillna(0.0)
    old_origin = pd.to_numeric(enriched["oldbalanceOrg"], errors="coerce").fillna(0.0)
    new_origin = pd.to_numeric(enriched["newbalanceOrig"], errors="coerce").fillna(0.0)
    old_destination = pd.to_numeric(enriched["oldbalanceDest"], errors="coerce").fillna(0.0)
    new_destination = pd.to_numeric(enriched["newbalanceDest"], errors="coerce").fillna(0.0)
    expected_origin = np.where(enriched["type"].eq("CASH_IN"), old_origin + amount, old_origin - amount)
    origin_error = np.abs(new_origin - expected_origin)
    destination_error = np.abs(new_destination - (old_destination + amount))
    enriched["origin_balance_error_log1p"] = np.log1p(origin_error).astype("float32")
    enriched["destination_balance_error_log1p"] = np.log1p(destination_error).astype("float32")
    enriched["origin_balance_zero_after"] = new_origin.eq(0).astype("int8")
    enriched["destination_balance_zero_before"] = old_destination.eq(0).astype("int8")
    return enriched


def _resolve_existing(project_root: Path, primary: str, fallback: str | None = None) -> Path:
    primary_path = (project_root / primary).resolve()
    if primary_path.is_file():
        return primary_path
    if fallback:
        fallback_path = (project_root / fallback).resolve()
        if fallback_path.is_file():
            return fallback_path
    raise FileNotFoundError(f"Dataset file not found. Expected: {primary_path}")


def _read_csv(
    path: Path,
    max_rows: int | None,
    sample_mode: str,
    total_rows: int,
    seed: int,
    time_column: str | None = None,
    development_window: dict | None = None,
) -> pd.DataFrame:
    if max_rows is None and sample_mode != "window":
        return pd.read_csv(path, low_memory=False)
    if sample_mode == "prefix":
        return pd.read_csv(path, nrows=max_rows, low_memory=False)
    if sample_mode == "window":
        if not time_column or not development_window:
            raise ValueError("Window sampling requires time_column and development_window in the dataset config")
        selected_chunks: list[pd.DataFrame] = []
        lower = development_window["min"]
        upper = development_window["max"]
        selected_rows = 0
        for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
            selected = chunk.loc[chunk[time_column].between(lower, upper)]
            if max_rows is not None:
                remaining = max_rows - selected_rows
                selected = selected.iloc[:remaining]
            if not selected.empty:
                selected_chunks.append(selected)
                selected_rows += len(selected)
            if max_rows is not None and selected_rows >= max_rows:
                break
        if not selected_chunks:
            raise ValueError(f"No rows found in development window {lower}..{upper}")
        return pd.concat(selected_chunks).reset_index(drop=True)
    if sample_mode != "uniform":
        raise ValueError(f"Unsupported sample mode: {sample_mode}")
    fraction = min(1.0, 1.20 * max_rows / max(total_rows, 1))
    sampled_chunks: list[pd.DataFrame] = []
    for chunk_number, chunk in enumerate(pd.read_csv(path, chunksize=100_000, low_memory=False)):
        sampled = chunk.sample(frac=fraction, random_state=seed + chunk_number)
        if not sampled.empty:
            sampled_chunks.append(sampled)
    frame = pd.concat(sampled_chunks).sort_index()
    if len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed).sort_index()
    return frame.reset_index(drop=True)


def _composite_key(frame: pd.DataFrame, columns: list[str], minimum_present: int = 2) -> pd.Series:
    available = [c for c in columns if c in frame.columns]
    if not available:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    values = frame[available].astype("string")
    present = values.notna().sum(axis=1)
    key = values.fillna("?").agg("|".join, axis=1)
    return key.where(present >= minimum_present).astype("string")


def load_ieee(config: dict, project_root: Path, max_rows: int | None = None, sample_mode: str = "prefix") -> DatasetBundle:
    data_cfg = config["data"]
    transaction_path = _resolve_existing(project_root, data_cfg["transaction_path"], data_cfg.get("fallback_transaction_path"))
    identity_path = _resolve_existing(project_root, data_cfg["identity_path"], data_cfg.get("fallback_identity_path"))
    seed = int(config["training"].get("seed", 42))
    transactions = _read_csv(transaction_path, max_rows, sample_mode, int(data_cfg["total_rows"]), seed, data_cfg["time_column"], data_cfg.get("development_window"))
    identity = pd.read_csv(identity_path, low_memory=False)
    identity.columns = [c.replace("-", "_") if c.startswith("id-") else c for c in identity.columns]
    frame = transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    hour = (frame[data_cfg["time_column"]].astype("int64") // 3600) % 24
    derived = pd.DataFrame(
        {
            "card_proxy": _composite_key(frame, ["card1", "card2", "card3", "card5", "addr1"], 3),
            "card_email_proxy": _composite_key(frame, ["card1", "addr1", "P_emaildomain"], 2),
            "device_proxy": _composite_key(frame, ["DeviceInfo", "id_31", "id_30", "id_33"], 2),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "elapsed_days": frame[data_cfg["time_column"]].astype("float64") / 86400,
        },
        index=frame.index,
    )
    frame = pd.concat([frame, derived], axis=1)
    bundle = DatasetBundle(
        name="ieee_cis",
        frame=frame,
        target_column=data_cfg["target"],
        time_column=data_cfg["time_column"],
        metadata={"transaction_path": str(transaction_path), "identity_path": str(identity_path), "sample_mode": sample_mode},
    )
    bundle.validate()
    return bundle


def load_paysim(config: dict, project_root: Path, max_rows: int | None = None, sample_mode: str = "prefix") -> DatasetBundle:
    data_cfg = config["data"]
    path = _resolve_existing(project_root, data_cfg["path"], data_cfg.get("fallback_path"))
    seed = int(config["training"].get("seed", 42))
    frame = _read_csv(path, max_rows, sample_mode, int(data_cfg["total_rows"]), seed, data_cfg["time_column"], data_cfg.get("development_window"))
    if data_cfg.get("add_causal_history_features", True):
        frame = add_paysim_causal_features(frame, data_cfg["time_column"])
    if data_cfg.get("add_balance_consistency_features", True):
        frame = add_paysim_balance_consistency_features(frame)
    dropped: list[str] = []
    if data_cfg.get("exclude_balance_columns", True):
        dropped.extend(c for c in BALANCE_COLUMNS if c in frame.columns)
    elif data_cfg.get("exclude_post_transaction_balances", False):
        dropped.extend(c for c in ("newbalanceOrig", "newbalanceDest") if c in frame.columns)
    if data_cfg.get("exclude_flagged_fraud", True) and "isFlaggedFraud" in frame.columns:
        dropped.append("isFlaggedFraud")
    hour = (frame[data_cfg["time_column"]].astype("int64") - 1) % 24
    derived = pd.DataFrame(
        {
            "_time_seconds": frame[data_cfg["time_column"]].astype("int64") * 3600,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "elapsed_days": frame[data_cfg["time_column"]].astype("float64") / 24,
        },
        index=frame.index,
    )
    frame = pd.concat([frame, derived], axis=1)
    bundle = DatasetBundle(
        name="paysim",
        frame=frame,
        target_column=data_cfg["target"],
        time_column="_time_seconds",
        metadata={
            "path": str(path),
            "excluded_from_features": sorted(set(dropped)),
            "sample_mode": sample_mode,
            "causal_history_features": bool(data_cfg.get("add_causal_history_features", True)),
            "balance_consistency_features": bool(data_cfg.get("add_balance_consistency_features", True)),
        },
    )
    bundle.validate()
    return bundle


def load_dataset(config: dict, project_root: Path, max_rows: int | None = None, sample_mode: str = "prefix") -> DatasetBundle:
    dataset = config["experiment"]["dataset"]
    if dataset == "ieee_cis":
        return load_ieee(config, project_root, max_rows, sample_mode)
    if dataset == "paysim":
        return load_paysim(config, project_root, max_rows, sample_mode)
    raise ValueError(f"Unsupported dataset: {dataset}")
