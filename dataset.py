import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple
import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

import torch
from torch.utils.data import Dataset as TorchDataset

LOGGER = logging.getLogger("multisensor.dataset")


def _to_float_dtype(value_dtype: str) -> np.dtype:
    return np.float32 if str(value_dtype).lower() == "float32" else np.float64


def _agg_func_name(agg: str) -> str:
    if agg in ("mean", "median", "last", "first"):
        return agg
    raise ValueError(f"Unsupported agg: {agg}")


def _apply_fill(df: pd.DataFrame, fill: str, limit: Optional[int]) -> pd.DataFrame:
    if fill == "none":
        return df
    if fill == "ffill":
        return df.ffill(limit=limit)
    if fill == "bfill":
        return df.bfill(limit=limit)
    if fill == "interpolate":
        return df.interpolate(limit=limit)
    raise ValueError(f"Unsupported fill: {fill}")


@dataclass(frozen=True)
class DatasetConfig:
    parquet_path: str = "multisensor.parquet"
    meta_path: str = "multisensor.parquet.meta.json"
    # Selection
    devices: Optional[List[int]] = None
    metrics: Optional[List[str]] = None
    # Time slicing
    start: Optional[str] = None
    end: Optional[str] = None
    # Resampling (no fill at this stage; filling happens per-window)
    freq: str = "1min"
    agg: str = "mean"  # mean|median|last|first
    value_dtype: str = "float32"  # float32|float64
    # Windowing
    window_size: int = 60
    horizon: int = 1
    stride: int = 1
    target_metric: Optional[str] = None
    # Training-time utility
    normalize: bool = False
    return_timestamps: bool = False
    return_observation_mask: bool = True
    # Window filtering based on activity of original (pre-fill) observations
    min_activity_ratio: float = (
        0.2  # minimum fraction of originally observed points in X window
    )
    # Optional per-metric overrides: {metric_name: ratio}
    min_activity_ratio_by_metric: Optional[Dict[str, float]] = None
    max_fill_ratio: Optional[float] = (
        None  # maximum fraction of filled points allowed in X window
    )
    # Optional per-metric overrides: {metric_name: ratio}
    max_fill_ratio_by_metric: Optional[Dict[str, float]] = None
    min_variance: Optional[float] = (
        None  # drop near-static windows; variance over X across all metrics
    )
    # If true, windows must be fully filled after applying window-level fill
    require_full_windows: bool = False
    # Window-level fill configuration
    fill: str = "ffill"  # none|ffill|bfill|interpolate
    fill_limit: Optional[int] = None
    # Encodings injected into X
    include_device_id: bool = True
    device_encoding: str = "one_hot"  # one_hot|index
    include_time_encoding: bool = True  # add time-of-day and day-of-week sin/cos


class MultisensorDataset(TorchDataset):
    """PyTorch Dataset that resamples without fill and applies fill per-window.

    Returns (x, y[, mask][, tx, ty]) suitable for transformer training.
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for MultisensorDataset")

        self.cfg = cfg
        if os.path.exists(cfg.meta_path):
            with open(cfg.meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
        else:
            raise FileNotFoundError(f"Meta file not found: {cfg.meta_path}")

        self._dataset = ds.dataset(cfg.parquet_path, format="parquet")
        # Remember the parquet timestamp type (unit and timezone) to build compatible scalars
        self._timestamp_type = self._dataset.schema.field("timestamp").type
        self._metric_name_to_id: Dict[str, int] = self._meta["metrics"]
        self._metric_id_to_name: Dict[int, str] = {
            v: k for k, v in self._metric_name_to_id.items()
        }
        self._device_id_to_name: Dict[int, str] = {
            v: k for k, v in self._meta["devices"].items()
        }

        if cfg.metrics is not None:
            unknown = [m for m in cfg.metrics if m not in self._metric_name_to_id]
            if unknown:
                raise ValueError(f"Unknown metrics requested: {unknown}")
            self._selected_metric_ids = sorted(
                self._metric_name_to_id[m] for m in cfg.metrics
            )
        else:
            self._selected_metric_ids = sorted(self._metric_id_to_name.keys())

        if cfg.devices is None:
            LOGGER.info("Scanning distinct device_id values")
            tbl = self._dataset.to_table(columns=["device_id"])  # light footprint
            device_ids = np.unique(np.asarray(tbl.column("device_id")).astype("int64"))
            self._selected_devices = [int(x) for x in device_ids]
        else:
            self._selected_devices = cfg.devices

        self._time_start = pd.to_datetime(cfg.start, utc=True) if cfg.start else None
        self._time_end = pd.to_datetime(cfg.end, utc=True) if cfg.end else None

        LOGGER.info(
            "Initialized dataset for %d devices, %d metrics",
            len(self._selected_devices),
            len(self._selected_metric_ids),
        )

        float_dtype = _to_float_dtype(cfg.value_dtype)
        self._metric_order = self._selected_metric_ids
        self._metric_names = [
            self._metric_id_to_name[mid] for mid in self._metric_order
        ]

        self._device_to_arrays: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = (
            []
        )
        self._device_start_indices: List[np.ndarray] = []
        # Filled values per device, aligned with self._device_to_arrays
        self._device_filled_values: List[np.ndarray] = []

        # Build per-metric thresholds (defaults filled from global values)
        num_metrics = len(self._metric_names)
        self._min_activity_thresholds = np.full(
            (num_metrics,), float(cfg.min_activity_ratio), dtype="float32"
        )
        if cfg.min_activity_ratio_by_metric:
            for name, val in cfg.min_activity_ratio_by_metric.items():
                if name in self._metric_names:
                    idx = self._metric_names.index(name)
                    self._min_activity_thresholds[idx] = float(val)
                else:
                    LOGGER.warning(
                        "Ignoring unknown metric in min_activity_ratio_by_metric: %s",
                        name,
                    )

        default_max_fill = (
            float(cfg.max_fill_ratio)
            if cfg.max_fill_ratio is not None
            else float("inf")
        )
        self._max_fill_thresholds = np.full(
            (num_metrics,), default_max_fill, dtype="float32"
        )
        if cfg.max_fill_ratio_by_metric:
            for name, val in cfg.max_fill_ratio_by_metric.items():
                if name in self._metric_names:
                    idx = self._metric_names.index(name)
                    self._max_fill_thresholds[idx] = float(val)
                else:
                    LOGGER.warning(
                        "Ignoring unknown metric in max_fill_ratio_by_metric: %s",
                        name,
                    )

        for (
            device_id,
            unfilled_values,
            observed_mask,
            timestamps,
        ) in self._iter_device_arrays_unfilled(cfg):
            total_len = cfg.window_size + cfg.horizon
            if unfilled_values.shape[0] < total_len:
                continue

            starts = self._compute_valid_window_starts(
                unfilled_values=unfilled_values,
                observed_mask=observed_mask,
                window_size=cfg.window_size,
                horizon=cfg.horizon,
                stride=cfg.stride,
                min_activity_thresholds=self._min_activity_thresholds,
                max_fill_thresholds=self._max_fill_thresholds,
                min_variance=cfg.min_variance,
                require_full_windows=cfg.require_full_windows,
                fill=cfg.fill,
                fill_limit=cfg.fill_limit,
                value_dtype=float_dtype,
            )

            if len(starts) == 0:
                continue

            self._device_to_arrays.append(
                (device_id, unfilled_values, observed_mask, timestamps)
            )
            self._device_start_indices.append(starts)
            # Compute filled values for later slicing and normalization
            filled_df = _apply_fill(
                pd.DataFrame(unfilled_values),
                fill=cfg.fill,
                limit=cfg.fill_limit,
            )
            filled_values = filled_df.astype(float_dtype).to_numpy(copy=False)
            self._device_filled_values.append(filled_values)

        if len(self._device_to_arrays) == 0:
            raise ValueError("No devices with enough qualifying data for windowing")

        self._device_offsets: List[int] = []
        total = 0
        for starts in self._device_start_indices:
            self._device_offsets.append(total)
            total += int(len(starts))
        self._total_windows = total

        self._target_col: Optional[int] = None
        if cfg.target_metric is not None:
            if cfg.target_metric not in self._metric_names:
                raise ValueError(
                    f"target_metric '{cfg.target_metric}' not in selected metrics {self._metric_names}"
                )
            self._target_col = self._metric_names.index(cfg.target_metric)

        self._return_timestamps = bool(cfg.return_timestamps)
        self._return_observation_mask = bool(cfg.return_observation_mask)

        # Compute normalization statistics if requested
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None
        if bool(cfg.normalize):
            num_metrics = len(self._metric_names)
            # Accumulate sums and counts ignoring NaNs for numerical stability
            sum_per_metric = np.zeros((num_metrics,), dtype="float64")
            count_per_metric = np.zeros((num_metrics,), dtype="int64")
            for filled in self._device_filled_values:
                mask = ~np.isnan(filled)
                sum_per_metric += np.where(mask, filled, 0.0).sum(axis=0)
                count_per_metric += mask.sum(axis=0)

            # Avoid division by zero; where count is 0, set mean to 0
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.divide(
                    sum_per_metric, count_per_metric, where=count_per_metric > 0
                )
            mean = np.nan_to_num(mean, nan=0.0)

            # Second pass for variance
            var_sum_per_metric = np.zeros((num_metrics,), dtype="float64")
            for filled in self._device_filled_values:
                diff = filled - mean
                diff[~np.isfinite(diff)] = 0.0
                var_sum_per_metric += np.where(~np.isnan(filled), diff * diff, 0.0).sum(
                    axis=0
                )

            with np.errstate(invalid="ignore", divide="ignore"):
                var = np.divide(
                    var_sum_per_metric, count_per_metric, where=count_per_metric > 0
                )
            var = np.nan_to_num(var, nan=0.0)
            std = np.sqrt(var)

            # Clamp std to a minimum epsilon to avoid divide-by-zero
            eps = 1e-6 if str(cfg.value_dtype).lower() == "float32" else 1e-12
            std = np.maximum(std, eps)

            # Cast to configured float dtype
            if cfg.value_dtype == "float32":
                self._norm_mean = mean.astype("float32")
                self._norm_std = std.astype("float32")
            else:
                self._norm_mean = mean.astype("float64")
                self._norm_std = std.astype("float64")

    def _build_filter(self, device_id: int) -> Optional[ds.Expression]:
        expr: Optional[ds.Expression] = ds.field("device_id") == device_id

        def _ts_scalar(ts: pd.Timestamp) -> pa.Scalar:
            # Build a scalar matching the parquet field's timestamp type (unit + tz)
            t = self._timestamp_type
            if pa.types.is_timestamp(t):
                unit = t.unit
                # pandas Timestamp.value is ns since epoch (UTC)
                ns = int(ts.value)
                divisor = {
                    "ns": 1,
                    "us": 1_000,
                    "ms": 1_000_000,
                    "s": 1_000_000_000,
                }[str(unit)]
                val = ns // divisor
                return pa.scalar(val, type=t)
            # Fallback: let Arrow infer
            return pa.scalar(ts.to_pydatetime())

        if self._time_start is not None:
            expr = expr & (ds.field("timestamp") >= _ts_scalar(self._time_start))
        if self._time_end is not None:
            expr = expr & (ds.field("timestamp") <= _ts_scalar(self._time_end))
        return expr

    def _iter_device_arrays_unfilled(
        self, cfg: DatasetConfig
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        agg_name = _agg_func_name(cfg.agg)
        float_dtype = _to_float_dtype(cfg.value_dtype)
        metric_ids_set = set(self._selected_metric_ids)

        for device_id in self._selected_devices:
            filt = self._build_filter(device_id)
            scanner = self._dataset.scanner(
                columns=["timestamp", "metric_id", "value"],
                filter=filt,
                use_threads=True,
            )
            tbl = scanner.to_table()
            if tbl.num_rows == 0:
                LOGGER.warning("Device %s has no rows in selected range", device_id)
                continue

            pdf = tbl.to_pandas(types_mapper=pd.ArrowDtype)
            pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True)
            pdf = pdf[pdf["metric_id"].isin(list(metric_ids_set))]
            if len(pdf) == 0:
                LOGGER.warning("Device %s has no selected metrics", device_id)
                continue

            wide = pdf.pivot_table(
                index="timestamp",
                columns="metric_id",
                values="value",
                aggfunc=agg_name,
            ).sort_index()

            wide = wide.reindex(columns=self._metric_order)
            wide = getattr(wide.resample(cfg.freq), agg_name)()

            observed_mask = (~wide.isna()).astype("int8")

            values_unfilled = wide.astype(float_dtype).to_numpy(copy=False)
            mask_np = observed_mask.to_numpy(copy=False)
            ts_np = wide.index.to_numpy(dtype="datetime64[ns]").astype(
                "int64", copy=False
            )

            LOGGER.info(
                "Device %s: %d rows after resample (unfilled), %.3f observed ratio",
                device_id,
                len(wide),
                float(mask_np.mean()) if mask_np.size > 0 else 0.0,
            )

            yield device_id, values_unfilled, mask_np, ts_np

    def _compute_valid_window_starts(
        self,
        unfilled_values: np.ndarray,
        observed_mask: np.ndarray,
        window_size: int,
        horizon: int,
        stride: int,
        min_activity_thresholds: np.ndarray,
        max_fill_thresholds: np.ndarray,
        min_variance: Optional[float],
        require_full_windows: bool,
        fill: str,
        fill_limit: Optional[int],
        value_dtype: np.dtype,
    ) -> np.ndarray:
        T = unfilled_values.shape[0]
        total_len = window_size + horizon
        if T < total_len:
            return np.array([], dtype=np.int64)

        filled_df = _apply_fill(
            pd.DataFrame(unfilled_values),
            fill=fill,
            limit=fill_limit,
        )
        filled_values = filled_df.astype(value_dtype).to_numpy(copy=False)

        max_start = T - total_len
        starts = np.arange(0, max_start + 1, dtype=np.int64)
        if stride > 1:
            starts = starts[::stride]

        keep: List[int] = []
        for s in starts:
            x_mask = observed_mask[s : s + window_size]
            # Per-metric observed ratios across time
            if x_mask.size == 0:
                continue
            observed_ratio_per_metric = x_mask.mean(axis=0).astype(
                "float32", copy=False
            )

            # Enforce per-metric thresholds (all metrics must satisfy)
            if np.any(observed_ratio_per_metric < min_activity_thresholds):
                continue

            fill_ratio_per_metric = 1.0 - observed_ratio_per_metric
            if np.any(fill_ratio_per_metric > max_fill_thresholds):
                continue

            if require_full_windows:
                x_filled = filled_values[s : s + window_size]
                y_filled = filled_values[s + window_size : s + window_size + horizon]
                if np.isnan(x_filled).any() or np.isnan(y_filled).any():
                    continue

            if min_variance is not None:
                x_filled = filled_values[s : s + window_size]
                var_value = float(np.var(x_filled, axis=0).mean())
                if var_value < min_variance:
                    continue

            keep.append(int(s))

        return np.asarray(keep, dtype=np.int64)

    def __len__(self) -> int:
        return self._total_windows

    def _locate(self, idx: int) -> Tuple[int, int]:
        lo = 0
        hi = len(self._device_offsets) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self._device_offsets[mid]
            end = (
                self._device_offsets[mid + 1]
                if mid + 1 < len(self._device_offsets)
                else self._total_windows
            )
            if start <= idx < end:
                return mid, idx - start
            if idx < start:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        cfg = self.cfg
        metric_names = self._metric_names

        dev_idx, local = self._locate(idx)
        device_id, unfilled_values, observed_mask, timestamps = self._device_to_arrays[
            dev_idx
        ]
        start = int(self._device_start_indices[dev_idx][local])

        window = cfg.window_size
        horizon = cfg.horizon

        filled_values = self._device_filled_values[dev_idx]
        x_filled = filled_values[start : start + window].copy()
        y_filled = filled_values[start + window : start + window + horizon].copy()

        if cfg.value_dtype == "float32":
            x_filled = x_filled.astype("float32", copy=False)
            y_filled = y_filled.astype("float32", copy=False)
        else:
            x_filled = x_filled.astype("float64", copy=False)
            y_filled = y_filled.astype("float64", copy=False)

        # Apply normalization if configured
        if self._norm_mean is not None and self._norm_std is not None:
            x_filled = (x_filled - self._norm_mean) / self._norm_std
            if self._target_col is None:
                y_filled = (y_filled - self._norm_mean) / self._norm_std
            else:
                col = self._target_col
                y_filled = (y_filled - self._norm_mean[col]) / self._norm_std[col]

        # Build additional encodings to inject into X
        extra_feats: List[np.ndarray] = []
        extra_mask_feats: List[np.ndarray] = []

        # Device encoding (constant per timestep in the window)
        if bool(cfg.include_device_id):
            if str(cfg.device_encoding) == "one_hot":
                num_devices = len(self._selected_devices)
                # Map device_id to index within selected_devices
                try:
                    device_pos = self._selected_devices.index(int(device_id))
                except ValueError:
                    device_pos = 0
                one_hot = np.zeros((window, num_devices), dtype=x_filled.dtype)
                one_hot[:, device_pos] = 1.0
                extra_feats.append(one_hot)
                extra_mask_feats.append(np.ones((window, num_devices), dtype="float32"))
            else:
                # index encoding as a single continuous feature normalized by num_devices
                num_devices = max(1, len(self._selected_devices))
                try:
                    device_pos = self._selected_devices.index(int(device_id))
                except ValueError:
                    device_pos = 0
                idx_feat = np.full(
                    (window, 1), float(device_pos), dtype=x_filled.dtype
                ) / float(num_devices)
                extra_feats.append(idx_feat)
                extra_mask_feats.append(np.ones((window, 1), dtype="float32"))

        # Time encoding per timestep using timestamps
        if bool(cfg.include_time_encoding):
            tx = timestamps[start : start + window]
            # Convert to UTC datetime -> DatetimeIndex
            dt = pd.to_datetime(tx, utc=True)
            # Time of day fraction using DatetimeIndex accessors
            hours = dt.hour.to_numpy()
            minutes = dt.minute.to_numpy()
            seconds = dt.second.to_numpy()
            seconds_in_day = (
                hours.astype("int64") * 3600
                + minutes.astype("int64") * 60
                + seconds.astype("int64")
            )
            frac_day = seconds_in_day / 86400.0
            tod_sin = np.sin(2 * np.pi * frac_day).astype(x_filled.dtype)
            tod_cos = np.cos(2 * np.pi * frac_day).astype(x_filled.dtype)
            # Day of week including fraction of day
            dow = dt.dayofweek.to_numpy()
            frac_week = (dow + frac_day) / 7.0
            dow_sin = np.sin(2 * np.pi * frac_week).astype(x_filled.dtype)
            dow_cos = np.cos(2 * np.pi * frac_week).astype(x_filled.dtype)
            time_feats = np.stack([tod_sin, tod_cos, dow_sin, dow_cos], axis=1)
            extra_feats.append(time_feats)
            extra_mask_feats.append(np.ones((window, 4), dtype="float32"))

        if extra_feats:
            x_filled = np.concatenate([x_filled] + extra_feats, axis=1)

        if self._target_col is None:
            y = y_filled
        else:
            y = y_filled[:, self._target_col : self._target_col + 1]

        outputs: Tuple[object, ...] = (
            torch.from_numpy(np.ascontiguousarray(x_filled)),
            torch.from_numpy(np.ascontiguousarray(y)),
        )

        if self._return_observation_mask:
            x_mask = observed_mask[start : start + window]
            if extra_mask_feats:
                x_mask = np.concatenate([x_mask] + extra_mask_feats, axis=1)
            outputs = outputs + (
                torch.from_numpy(np.ascontiguousarray(x_mask.astype("float32"))),
            )

        if self._return_timestamps:
            tx = timestamps[start : start + window]
            ty = timestamps[start + window : start + window + horizon]
            outputs = outputs + (
                torch.from_numpy(tx.copy()),
                torch.from_numpy(ty.copy()),
            )

        return outputs
