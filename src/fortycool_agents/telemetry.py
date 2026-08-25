from __future__ import annotations

from io import StringIO
from uuid import uuid4

import numpy as np
import pandas as pd

from .models import TelemetryUpload


REQUIRED_COLUMNS = {
    "timestamp",
    "it_load_kw",
    "cooling_power_kw",
    "server_inlet_temperature_c",
}
OPTIONAL_NUMERIC_COLUMNS = {
    "total_facility_power_kw",
    "supply_air_setpoint_c",
    "return_air_temperature_c",
    "chilled_water_supply_c",
    "chilled_water_return_c",
    "fan_speed_percent",
    "indoor_humidity_percent",
    "economizer_state",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_ROWS = 200_000


class TelemetryValidationError(ValueError):
    pass


class TelemetryStore:
    """Validated in-memory telemetry store for the hackathon service.

    The store intentionally retains only normalized numeric columns. A durable
    object-store adapter can replace it later without changing the request model.
    """

    def __init__(self) -> None:
        self._frames: dict[str, pd.DataFrame] = {}
        self._metadata: dict[str, TelemetryUpload] = {}

    def ingest_csv(self, raw: bytes) -> TelemetryUpload:
        if not raw:
            raise TelemetryValidationError("CSV body is empty")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise TelemetryValidationError("CSV exceeds the 10 MB upload limit")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise TelemetryValidationError("CSV must use UTF-8 encoding") from exc
        try:
            frame = pd.read_csv(StringIO(text))
        except Exception as exc:  # pandas emits several parser-specific errors
            raise TelemetryValidationError(f"CSV could not be parsed: {exc}") from exc
        if len(frame) > MAX_UPLOAD_ROWS:
            raise TelemetryValidationError(f"CSV exceeds the {MAX_UPLOAD_ROWS:,}-row limit")
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise TelemetryValidationError(
                "CSV is missing required columns: " + ", ".join(sorted(missing))
            )

        warnings: list[str] = []
        timestamp_text = frame["timestamp"].astype(str)
        parsed_timestamps = pd.to_datetime(timestamp_text, utc=True, errors="coerce")
        if parsed_timestamps.isna().any():
            bad_count = int(parsed_timestamps.isna().sum())
            raise TelemetryValidationError(f"{bad_count} timestamps could not be parsed")
        if not timestamp_text.str.contains(r"(?:Z|[+-]\d\d:?\d\d)$", regex=True).all():
            warnings.append("Timestamps without UTC offsets were interpreted as UTC")
        frame["timestamp"] = parsed_timestamps

        numeric_columns = (REQUIRED_COLUMNS - {"timestamp"}) | (
            OPTIONAL_NUMERIC_COLUMNS & set(frame.columns)
        )
        for column in numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any():
                raise TelemetryValidationError(
                    f"column {column} contains missing or non-numeric values"
                )
            frame[column] = values.astype(float)

        if (frame["it_load_kw"] < 0).any() or (frame["cooling_power_kw"] < 0).any():
            raise TelemetryValidationError("power values cannot be negative")
        if not frame["server_inlet_temperature_c"].between(-10, 60).all():
            raise TelemetryValidationError("server inlet temperatures fall outside -10°C to 60°C")

        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        if len(frame) < 2:
            raise TelemetryValidationError("CSV must contain at least two unique timestamps")
        intervals = frame["timestamp"].diff().dropna().dt.total_seconds() / 60
        median_interval = float(intervals.median())
        if not 14 <= median_interval <= 65:
            raise TelemetryValidationError(
                "telemetry interval must be between approximately 15 and 60 minutes"
            )
        duration_days = (
            frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0]
        ).total_seconds() / 86400
        if duration_days < 13.9:
            raise TelemetryValidationError("at least 14 days of telemetry are required")
        irregular_share = float(np.mean(np.abs(intervals - median_interval) > median_interval * 0.2))
        if irregular_share > 0.05:
            warnings.append(
                f"{irregular_share:.1%} of intervals differ materially from the median cadence"
            )

        retained_numeric = (
            (REQUIRED_COLUMNS | OPTIONAL_NUMERIC_COLUMNS) & set(frame.columns)
        ) - {"timestamp"}
        retained = ["timestamp"] + sorted(retained_numeric)
        frame = frame[retained].reset_index(drop=True)
        upload_id = uuid4().hex
        result = TelemetryUpload(
            upload_id=upload_id,
            rows=len(frame),
            start_timestamp=frame["timestamp"].iloc[0].to_pydatetime(),
            end_timestamp=frame["timestamp"].iloc[-1].to_pydatetime(),
            median_interval_minutes=round(median_interval, 3),
            columns=list(frame.columns),
            warnings=warnings,
        )
        self._frames[upload_id] = frame
        self._metadata[upload_id] = result
        return result

    def get(self, upload_id: str) -> pd.DataFrame:
        if upload_id not in self._frames:
            raise KeyError(upload_id)
        return self._frames[upload_id].copy(deep=True)

    def metadata(self, upload_id: str) -> TelemetryUpload:
        if upload_id not in self._metadata:
            raise KeyError(upload_id)
        return self._metadata[upload_id]

    def delete(self, upload_id: str) -> bool:
        existed = upload_id in self._frames
        self._frames.pop(upload_id, None)
        self._metadata.pop(upload_id, None)
        return existed
