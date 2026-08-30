from __future__ import annotations

import os
import time
from io import StringIO
from uuid import uuid4

import numpy as np
import pandas as pd

from .database import Database
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
# The model needs 336 hourly rows. The binding constraint is that row count
# after resampling, not elapsed days: 336 hourly samples span 335 hours, which
# is 13.96 days. The validator used to test only the span, against a threshold
# one hour below the model's, so a 335-row upload was accepted with a 201 and
# then failed every run made against it, permanently and with no explanation.
MINIMUM_HOURLY_ROWS = 14 * 24
MINIMUM_DURATION_DAYS = (MINIMUM_HOURLY_ROWS - 1) / 24
# Per-column magnitude ceilings. `inf` parsed cleanly through `to_numeric`,
# survived the NaN and negativity checks, and then produced either a 500 or a
# stored metric of 2.09e+285 kW carrying a 0.96 confidence score.
COLUMN_CEILINGS: dict[str, float] = {
    "it_load_kw": 5_000_000.0,
    "cooling_power_kw": 5_000_000.0,
    "total_facility_power_kw": 10_000_000.0,
    "supply_air_setpoint_c": 100.0,
    "return_air_temperature_c": 150.0,
    "chilled_water_supply_c": 100.0,
    "chilled_water_return_c": 100.0,
    "fan_speed_percent": 100.0,
    "indoor_humidity_percent": 100.0,
    "economizer_state": 1.0,
}


class TelemetryValidationError(ValueError):
    pass


# Retention bounds protect both the local file and the production Postgres
# database from anonymous uploads growing without limit.
MAX_RETAINED_UPLOADS = int(os.getenv("FORTYCOOL_MAX_UPLOADS", 32))
UPLOAD_TTL_SECONDS = float(os.getenv("FORTYCOOL_UPLOAD_TTL_SECONDS", 3600))


class TelemetryStore:
    """Validated telemetry persisted in Postgres or the local SQLite fallback."""

    def __init__(
        self,
        *,
        database: Database | None = None,
        max_uploads: int = MAX_RETAINED_UPLOADS,
        ttl_seconds: float = UPLOAD_TTL_SECONDS,
    ) -> None:
        self.database = database or Database()
        self.max_uploads = max_uploads
        self.ttl_seconds = ttl_seconds

    def _evict(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        with self.database.session() as session:
            session.execute(
                "DELETE FROM telemetry_uploads WHERE created_at_epoch < ?", (cutoff,)
            )
            row = session.execute("SELECT COUNT(*) FROM telemetry_uploads").fetchone()
            excess = max(0, int(row[0]) - self.max_uploads)
            if excess:
                session.execute(
                    """
                    DELETE FROM telemetry_uploads
                    WHERE upload_id IN (
                        SELECT upload_id FROM telemetry_uploads
                        ORDER BY accessed_at_epoch ASC, upload_id ASC
                        LIMIT ?
                    )
                    """,
                    (excess,),
                )

    def _forget(self, upload_id: str) -> None:
        with self.database.session() as session:
            session.execute(
                "DELETE FROM telemetry_uploads WHERE upload_id = ?", (upload_id,)
            )

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
        for column in sorted(numeric_columns):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any():
                raise TelemetryValidationError(
                    f"column {column} contains missing or non-numeric values"
                )
            values = values.astype(float)
            if not np.isfinite(values.to_numpy()).all():
                raise TelemetryValidationError(
                    f"column {column} contains infinite values"
                )
            ceiling = COLUMN_CEILINGS.get(column)
            if ceiling is not None and (values.abs() > ceiling).any():
                raise TelemetryValidationError(
                    f"column {column} contains values beyond the plausible range "
                    f"(|value| > {ceiling:g})"
                )
            frame[column] = values

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
        # Check the row count that will actually exist after the hourly
        # resample, which is what the model consumes.
        projected_hourly_rows = int(round(duration_days * 24)) + 1
        if duration_days < MINIMUM_DURATION_DAYS or (
            projected_hourly_rows < MINIMUM_HOURLY_ROWS
        ):
            raise TelemetryValidationError(
                f"at least {MINIMUM_HOURLY_ROWS} hourly observations are required "
                f"after resampling; this file covers {duration_days:.2f} days and "
                f"yields about {projected_hourly_rows}"
            )
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
        stored_at = time.time()
        frame_json = frame.to_json(
            orient="split", date_format="iso", date_unit="us", double_precision=15
        )
        with self.database.session() as session:
            session.execute(
                """
                INSERT INTO telemetry_uploads (
                    upload_id, created_at_epoch, accessed_at_epoch,
                    metadata_json, frame_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    stored_at,
                    stored_at,
                    result.model_dump_json(),
                    frame_json,
                ),
            )
        self._evict()
        return result

    def get(self, upload_id: str) -> pd.DataFrame:
        self._evict()
        with self.database.session() as session:
            row = session.execute(
                "SELECT frame_json FROM telemetry_uploads WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
            if row is not None:
                session.execute(
                    """
                    UPDATE telemetry_uploads SET accessed_at_epoch = ?
                    WHERE upload_id = ?
                    """,
                    (time.time(), upload_id),
                )
        if row is None:
            raise KeyError(upload_id)
        frame = pd.read_json(StringIO(row[0]), orient="split")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    def metadata(self, upload_id: str) -> TelemetryUpload:
        self._evict()
        with self.database.session() as session:
            row = session.execute(
                "SELECT metadata_json FROM telemetry_uploads WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
        if row is None:
            raise KeyError(upload_id)
        return TelemetryUpload.model_validate_json(row[0])

    def delete(self, upload_id: str) -> bool:
        with self.database.session() as session:
            cursor = session.execute(
                "DELETE FROM telemetry_uploads WHERE upload_id = ?", (upload_id,)
            )
            return cursor.rowcount > 0
