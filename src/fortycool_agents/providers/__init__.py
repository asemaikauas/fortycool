import os

from .fixture import FixtureThermalProvider, ThermalDataProvider
from .fortyguard import FortyGuardClient, FortyGuardError
from .live import FortyGuardThermalProvider


def build_thermal_provider(mode: str | None = None) -> ThermalDataProvider:
    selected = (mode or os.getenv("FORTYCOOL_THERMAL_PROVIDER", "fixture")).strip().lower()
    if selected == "fixture":
        return FixtureThermalProvider()
    if selected == "live":
        client = FortyGuardClient()
        if not client.configured:
            raise FortyGuardError(
                "FORTYCOOL_THERMAL_PROVIDER=live requires FORTYGUARD_API_KEY"
            )
        return FortyGuardThermalProvider(client)
    raise ValueError("FORTYCOOL_THERMAL_PROVIDER must be 'fixture' or 'live'")

__all__ = [
    "build_thermal_provider",
    "FixtureThermalProvider",
    "FortyGuardClient",
    "FortyGuardError",
    "FortyGuardThermalProvider",
    "ThermalDataProvider",
]
