import os

from .dynamic_world import (
    DynamicWorldError,
    DynamicWorldUrbanProvider,
    EarthEngineDynamicWorldClient,
)
from .fixture import FixtureThermalProvider, ThermalDataProvider
from .fortyguard import FortyGuardClient, FortyGuardError
from .live import FortyGuardThermalProvider
from .urban import FixtureUrbanProvider, FortyGuardUrbanProvider, UrbanContextProvider


def build_thermal_provider(mode: str | None = None) -> ThermalDataProvider:
    selected = (
        (mode or os.getenv("FORTYCOOL_THERMAL_PROVIDER", "fixture")).strip().lower()
    )
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


def build_urban_provider(
    mode: str | None = None,
    *,
    thermal_provider: ThermalDataProvider | None = None,
) -> UrbanContextProvider:
    inferred_live_mode = (
        mode is None
        and isinstance(thermal_provider, FortyGuardThermalProvider)
        and isinstance(thermal_provider.client, FortyGuardClient)
    )
    configured_mode = mode or os.getenv("FORTYCOOL_URBAN_PROVIDER")
    selected = (
        (
            configured_mode
            or ("live" if inferred_live_mode else None)
            or os.getenv("FORTYCOOL_THERMAL_PROVIDER", "fixture")
        )
        .strip()
        .lower()
    )
    if selected == "fixture":
        return FixtureUrbanProvider()
    if selected == "live":
        if isinstance(thermal_provider, FortyGuardThermalProvider) and isinstance(
            thermal_provider.client, FortyGuardClient
        ):
            return FortyGuardUrbanProvider(
                thermal_provider.client,
                granularity_m=thermal_provider.granularity_m,
                max_concurrency=thermal_provider.max_concurrency,
                clock=thermal_provider.clock,
            )
        client = FortyGuardClient()
        if not client.configured:
            raise FortyGuardError(
                "FORTYCOOL_THERMAL_PROVIDER=live requires FORTYGUARD_API_KEY"
            )
        return FortyGuardUrbanProvider(client)
    if selected == "dynamic_world":
        return DynamicWorldUrbanProvider()
    raise ValueError(
        "FORTYCOOL_URBAN_PROVIDER must be 'fixture', 'live', or 'dynamic_world'"
    )


__all__ = [
    "build_thermal_provider",
    "build_urban_provider",
    "DynamicWorldError",
    "DynamicWorldUrbanProvider",
    "EarthEngineDynamicWorldClient",
    "FixtureThermalProvider",
    "FortyGuardClient",
    "FortyGuardError",
    "FortyGuardThermalProvider",
    "FixtureUrbanProvider",
    "FortyGuardUrbanProvider",
    "ThermalDataProvider",
    "UrbanContextProvider",
]
