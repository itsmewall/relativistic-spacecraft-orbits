"""Mission package exports kept lazy to avoid import cycles."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ManeuverRecord": ("relorbit_py.mission.mission", "ManeuverRecord"),
    "MissionResult": ("relorbit_py.mission.mission", "MissionResult"),
    "run_mission": ("relorbit_py.mission.mission", "run_mission"),
    "AttitudeMissionResult": (
        "relorbit_py.mission.attitude_mission",
        "AttitudeMissionResult",
    ),
    "from_yaml_dict": ("relorbit_py.mission.attitude_mission", "from_yaml_dict"),
    "run_attitude_mission": (
        "relorbit_py.mission.attitude_mission",
        "run_attitude_mission",
    ),
    "validate_attitude": (
        "relorbit_py.mission.attitude_mission",
        "validate_attitude",
    ),
    "main": ("relorbit_py.mission.run_mission", "main"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + __all__)
