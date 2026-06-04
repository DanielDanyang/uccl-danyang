from pathlib import Path

_DEEPEP_V2_SUBMODULE_UTILS = (
    Path(__file__).resolve().parents[4]
    / "thirdparty"
    / "DeepEP-v2-d4f41e4"
    / "deep_ep"
    / "utils"
)
if _DEEPEP_V2_SUBMODULE_UTILS.exists():
    __path__.append(str(_DEEPEP_V2_SUBMODULE_UTILS))

from .event import EventOverlap

__all__ = ["EventOverlap"]
