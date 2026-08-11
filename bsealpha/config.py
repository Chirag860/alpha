"""Typed configuration loader.

The whole pipeline is config-driven (§9, §11.1). We load ``config/default.yaml``
into a nested, attribute-accessible structure so downstream modules can pull knobs
by name (``cfg.labeling.horizon_min``) while remaining swappable.

We deliberately keep this a thin, dependency-light wrapper over a plain dict rather
than pydantic: the config is validated by the modules that consume it, and a light
wrapper keeps the package importable with only pyyaml present.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


class Config(Mapping[str, Any]):
    """Immutable-ish, attribute- and item-accessible nested config node.

    Nested dicts are wrapped recursively so ``cfg.model.meta.num_leaves`` works.
    Still supports ``cfg["model"]`` and ``dict(cfg)`` for interop.
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", {})
        for key, value in data.items():
            self._data[key] = _wrap(value)

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- attribute access ---------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:  # pragma: no cover
        raise AttributeError("Config is read-only; use .with_overrides(...) instead")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Recursively unwrap back to plain dicts."""
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self._data.items()}

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a new Config with a deep-merged set of overrides applied."""
        merged = _deep_merge(self.to_dict(), dict(overrides))
        return Config(merged)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({self._data!r})"


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None,
                overrides: Mapping[str, Any] | None = None) -> Config:
    """Load a :class:`Config` from YAML, optionally deep-merging ``overrides``.

    Parameters
    ----------
    path
        Path to a YAML file. Defaults to ``config/default.yaml``.
    overrides
        Optional nested mapping merged on top of the file contents.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config(raw)
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg
