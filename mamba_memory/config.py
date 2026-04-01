"""Configuration file loading and management.

Supports loading EngineConfig from YAML/TOML files and environment variables.

Config resolution order (later overrides earlier):
  1. Built-in defaults (EngineConfig())
  2. Config file (~/.mamba-memory/config.yaml or --config path)
  3. Environment variables (MAMBA_MEMORY_*)

Config file format (YAML):

    embedding_provider: google
    compression_model: none

    l1:
      window_size: 8
      max_compressed_segments: 20

    l2:
      slot_count: 64
      base_decay_rate: 0.98
      eviction_threshold: 0.05

    l3:
      db_path: ~/.mamba-memory/default.db
"""

from __future__ import annotations

import os
from pathlib import Path

from mamba_memory.core.types import EngineConfig, L1Config, L2Config, L3Config

DEFAULT_CONFIG_DIR = Path("~/.mamba-memory").expanduser()
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"


def load_config(path: str | Path | None = None) -> EngineConfig:
    """Load EngineConfig from a YAML file + environment overrides.

    Args:
        path: Config file path. If None, tries ~/.mamba-memory/config.yaml.
              If the file doesn't exist, returns defaults.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict = {}

    if config_path.exists():
        data = _load_yaml(config_path)

    # Environment overrides
    data = _apply_env_overrides(data)

    return _dict_to_config(data)


def save_config(config: EngineConfig, path: str | Path | None = None) -> Path:
    """Save EngineConfig to a YAML file."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = _config_to_dict(config)
    _save_yaml(config_path, data)
    return config_path


def _load_yaml(path: Path) -> dict:
    """Load YAML file, with fallback to plain key=value if PyYAML not available."""
    text = path.read_text(encoding="utf-8")

    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    # Fallback: simple key: value parser (flat only)
    data: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value.lower() in ("true", "false"):
                data[key] = value.lower() == "true"
            elif value.replace(".", "").isdigit():
                data[key] = float(value) if "." in value else int(value)
            else:
                data[key] = value
    return data


def _save_yaml(path: Path, data: dict) -> None:
    """Save dict to YAML, with fallback to simple format."""
    try:
        import yaml
        text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except ImportError:
        lines = []
        _flatten_to_lines(data, lines, indent=0)
        text = "\n".join(lines) + "\n"

    path.write_text(text, encoding="utf-8")


def _flatten_to_lines(data: dict, lines: list[str], indent: int) -> None:
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            _flatten_to_lines(value, lines, indent + 1)
        else:
            lines.append(f"{prefix}{key}: {value}")


def _apply_env_overrides(data: dict) -> dict:
    """Apply MAMBA_MEMORY_* environment variables as overrides."""
    env_map = {
        "MAMBA_MEMORY_DB": ("l3", "db_path"),
        "MAMBA_MEMORY_EMBEDDING": ("embedding_provider",),
        "MAMBA_MEMORY_COMPRESSION": ("compression_model",),
        "MAMBA_MEMORY_SLOTS": ("l2", "slot_count"),
        "MAMBA_MEMORY_DECAY_RATE": ("l2", "base_decay_rate"),
        "MAMBA_MEMORY_WINDOW_SIZE": ("l1", "window_size"),
    }

    for env_key, path in env_map.items():
        value = os.environ.get(env_key)
        if value is None:
            continue

        # Type conversion
        if path[-1] in ("slot_count", "window_size"):
            value = int(value)
        elif path[-1] in ("base_decay_rate",):
            value = float(value)

        # Set nested key
        target = data
        for key in path[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[path[-1]] = value

    return data


def _dict_to_config(data: dict) -> EngineConfig:
    """Convert a flat/nested dict to EngineConfig."""
    l1_data = data.get("l1", {})
    l2_data = data.get("l2", {})
    l3_data = data.get("l3", {})

    return EngineConfig(
        l1=L1Config(**{k: v for k, v in l1_data.items() if k in L1Config.model_fields}),
        l2=L2Config(**{k: v for k, v in l2_data.items() if k in L2Config.model_fields}),
        l3=L3Config(**{k: v for k, v in l3_data.items() if k in L3Config.model_fields}),
        embedding_provider=data.get("embedding_provider", "auto"),
        compression_model=data.get("compression_model", "auto"),
    )


def _config_to_dict(config: EngineConfig) -> dict:
    """Convert EngineConfig to a clean dict for YAML serialization."""
    return {
        "embedding_provider": config.embedding_provider,
        "compression_model": config.compression_model,
        "l1": {
            "window_size": config.l1.window_size,
            "max_compressed_segments": config.l1.max_compressed_segments,
        },
        "l2": {
            "slot_count": config.l2.slot_count,
            "base_decay_rate": config.l2.base_decay_rate,
            "eviction_threshold": config.l2.eviction_threshold,
            "snapshot_interval": config.l2.snapshot_interval,
            "weight_semantic": config.l2.weight_semantic,
            "weight_activation": config.l2.weight_activation,
            "weight_recency": config.l2.weight_recency,
            "weight_importance": config.l2.weight_importance,
        },
        "l3": {
            "db_path": config.l3.db_path,
        },
    }
