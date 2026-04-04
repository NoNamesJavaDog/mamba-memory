"""Scene presets — switch MambaMemory between different usage scenarios.

Available presets:
  - fiction: Novel/story writing (characters, plot, world-building)
  - default: Technical/DevOps (the built-in default)

Usage:
    from mamba_memory.presets.fiction import create_fiction_engine

    engine = create_fiction_engine(db_path="~/.mamba-memory/my-novel.db")
    await engine.start()
"""

from mamba_memory.presets.fiction import create_fiction_config, create_fiction_engine

__all__ = ["create_fiction_config", "create_fiction_engine"]
