"""Scene presets — switch MambaMemory between different usage scenarios.

Available presets:
  - fiction:  Novel/story writing
  - medical:  Clinical/healthcare
  - finance:  Accounting/financial
  - legal:    Law/litigation
  - sales:    CRM/sales pipeline
  - default:  Technical/DevOps (built-in)

Usage:
    from mamba_memory.presets import create_fiction_engine
    from mamba_memory.presets import create_medical_engine
    from mamba_memory.presets import create_finance_engine
    from mamba_memory.presets import create_legal_engine
    from mamba_memory.presets import create_sales_engine
"""

from mamba_memory.presets.fiction import create_fiction_config, create_fiction_engine
from mamba_memory.presets.finance import create_finance_config, create_finance_engine
from mamba_memory.presets.legal import create_legal_config, create_legal_engine
from mamba_memory.presets.medical import create_medical_config, create_medical_engine
from mamba_memory.presets.sales import create_sales_config, create_sales_engine

__all__ = [
    "create_fiction_config", "create_fiction_engine",
    "create_medical_config", "create_medical_engine",
    "create_finance_config", "create_finance_engine",
    "create_legal_config", "create_legal_engine",
    "create_sales_config", "create_sales_engine",
]
