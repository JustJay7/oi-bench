"""
OI-Bench baseline models.

  lif_network.py : LIFNetwork  — standard LIF, same learning rules as CAdEx
  reservoir.py   : LiquidStateMachine — fixed reservoir, linear readout only
"""
from .lif_network import LIFNetwork
from .reservoir   import LiquidStateMachine

__all__ = ["LIFNetwork", "LiquidStateMachine"]
