"""
Indices BoSC alignés sur ``agent/preprocessor.py`` (SYSCALL_TABLE).

Évite les incohérences historiques où ptrace/keyctl/etc. pointaient vers les
mauvaises dimensions hors table noyau.
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_PKG = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "agent"))
if _AGENT_PKG not in sys.path:
    sys.path.insert(0, _AGENT_PKG)

from preprocessor import REVERSE_SYSCALL_TABLE  # noqa: E402
from preprocessor import SYSCALL_TABLE as SC  # noqa: E402

__all__ = ["SC", "REVERSE_SYSCALL_TABLE"]
