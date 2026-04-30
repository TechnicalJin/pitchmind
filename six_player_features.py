"""Compatibility shim for legacy imports.

The dashboard still tries to import `six_player_features` in one fallback path.
This module re-exports the current player feature engine so that import resolves
cleanly without duplicating logic.
"""

from pitchmind_player_features import *
