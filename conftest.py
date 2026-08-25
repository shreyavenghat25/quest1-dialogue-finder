"""Ensures the project root is importable regardless of how pytest is
invoked (`pytest`, `python -m pytest`, from a different cwd, etc.)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
