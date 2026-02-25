"""Compatibility shim for joblib artifacts serialized under scripts.models.*

The .joblib files contain pickled objects that reference 'scripts.models.X'
as their module path (because they were trained in the hack-week repo where
code lives at scripts/models/). This shim creates fake 'scripts' and
'scripts.models' entries in sys.modules pointing to the current directory,
so pickle can resolve those references without error.

Import this module before loading any .joblib file.
"""
import sys
import types
from pathlib import Path

_this_dir = str(Path(__file__).parent)

if "scripts" not in sys.modules:
    _scripts = types.ModuleType("scripts")
    _scripts.__path__ = [_this_dir]
    sys.modules["scripts"] = _scripts

if "scripts.models" not in sys.modules:
    _scripts_models = types.ModuleType("scripts.models")
    _scripts_models.__path__ = [_this_dir]
    sys.modules["scripts.models"] = _scripts_models
