from __future__ import annotations

from pathlib import Path as _Path

_PART_DIR = _Path(__file__).with_name("parts")
_SOURCE_PARTS = (
    _PART_DIR / "part_01.pyfrag",
    _PART_DIR / "part_02.pyfrag",
    _PART_DIR / "part_03.pyfrag",
    _PART_DIR / "part_04.pyfrag",

)
_source = "".join(_path.read_text(encoding="utf-8") for _path in _SOURCE_PARTS)
exec(compile(_source, str(_PART_DIR / "workspace_env__combined__.py"), "exec"), globals(), globals())
