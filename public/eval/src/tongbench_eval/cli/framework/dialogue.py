from __future__ import annotations

import sys as _sys

from tongbench_eval.cli.framework.dialogue_runtime import main as _impl

globals().update({_name: getattr(_impl, _name) for _name in dir(_impl) if not _name.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_impl.main())
else:
    _sys.modules[__name__] = _impl
