from __future__ import annotations

import sys as _sys

from tongbench_eval.graph_env.interface import framework_prompt as _impl

globals().update({_name: getattr(_impl, _name) for _name in dir(_impl) if not _name.startswith("__")})
_sys.modules[__name__] = _impl
