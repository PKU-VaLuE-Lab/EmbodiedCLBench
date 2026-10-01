from __future__ import annotations

import os

from .openai_compatible import OpenAICompatibleBackend


class DashScopeQwenBackend(OpenAICompatibleBackend):
    name = "dashscope-qwen"

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        try:
            import dashscope  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "dashscope package is required for dashscope-qwen backend. Install requirements-api-vlm.txt."
            ) from exc
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY for dashscope-qwen backend")
        super().__init__(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            model=model or os.environ.get("DASHSCOPE_MODEL"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
