from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

from ..base import AgentDecision, VLMBackend
from ..parser import parse_model_decision


class LocalHFQwenBackend(VLMBackend):
    name = "local-hf-qwen"

    def __init__(
        self,
        model_path: str | None = None,
        device_map: str | None = None,
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.model_path = model_path or os.environ.get("LOCAL_VLM_MODEL_PATH")
        self.device_map = device_map or os.environ.get("LOCAL_VLM_DEVICE", "auto")
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        if not self.model_path:
            raise RuntimeError("Missing model_path or LOCAL_VLM_MODEL_PATH for local-hf-qwen backend")
        self._model = None
        self._processor = None
        self._torch = None
        self._loader_name = ""

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                'Missing local VLM dependencies. Install: pip install "transformers>=4.45.0" torch accelerate qwen-vl-utils'
            ) from exc

        self._torch = torch
        dtype = None if self.torch_dtype == "auto" else getattr(torch, self.torch_dtype)
        processor_cls = getattr(transformers, "AutoProcessor", None)
        if processor_cls is None:
            raise RuntimeError("transformers.AutoProcessor is unavailable in the current environment")
        self._processor = processor_cls.from_pretrained(self.model_path, trust_remote_code=True)

        errors: list[str] = []
        loader_candidates = [
            "Qwen3VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModel",
        ]
        for loader_name in loader_candidates:
            loader_cls = getattr(transformers, loader_name, None)
            if loader_cls is None:
                continue
            try:
                load_kwargs: dict[str, Any] = {
                    "trust_remote_code": True,
                    "device_map": self.device_map,
                }
                if loader_name == "Qwen3VLForConditionalGeneration":
                    load_kwargs["dtype"] = "auto" if self.torch_dtype == "auto" else dtype
                else:
                    load_kwargs["torch_dtype"] = dtype
                model = loader_cls.from_pretrained(self.model_path, **load_kwargs)
            except Exception as exc:
                errors.append(f"{loader_name}: {exc}")
                continue
            if hasattr(model, "generate") or hasattr(model, "chat"):
                self._model = model
                self._loader_name = loader_name
                return
            errors.append(f"{loader_name}: loaded {type(model).__name__} without generate/chat")

        message = "Unable to load a compatible local Qwen VLM model."
        if errors:
            message = f"{message} Attempts: " + " | ".join(errors)
        raise RuntimeError(message)

    def _to_qwen_image_value(self, path: Path) -> str:
        return path.resolve().as_uri()

    def _messages(self, prompt: str, image_paths: list[Path]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for path in image_paths:
            content.append({"type": "image", "image": self._to_qwen_image_value(path)})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _load_pil_images(self, image_paths: list[Path]) -> list[Any] | None:
        if not image_paths:
            return None
        try:
            from PIL import Image
        except ImportError:
            return [str(path) for path in image_paths]
        images: list[Any] = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        return images

    def _prepare_inputs(self, prompt: str, image_paths: list[Path]) -> dict[str, Any]:
        messages = self._messages(prompt, image_paths)
        chat_prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs = None
        video_inputs = None
        video_kwargs: dict[str, Any] = {}
        video_metadata = None
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore

            patch_size = 16
            image_processor = getattr(self._processor, "image_processor", None)
            if image_processor is not None and getattr(image_processor, "patch_size", None):
                patch_size = int(image_processor.patch_size)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if video_inputs is not None:
                videos_only = []
                metadata_only = []
                for item in video_inputs:
                    if isinstance(item, tuple) and len(item) == 2:
                        videos_only.append(item[0])
                        metadata_only.append(item[1])
                    else:
                        videos_only.append(item)
                video_inputs = videos_only
                video_metadata = metadata_only if metadata_only else None
        except Exception:
            image_inputs = self._load_pil_images(image_paths)
            video_inputs = None
            video_kwargs = {}
            video_metadata = None

        processor_kwargs: dict[str, Any] = {
            "text": chat_prompt,
            "return_tensors": "pt",
        }
        if image_inputs:
            processor_kwargs["images"] = image_inputs
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
        if video_metadata:
            processor_kwargs["video_metadata"] = video_metadata
        if video_kwargs:
            processor_kwargs.update(video_kwargs)
        processor_kwargs["do_resize"] = False

        inputs = self._processor(**processor_kwargs)
        if hasattr(self._model, "device"):
            moved_inputs: dict[str, Any] = {}
            for key, value in dict(inputs).items():
                moved_inputs[key] = value.to(self._model.device) if hasattr(value, "to") else value
            return moved_inputs
        return dict(inputs)

    def _decode_generated(self, inputs: dict[str, Any], outputs: Any) -> str:
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return self._processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        trimmed_outputs = []
        for input_token_ids, output_token_ids in zip(input_ids, outputs):
            trimmed_outputs.append(output_token_ids[len(input_token_ids) :])
        return self._processor.batch_decode(
            trimmed_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _generate_with_generate(self, prompt: str, image_paths: list[Path]) -> str:
        inputs = self._prepare_inputs(prompt, image_paths)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        return self._decode_generated(inputs, outputs)

    def _extract_chat_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("text", "response", "content", "output"):
                if value.get(key) is not None:
                    return self._extract_chat_text(value.get(key))
            return str(value)
        if isinstance(value, (list, tuple)):
            if not value:
                return ""
            return self._extract_chat_text(value[0])
        return str(value)

    def _call_chat(self, prompt: str, image_paths: list[Path]) -> str:
        chat_method = getattr(self._model, "chat")
        messages = self._messages(prompt, image_paths)
        image_strings = [self._to_qwen_image_value(path) for path in image_paths]

        call_attempts = [
            {"messages": messages, "processor": self._processor, "max_new_tokens": self.max_new_tokens, "temperature": self.temperature},
            {"msgs": messages, "processor": self._processor, "max_new_tokens": self.max_new_tokens, "temperature": self.temperature},
            {"tokenizer": self._processor, "messages": messages, "max_new_tokens": self.max_new_tokens, "temperature": self.temperature},
            {"tokenizer": self._processor, "msgs": messages, "max_new_tokens": self.max_new_tokens, "temperature": self.temperature},
            {"processor": self._processor, "query": prompt, "images": image_strings, "max_new_tokens": self.max_new_tokens, "temperature": self.temperature},
            {"tokenizer": self._processor, "query": prompt, "images": image_strings, "max_new_tokens": self.max_new_tokens, "temperature": self.temperature},
            {"prompt": prompt, "images": image_strings, "processor": self._processor, "max_new_tokens": self.max_new_tokens, "temperature": self.temperature},
            {"text": prompt, "images": image_strings, "processor": self._processor, "max_new_tokens": self.max_new_tokens, "temperature": self.temperature},
        ]

        signature = None
        try:
            signature = inspect.signature(chat_method)
        except (TypeError, ValueError):
            signature = None

        errors: list[str] = []
        for kwargs in call_attempts:
            filtered_kwargs = dict(kwargs)
            if signature is not None:
                parameter_names = set(signature.parameters.keys())
                filtered_kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key in parameter_names
                }
                if not filtered_kwargs:
                    continue
            try:
                response = chat_method(**filtered_kwargs)
                return self._extract_chat_text(response)
            except TypeError as exc:
                errors.append(str(exc))
            except Exception as exc:
                errors.append(str(exc))

        raise RuntimeError(
            "Unable to invoke chat() on local Qwen model. "
            + (" | ".join(errors) if errors else "No compatible chat signature found.")
        )

    def _infer_raw_response(
        self,
        prompt: str,
        image_paths: list[Path],
    ) -> str:
        if hasattr(self._model, "generate"):
            return self._generate_with_generate(prompt, image_paths)
        if hasattr(self._model, "chat"):
            return self._call_chat(prompt, image_paths)
        raise RuntimeError(
            f"Loaded model {type(self._model).__name__} does not expose generate() or chat(). "
            f"Loader used: {self._loader_name or 'unknown'}"
        )

    def choose_action(
        self,
        observation: dict[str, Any],
        prompt: str,
        image_paths: list[Path],
    ) -> AgentDecision:
        self._load()
        try:
            raw_response = self._infer_raw_response(prompt, image_paths)
        except Exception as exc:
            raw_response = str(exc)
        return parse_model_decision(
            raw_response=raw_response,
            action_interface=observation.get("action_interface", "full_action"),
            candidate_action_ids=[item["action_id"] for item in observation.get("candidate_actions", [])],
            allowed_action_types=[item["action_type"] for item in observation.get("action_types", [])],
            object_choice_ids=[item["object_id"] for item in observation.get("object_choices", [])],
            action_type_requirements={
                item["action_type"]: list(item.get("required_args", []))
                for item in observation.get("action_types", [])
            },
            action_candidates=observation.get("model_facing_action_candidates", []),
        )
