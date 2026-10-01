from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _to_file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _build_messages(prompt: str, image_path: Path | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image_path is not None:
        content.append({"type": "image", "image": _to_file_uri(image_path)})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _load_model_and_processor(model_path: str, device_map: str, torch_dtype: str):
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            'Missing dependencies. Install: pip install "transformers>=4.45.0" torch accelerate qwen-vl-utils pillow'
        ) from exc

    processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    dtype = None if torch_dtype == "auto" else getattr(torch, torch_dtype)
    errors: list[str] = []
    loader_names = [
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ]
    for loader_name in loader_names:
        loader_cls = getattr(transformers, loader_name, None)
        if loader_cls is None:
            continue
        try:
            load_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "device_map": device_map,
            }
            if loader_name == "Qwen3VLForConditionalGeneration":
                load_kwargs["dtype"] = "auto" if torch_dtype == "auto" else dtype
            else:
                load_kwargs["torch_dtype"] = dtype
            model = loader_cls.from_pretrained(model_path, **load_kwargs)
            return model, processor, loader_name
        except Exception as exc:
            errors.append(f"{loader_name}: {exc}")
    raise RuntimeError("Unable to load Qwen3-VL model. Attempts: " + " | ".join(errors))


def _prepare_inputs(processor, messages: list[dict[str, Any]], image_path: Path | None) -> dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = None
    video_inputs = None
    video_kwargs: dict[str, Any] = {}
    video_metadata = None

    try:
        from qwen_vl_utils import process_vision_info  # type: ignore

        patch_size = 16
        image_processor = getattr(processor, "image_processor", None)
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
        if image_path is not None:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("Pillow is required for image fallback loading") from exc
            with Image.open(image_path) as image:
                image_inputs = [image.convert("RGB").copy()]

    kwargs: dict[str, Any] = {
        "text": text,
        "return_tensors": "pt",
        "do_resize": False,
    }
    if image_inputs:
        kwargs["images"] = image_inputs
    if video_inputs:
        kwargs["videos"] = video_inputs
    if video_metadata:
        kwargs["video_metadata"] = video_metadata
    if video_kwargs:
        kwargs.update(video_kwargs)
    return dict(processor(**kwargs))


def _decode_generated(processor, inputs: dict[str, Any], outputs: Any) -> str:
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    trimmed_outputs = []
    for input_token_ids, output_token_ids in zip(input_ids, outputs):
        trimmed_outputs.append(output_token_ids[len(input_token_ids) :])
    return processor.batch_decode(
        trimmed_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone local Qwen3-VL generation check")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image")
    parser.add_argument("--prompt", default='Reply with a JSON object like {"status":"ok","object":"..."} based on the input.')
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    image_path = Path(args.image).resolve() if args.image else None
    if image_path is not None and not image_path.exists():
        raise SystemExit(f"Image path does not exist: {image_path}")

    model, processor, loader_name = _load_model_and_processor(args.model_path, args.device, args.torch_dtype)
    messages = _build_messages(args.prompt, image_path)
    inputs = _prepare_inputs(processor, messages, image_path)
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    text = _decode_generated(processor, inputs, outputs)
    print(
        json.dumps(
            {
                "loader": loader_name,
                "model_class": type(model).__name__,
                "processor_class": type(processor).__name__,
                "image_used": str(image_path) if image_path else None,
                "raw_output": text,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
