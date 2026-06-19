"""Pipeline B: Video OCR using ``Qwen/Qwen2.5-VL-7B-Instruct`` via transformers."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from . import free_memory, get_device_and_dtype, get_logger
from .audio_extractor import validate_video
from .frame_extractor import extract_frames
from .models import QWEN_MODEL_ID, OCRFrameResult, OCRResult

log = get_logger("ocr")

# Sentinel the model is told to emit for frames containing no text.
_NO_TEXT = "NO_TEXT"

OCR_PROMPT = (
    "You are a precise OCR engine. Transcribe ALL text visible in this image "
    "exactly as it appears, preserving line breaks and reading order. "
    "Do not translate, summarise, or describe the image. "
    f"If the image contains no readable text, respond with exactly: {_NO_TEXT}"
)


def _normalise(text: str) -> str:
    """Normalise text for dedup comparison (lowercase, collapsed whitespace)."""
    return re.sub(r"\s+", " ", text.strip().lower())


def deduplicate_frames(texts: list[str]) -> tuple[list[str], str]:
    """Deduplicate OCR text across frames.

    Args:
        texts: Per-frame OCR text (empty string for frames with no text).

    Returns:
        ``(unique_blocks, combined_text)`` where ``unique_blocks`` are whole-frame
        text blocks with consecutive/repeat duplicates removed, and ``combined_text``
        is a line-level deduplicated concatenation preserving first-seen order.
    """
    unique_blocks: list[str] = []
    seen_blocks: set[str] = set()
    for text in texts:
        text = text.strip()
        if not text:
            continue
        key = _normalise(text)
        if key in seen_blocks:
            continue
        seen_blocks.add(key)
        unique_blocks.append(text)

    seen_lines: set[str] = set()
    combined_lines: list[str] = []
    for block in unique_blocks:
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            key = _normalise(line)
            if key in seen_lines:
                continue
            seen_lines.add(key)
            combined_lines.append(line)

    return unique_blocks, "\n".join(combined_lines)


class OCRPipeline:
    """Loads Qwen2.5-VL once and runs OCR over sampled video frames.

    Use as a context manager so the model is unloaded on exit::

        with OCRPipeline() as op:
            result = op.run("clip.mp4", fps=1.0)
    """

    def __init__(
        self,
        model_id: str = QWEN_MODEL_ID,
        max_new_tokens: int = 512,
        load_in_4bit: bool = True,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        # 4-bit (nf4) quantisation keeps the 7B model under ~6 GB so it fits a
        # 16 GB GPU (e.g. a T4). Set False on a larger GPU for full fp16 weights.
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._processor = None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        """Load the Qwen2.5-VL model and processor."""
        if self._model is not None:
            return
        from transformers import AutoProcessor, AutoModelForImageTextToText # Qwen2_5_VLForConditionalGeneration

        device, dtype = get_device_and_dtype()
        log.info("Loading %s on %s (%s)", self.model_id, device, dtype)
        load_kwargs: dict = {"dtype": dtype}
        if device.startswith("cuda"):
            # Pin to the chosen compatible GPU rather than spreading across all.
            load_kwargs["device_map"] = device
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
        # self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     self.model_id, **load_kwargs
        # )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **load_kwargs
        )
        if not device.startswith("cuda"):
            self._model = self._model.to(device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def unload(self) -> None:
        """Release the model and free GPU memory."""
        if self._model is not None:
            log.info("Unloading Qwen2.5-VL model")
            del self._model
            self._model = None
        self._processor = None
        free_memory()

    def __enter__(self) -> "OCRPipeline":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    # -- inference ---------------------------------------------------------
    def ocr_image(self, image_path: Path) -> str:
        """Run OCR on a single image file and return the detected text."""
        if self._model is None or self._processor is None:
            self.load()
        assert self._model is not None and self._processor is not None

        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ]
        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            # processor_kwargs={
            #     "padding": True,
            #     # For SmolVLM, if you want to pass downsampling or formatting configs:
            #     # "do_pad": False, 
            # }
            # padding=True,
        ).to(self._model.device)

        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        if decoded.upper().strip() == _NO_TEXT or not decoded:
            return ""
        return decoded

    def run(self, video_path: Path | str, fps: float = 1.0) -> OCRResult:
        """Extract frames at *fps*, OCR each, and return a deduplicated result."""
        video_path = validate_video(Path(video_path))
        frames_dir = Path(tempfile.mkdtemp(prefix="vp_ocr_frames_"))
        try:
            frames = extract_frames(video_path, fps=fps, output_dir=frames_dir)
            frame_results: list[OCRFrameResult] = []
            raw_texts: list[str] = []
            for frame in frames:
                log.info("OCR frame %d/%d (t=%.2fs)", frame.frame_number, len(frames), frame.timestamp)
                text = self.ocr_image(frame.path)
                raw_texts.append(text)
                frame_results.append(
                    OCRFrameResult(
                        frame_number=frame.frame_number,
                        timestamp=frame.timestamp,
                        text=text,
                    )
                )
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)

        unique_blocks, combined = deduplicate_frames(raw_texts)
        result = OCRResult(
            video_path=str(video_path),
            model=self.model_id,
            fps=fps,
            frame_count=len(frame_results),
            text=combined,
            unique_text_blocks=unique_blocks,
            frames=frame_results,
        )
        log.info(
            "OCR complete: %d frames, %d unique text blocks",
            len(frame_results),
            len(unique_blocks),
        )
        return result


def ocr_video(
    video_path: Path | str,
    fps: float = 1.0,
    max_new_tokens: int = 512,
) -> OCRResult:
    """Convenience wrapper: load, OCR one video, and unload."""
    with OCRPipeline(max_new_tokens=max_new_tokens) as op:
        return op.run(video_path, fps=fps)
