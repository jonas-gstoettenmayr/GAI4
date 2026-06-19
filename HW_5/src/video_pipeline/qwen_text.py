"""Text-only language tasks with ``Qwen/Qwen2.5-VL-7B-Instruct``.

Qwen2.5-VL is a vision-language model, but it is also a capable text chat model.
This module drives it **without any image input** to translate and summarise the
Whisper transcript produced by :mod:`video_pipeline.whisper_pipeline`.

The model is loaded once and unloaded (with the GPU cache cleared) on exit, using
the same lifecycle discipline as :class:`video_pipeline.ocr_pipeline.OCRPipeline`,
so Whisper and Qwen never occupy GPU memory at the same time.
"""

from __future__ import annotations

from . import free_memory, get_device_and_dtype, get_logger
from .models import QWEN_MODEL_ID, SummaryResult, TranslationResult

log = get_logger("qwen_text")

TRANSLATE_SYSTEM = (
    "You are a professional translator. Translate the user's text from "
    "{source} into {target}. Preserve meaning, names and numbers. "
    "Respond with the translation ONLY — no preamble, notes or explanation."
)

SUMMARY_SYSTEM = (
    "You are a precise summariser. Write a clear, faithful summary of the "
    "user's text in {language}. Capture the main points and key facts; do not "
    "add information that is not present. Respond with the summary ONLY."
)


class QwenChat:
    """Loads Qwen2.5-VL once and runs text-only chat (translate / summarise).

    Use as a context manager so the model is unloaded on exit::

        with QwenChat() as qc:
            english = qc.translate(german_text, source="German", target="English")
            summary = qc.summarize(english)
    """

    def __init__(
        self,
        model_id: str = QWEN_MODEL_ID,
        max_new_tokens: int = 1024,
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
        from transformers import AutoProcessor, AutoModelForImageTextToText #Qwen2_5_VLForConditionalGeneration

        device, dtype = get_device_and_dtype()
        log.info("Loading %s on %s (%s)", self.model_id, device, dtype)
        load_kwargs: dict = {"dtype": dtype}
        if device.startswith("cuda"):
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

    def __enter__(self) -> "QwenChat":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    # -- inference ---------------------------------------------------------
    def chat(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        """Run a single text-only system+user turn and return the reply."""
        if self._model is None or self._processor is None:
            self.load()
        assert self._model is not None and self._processor is not None

        import torch

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[chat_text],
            return_tensors="pt",
            # processor_kwargs={
            #     "padding": True,
            #     # For SmolVLM, if you want to pass downsampling or formatting configs:
            #     # "do_pad": False, 
            # }
            # # padding=True,
        ).to(self._model.device)

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return decoded

    def translate(
        self, text: str, source: str = "German", target: str = "English"
    ) -> str:
        """Translate *text* from *source* into *target* and return the translation."""
        log.info("Translating %d chars: %s -> %s", len(text), source, target)
        system = TRANSLATE_SYSTEM.format(source=source, target=target)
        return self.chat(system, text)

    def summarize(self, text: str, language: str = "English") -> str:
        """Summarise *text* in *language* and return the summary."""
        log.info("Summarising %d chars in %s", len(text), language)
        system = SUMMARY_SYSTEM.format(language=language)
        return self.chat(system, text)


def translate_text(
    text: str,
    source_path: str,
    source: str = "German",
    target: str = "English",
    max_new_tokens: int = 1024,
) -> TranslationResult:
    """Convenience wrapper: load Qwen, translate one text, and unload."""
    with QwenChat(max_new_tokens=max_new_tokens) as qc:
        translated = qc.translate(text, source=source, target=target)
    return TranslationResult(
        source_path=source_path,
        source_language=source,
        target_language=target,
        source_text=text,
        text=translated,
    )


def summarize_text(
    text: str,
    source_path: str,
    language: str = "English",
    max_new_tokens: int = 1024,
) -> SummaryResult:
    """Convenience wrapper: load Qwen, summarise one text, and unload."""
    with QwenChat(max_new_tokens=max_new_tokens) as qc:
        summary = qc.summarize(text, language=language)
    return SummaryResult(
        source_path=source_path,
        language=language,
        source_text=text,
        summary=summary,
    )
