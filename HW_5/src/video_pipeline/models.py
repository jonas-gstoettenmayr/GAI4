"""Pydantic data models for pipeline inputs and outputs."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

# WHISPER_MODEL_ID = "openai/whisper-large-v3"
WHISPER_MODEL_ID = "openai/whisper-tiny" # 0.6B params
# QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Pipeline A: Speech-to-Text
# --------------------------------------------------------------------------
class TranscriptSegment(BaseModel):
    """A single timestamped chunk of transcribed speech."""

    id: int
    start: float | None = Field(None, description="Segment start time in seconds")
    end: float | None = Field(None, description="Segment end time in seconds")
    text: str


class WhisperResult(BaseModel):
    """Output of the Whisper speech-to-text pipeline for one video."""

    video_path: str
    model: str = WHISPER_MODEL_ID
    language: str | None = None
    duration: float | None = Field(None, description="Audio duration in seconds")
    text: str = Field(..., description="Full transcript text")
    segments: list[TranscriptSegment] = Field(default_factory=list)
    generated_at: str = Field(default_factory=_utc_now)


# --------------------------------------------------------------------------
# Pipeline B: Video OCR
# --------------------------------------------------------------------------
class OCRFrameResult(BaseModel):
    """OCR result for a single extracted video frame."""

    frame_number: int = Field(..., description="1-based index of the extracted frame")
    timestamp: float = Field(..., description="Frame time in the source video, seconds")
    text: str = Field(..., description="Text detected in this frame")


class OCRResult(BaseModel):
    """Output of the Qwen2.5-VL OCR pipeline for one video."""

    video_path: str
    model: str = QWEN_MODEL_ID
    fps: float = Field(..., description="Frames sampled per second of video")
    frame_count: int = 0
    text: str = Field("", description="Deduplicated text across all frames")
    unique_text_blocks: list[str] = Field(default_factory=list)
    frames: list[OCRFrameResult] = Field(default_factory=list)
    generated_at: str = Field(default_factory=_utc_now)


# --------------------------------------------------------------------------
# Pipeline C: Translation (Qwen2.5-VL, text-only)
# --------------------------------------------------------------------------
class TranslationResult(BaseModel):
    """Output of the Qwen2.5-VL translation pipeline for one transcript."""

    source_path: str = Field(..., description="Transcript JSON the text came from")
    model: str = QWEN_MODEL_ID
    source_language: str = Field(..., description="Language of the input text")
    target_language: str = Field(..., description="Language of the translation")
    source_text: str = Field(..., description="Original text fed to the model")
    text: str = Field(..., description="Translated text")
    generated_at: str = Field(default_factory=_utc_now)


# --------------------------------------------------------------------------
# Pipeline D: Summarisation (Qwen2.5-VL, text-only)
# --------------------------------------------------------------------------
class SummaryResult(BaseModel):
    """Output of the Qwen2.5-VL summarisation pipeline for one transcript."""

    source_path: str = Field(..., description="Transcript/translation JSON the text came from")
    model: str = QWEN_MODEL_ID
    language: str = Field(..., description="Language of the input text")
    source_text: str = Field(..., description="Original text fed to the model")
    summary: str = Field(..., description="Generated summary")
    generated_at: str = Field(default_factory=_utc_now)
