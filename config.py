"""Application settings loaded from environment and ``.env``.

All critical configuration is validated here at import time via Pydantic.
If a required variable is missing, ``Settings()`` raises ``ValidationError``
immediately, preventing the application from booting with an invalid
configuration. This is the single source of truth for environment-driven
settings across the pipeline.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application settings.

    Attributes:
        NVIDIA_API_KEY: API key for the NVIDIA OpenAI-compatible endpoint.
        HF_TOKEN: Hugging Face token for Pyannote speaker diarization.
        LLM_MODEL_NAME: Default model for both Map and Reduce stages.
        MAP_MODEL: Override for the Stage 1 (Map) vision model.
        REDUCE_MODEL: Override for the Stage 3 (Reduce) reasoning model.
        REDUCE_REASONING_ENABLED: Whether to enable chain-of-thought on the
            REDUCE_MODEL (only for reasoning models).
        WHISPER_COMPUTE_TYPE: CTranslate2 quantization for WhisperX.
        WHISPER_BATCH_SIZE: Whisper decode batch size.
        DRIVE_FOLDER_ID: Optional Google Drive folder ID for sync.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Required (no defaults; boot fails immediately if missing) ---
    NVIDIA_API_KEY: str
    HF_TOKEN: str

    # --- Optional (sensible defaults) ---
    LLM_MODEL_NAME: str = "meta/llama-3.1-8b-instruct"
    MAP_MODEL: str | None = None
    REDUCE_MODEL: str | None = None
    REDUCE_REASONING_ENABLED: bool = False

    # --- Whisper / transcription ---
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_BATCH_SIZE: int = 16

    # --- Google Drive ---
    DRIVE_FOLDER_ID: str | None = None


settings = Settings()
