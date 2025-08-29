# backend/config.py
import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional


class Settings(BaseSettings):
    """
    Global project configuration loaded from environment variables (.env file).
    Provides all tunable parameters like file upload limits, OCR paths, 
    and LLM provider keys.
    """

    # === App Info ===
    APP_NAME: str = "Marksheet Extraction API"
    APP_VERSION: str = "0.1.0"

    # === Uploads ===
    MAX_FILE_MB: int = 10
    ALLOWED_MIME_TYPES: List[str] = Field(
        default_factory=lambda: [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/jpg",
            "application/pdf",
        ]
    )

    # === OCR / PDF Processing ===
    # Example Windows paths: 
    #   TESSERACT_CMD="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    #   POPPLER_PATH="C:\\tools\\poppler-xx\\bin"
    TESSERACT_CMD: Optional[str] = os.getenv("TESSERACT_CMD")
    POPPLER_PATH: Optional[str] = os.getenv("POPPLER_PATH")

    # === LLM Provider ===
    # Default → OpenAI (you can extend later for Azure, Anthropic, etc.)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # === API Key Authentication (optional) ===
    ENABLE_API_KEY: bool = os.getenv("ENABLE_API_KEY", "false").lower() == "true"
    API_KEY_HEADER: str = "x-api-key"
    API_KEY_VALUE: Optional[str] = os.getenv("API_KEY_VALUE")

    # === CORS (Frontend integration) ===
    CORS_ALLOW_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create a global settings instance
settings = Settings()
