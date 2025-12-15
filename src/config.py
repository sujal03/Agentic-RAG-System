"""Configuration settings for the AI pipeline."""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Google Gemini API
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/text-embedding-004"
    
    # OpenWeatherMap API
    openweathermap_api_key: str = ""
    
    # LangSmith Configuration
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "ai-pipeline"
    
    # Qdrant Configuration
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "pdf_documents"
    
    # Embedding dimensions for text-embedding-004
    embedding_dimensions: int = 768


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
