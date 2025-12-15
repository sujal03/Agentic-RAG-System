"""Embedding service using Google's text-embedding model."""
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import get_settings


class EmbeddingService:
    """Service for generating embeddings using Google's embedding model."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the embedding service.
        
        Args:
            api_key: Google API key. If not provided, uses env variable.
        """
        settings = get_settings()
        self.api_key = api_key or settings.google_api_key
        self.model_name = settings.gemini_embedding_model
        self.dimensions = settings.embedding_dimensions
        
        if not self.api_key:
            raise ValueError("Google API key is required for embeddings")
        
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=self.api_key
        )
    
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        return self._embeddings.embed_query(text)
    
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents.
        
        Args:
            documents: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self._embeddings.embed_documents(documents)
    
    @property
    def langchain_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Get the underlying LangChain embeddings object.
        
        Returns:
            GoogleGenerativeAIEmbeddings instance for use with LangChain
        """
        return self._embeddings
