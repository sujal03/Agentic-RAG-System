"""Vector store service for Qdrant operations."""
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from src.config import get_settings
from src.services.embeddings import EmbeddingService


class VectorStoreService:
    """Service for managing document storage and retrieval in Qdrant."""
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        collection_name: Optional[str] = None
    ):
        """Initialize the vector store service.
        
        Args:
            embedding_service: EmbeddingService instance. Created if not provided.
            collection_name: Name of the Qdrant collection. Uses config if not provided.
        """
        settings = get_settings()
        
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Initialize Qdrant client
        if settings.qdrant_url and settings.qdrant_api_key:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
            # Use in-memory Qdrant for development
            self._client = QdrantClient(":memory:")
        
        self._vector_store: Optional[QdrantVectorStore] = None
    
    def create_collection(self, recreate: bool = False) -> None:
        """Create the Qdrant collection if it doesn't exist.
        
        Args:
            recreate: If True, delete existing collection and create new one
        """
        settings = get_settings()
        
        # Check if collection exists
        collections = self._client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self.collection_name in collection_names:
            if recreate:
                self._client.delete_collection(self.collection_name)
            else:
                return
        
        # Create collection
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE
            )
        )
    
    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of document IDs
        """
        self.create_collection()
        
        # Get LangChain vector store
        vector_store = self._get_vector_store()
        
        # Add documents and return IDs
        return vector_store.add_documents(documents)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> list[Document]:
        """Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of matching Document objects
        """
        vector_store = self._get_vector_store()
        
        if score_threshold:
            return vector_store.similarity_search_with_score(
                query,
                k=k,
                score_threshold=score_threshold
            )
        
        return vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 4
    ) -> list[tuple[Document, float]]:
        """Search for similar documents with similarity scores.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        vector_store = self._get_vector_store()
        return vector_store.similarity_search_with_score(query, k=k)
    
    def _get_vector_store(self) -> QdrantVectorStore:
        """Get or create the LangChain vector store wrapper.
        
        Returns:
            QdrantVectorStore instance
        """
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                client=self._client,
                collection_name=self.collection_name,
                embedding=self.embedding_service.langchain_embeddings
            )
        return self._vector_store
    
    def get_retriever(self, k: int = 4):
        """Get a retriever for the vector store.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            LangChain retriever object
        """
        vector_store = self._get_vector_store()
        return vector_store.as_retriever(search_kwargs={"k": k})
    
    def delete_collection(self) -> None:
        """Delete the collection from Qdrant."""
        self._client.delete_collection(self.collection_name)
    
    def get_collection_info(self) -> dict:
        """Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            info = self._client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception:
            return {"name": self.collection_name, "exists": False}
