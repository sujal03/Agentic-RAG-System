"""RAG agent for handling PDF document queries."""
from typing import Optional, BinaryIO
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.services.vector_store import VectorStoreService


class RAGAgent:
    """Agent that handles PDF document queries using RAG."""
    
    RAG_PROMPT = """You are a helpful assistant that answers questions based on the provided context from PDF documents.

Context from documents:
{context}

User Question: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information to answer fully, say so
3. Quote relevant parts of the context when helpful
4. Be concise but thorough
5. If the question cannot be answered from the context, explain what information is available instead

Answer:"""

    def __init__(self, vector_store: Optional[VectorStoreService] = None):
        """Initialize the RAG agent.
        
        Args:
            vector_store: VectorStoreService instance. Created if not provided.
        """
        settings = get_settings()
        
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.3
        )
        
        self._vector_store = vector_store or VectorStoreService()
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        self._prompt = ChatPromptTemplate.from_template(self.RAG_PROMPT)
    
    def load_pdf(self, pdf_path: str) -> list[Document]:
        """Load and split a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document chunks
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split into chunks
        chunks = self._text_splitter.split_documents(documents)
        
        # Add source metadata
        for chunk in chunks:
            chunk.metadata["source_file"] = Path(pdf_path).name
        
        return chunks
    
    def load_pdf_from_bytes(self, pdf_bytes: BinaryIO, filename: str) -> list[Document]:
        """Load PDF from bytes (for Streamlit file upload).
        
        Args:
            pdf_bytes: File-like object containing PDF bytes
            filename: Original filename
            
        Returns:
            List of Document chunks
        """
        import tempfile
        import os
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes.read())
            tmp_path = tmp_file.name
        
        try:
            chunks = self.load_pdf(tmp_path)
            # Update source metadata
            for chunk in chunks:
                chunk.metadata["source_file"] = filename
            return chunks
        finally:
            os.unlink(tmp_path)
    
    def index_documents(self, documents: list[Document]) -> int:
        """Index documents in the vector store.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            Number of documents indexed
        """
        self._vector_store.add_documents(documents)
        return len(documents)
    
    def retrieve_context(self, query: str, k: int = 4) -> list[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        return self._vector_store.similarity_search(query, k=k)
    
    def _format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into context string.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "?")
            context_parts.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def answer_question(self, question: str, k: int = 4) -> dict:
        """Answer a question using RAG.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        # Retrieve relevant documents
        documents = self.retrieve_context(question, k=k)
        
        if not documents:
            return {
                "response": "I don't have any documents indexed yet. Please upload a PDF first.",
                "sources": [],
                "context_used": ""
            }
        
        # Format context
        context = self._format_context(documents)
        
        # Generate response
        chain = self._prompt | self._llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "question": question
        })
        
        # Extract sources
        sources = list(set([
            doc.metadata.get("source_file", "Unknown") 
            for doc in documents
        ]))
        
        return {
            "response": response,
            "sources": sources,
            "context_used": context,
            "num_chunks_retrieved": len(documents)
        }
    
    def process(self, query: str) -> dict:
        """Process a PDF-related query.
        
        Args:
            query: User's question about the document
            
        Returns:
            Dictionary with response and metadata
        """
        result = self.answer_question(query)
        return {
            "response": result["response"],
            "sources": result["sources"],
            "success": len(result["sources"]) > 0
        }
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the indexed documents.
        
        Returns:
            Dictionary with collection information
        """
        return self._vector_store.get_collection_info()
