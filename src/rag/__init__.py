"""RAG (Retrieval-Augmented Generation) package for email processing and search."""

from .pinecone_client import PineconeClient
from .embeddings import EmbeddingService

__all__ = ['PineconeClient', 'EmbeddingService']