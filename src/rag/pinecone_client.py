"""Pinecone client for vector database operations with separate dense/sparse embeddings."""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EmailVector(BaseModel):
    """Data structure for email vector with metadata."""
    id: str
    dense_values: Optional[List[float]] = None
    sparse_values: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any]


class PineconeClient:
    """Client for Pinecone vector database operations with separate dense/sparse indexes."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key. If not provided, will use PINECONE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        if not self.api_key:
            raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        # Separate indexes for dense and sparse vectors
        self.dense_index_name = os.getenv('PINECONE_DENSE_INDEX', 'email-dense-index')
        self.sparse_index_name = os.getenv('PINECONE_SPARSE_INDEX', 'email-sparse-index')
        
        self.dense_index = None
        self.sparse_index = None
        
        logger.info(f"Initialized Pinecone client - Dense: {self.dense_index_name}, Sparse: {self.sparse_index_name}")
    
    def create_dense_index(self, dimension: int = 384, metric: str = "cosine") -> bool:
        """
        Create a new Pinecone index for dense vectors (semantic embeddings).
        
        Args:
            dimension: Dimension of dense vectors (default 384 for sentence-transformers)
            metric: Distance metric to use
            
        Returns:
            bool: True if index was created or already exists
        """
        try:
            # Check if index already exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.dense_index_name in existing_indexes:
                logger.info(f"Dense index '{self.dense_index_name}' already exists")
                return True
            
            # Create new dense index
            logger.info(f"Creating new dense Pinecone index: {self.dense_index_name}")
            self.pc.create_index(
                name=self.dense_index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            logger.info(f"Successfully created dense index: {self.dense_index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating dense Pinecone index: {str(e)}")
            raise
    
    def create_sparse_index(self, dimension: int = 10000, metric: str = "cosine") -> bool:
        """
        Create a new Pinecone index for sparse vectors (keyword/TF-IDF embeddings).
        
        Args:
            dimension: Dimension of sparse vectors (vocabulary size)
            metric: Distance metric to use
            
        Returns:
            bool: True if index was created or already exists
        """
        try:
            # Check if index already exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.sparse_index_name in existing_indexes:
                logger.info(f"Sparse index '{self.sparse_index_name}' already exists")
                return True
            
            # Create new sparse index
            logger.info(f"Creating new sparse Pinecone index: {self.sparse_index_name}")
            self.pc.create_index(
                name=self.sparse_index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            logger.info(f"Successfully created sparse index: {self.sparse_index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating sparse Pinecone index: {str(e)}")
            raise
    
    def create_indexes(self, dense_dimension: int = 384, sparse_dimension: int = 10000) -> bool:
        """
        Create both dense and sparse indexes.
        
        Args:
            dense_dimension: Dimension for dense vectors
            sparse_dimension: Dimension for sparse vectors
            
        Returns:
            bool: True if both indexes were created successfully
        """
        dense_success = self.create_dense_index(dimension=dense_dimension)
        sparse_success = self.create_sparse_index(dimension=sparse_dimension)
        
        return dense_success and sparse_success
    
    def get_dense_index(self):
        """Get the dense Pinecone index instance."""
        if not self.dense_index:
            try:
                self.dense_index = self.pc.Index(self.dense_index_name)
                logger.info(f"Connected to dense index: {self.dense_index_name}")
            except Exception as e:
                logger.error(f"Error connecting to dense index {self.dense_index_name}: {str(e)}")
                raise
        
        return self.dense_index
    
    def get_sparse_index(self):
        """Get the sparse Pinecone index instance."""
        if not self.sparse_index:
            try:
                self.sparse_index = self.pc.Index(self.sparse_index_name)
                logger.info(f"Connected to sparse index: {self.sparse_index_name}")
            except Exception as e:
                logger.error(f"Error connecting to sparse index {self.sparse_index_name}: {str(e)}")
                raise
        
        return self.sparse_index
    
    def upsert_dense_vectors(self, email_vectors: List[EmailVector], 
                           namespace: str = "emails") -> Dict[str, Any]:
        """
        Upsert dense email vectors to dense index.
        
        Args:
            email_vectors: List of EmailVector objects with dense_values
            namespace: Pinecone namespace for organization
            
        Returns:
            Dict with upsert results
        """
        try:
            index = self.get_dense_index()
            
            # Prepare dense vectors for upsert
            vectors = []
            for email_vec in email_vectors:
                if email_vec.dense_values is None:
                    logger.warning(f"Email {email_vec.id} has no dense values, skipping")
                    continue
                    
                vector_data = {
                    "id": email_vec.id,
                    "values": email_vec.dense_values,
                    "metadata": email_vec.metadata
                }
                vectors.append(vector_data)
            
            if not vectors:
                logger.warning("No dense vectors to upsert")
                return {"upserted_count": 0}
            
            # Upsert dense vectors
            result = index.upsert(vectors=vectors, namespace=namespace)
            
            logger.info(f"Successfully upserted {len(vectors)} dense vectors to namespace '{namespace}'")
            return result
            
        except Exception as e:
            logger.error(f"Error upserting dense vectors: {str(e)}")
            raise
    
    def upsert_sparse_vectors(self, email_vectors: List[EmailVector], 
                            namespace: str = "emails") -> Dict[str, Any]:
        """
        Upsert sparse email vectors to sparse index.
        
        Args:
            email_vectors: List of EmailVector objects with sparse_values
            namespace: Pinecone namespace for organization
            
        Returns:
            Dict with upsert results
        """
        try:
            index = self.get_sparse_index()
            
            # Prepare sparse vectors for upsert
            vectors = []
            for email_vec in email_vectors:
                if email_vec.sparse_values is None:
                    logger.warning(f"Email {email_vec.id} has no sparse values, skipping")
                    continue
                
                # Convert sparse dict to dense vector format for Pinecone
                # Create a dense vector with zeros and fill in sparse positions
                sparse_dimension = 10000  # Should match index dimension
                dense_values = [0.0] * sparse_dimension
                
                for idx_str, val in email_vec.sparse_values.items():
                    idx = int(idx_str)
                    if idx < sparse_dimension:
                        dense_values[idx] = val
                
                vector_data = {
                    "id": email_vec.id,
                    "values": dense_values,
                    "metadata": email_vec.metadata
                }
                vectors.append(vector_data)
            
            if not vectors:
                logger.warning("No sparse vectors to upsert")
                return {"upserted_count": 0}
            
            # Upsert sparse vectors
            result = index.upsert(vectors=vectors, namespace=namespace)
            
            logger.info(f"Successfully upserted {len(vectors)} sparse vectors to namespace '{namespace}'")
            return result
            
        except Exception as e:
            logger.error(f"Error upserting sparse vectors: {str(e)}")
            raise
    
    def query_dense_similarity(self, dense_vector: List[float], top_k: int = 10, 
                             namespace: str = "emails",
                             filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query for similar emails using dense semantic search.
        
        Args:
            dense_vector: Dense vector for semantic similarity
            top_k: Number of results to return
            namespace: Pinecone namespace to search
            filter_metadata: Metadata filter for search
            
        Returns:
            Query results from Pinecone dense index
        """
        try:
            index = self.get_dense_index()
            
            query_params = {
                "vector": dense_vector,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": True,
                "include_values": False
            }
            
            # Add metadata filter if provided
            if filter_metadata:
                query_params["filter"] = filter_metadata
            
            results = index.query(**query_params)
            
            logger.info(f"Found {len(results.matches)} similar emails (dense search)")
            return results
            
        except Exception as e:
            logger.error(f"Error querying dense similarity: {str(e)}")
            raise
    
    def query_sparse_similarity(self, sparse_vector: Dict[str, float], top_k: int = 10,
                              namespace: str = "emails", 
                              filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query for similar emails using sparse keyword search.
        
        Args:
            sparse_vector: Sparse vector for keyword matching
            top_k: Number of results to return
            namespace: Pinecone namespace to search
            filter_metadata: Metadata filter for search
            
        Returns:
            Query results from Pinecone sparse index
        """
        try:
            index = self.get_sparse_index()
            
            # Convert sparse dict to dense vector format
            sparse_dimension = 10000  # Should match index dimension
            dense_values = [0.0] * sparse_dimension
            
            for idx_str, val in sparse_vector.items():
                idx = int(idx_str)
                if idx < sparse_dimension:
                    dense_values[idx] = val
            
            query_params = {
                "vector": dense_values,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": True,
                "include_values": False
            }
            
            # Add metadata filter if provided
            if filter_metadata:
                query_params["filter"] = filter_metadata
            
            results = index.query(**query_params)
            
            logger.info(f"Found {len(results.matches)} similar emails (sparse search)")
            return results
            
        except Exception as e:
            logger.error(f"Error querying sparse similarity: {str(e)}")
            raise
    
    def delete_emails(self, email_ids: List[str], namespace: str = "emails") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Delete emails from both indexes.
        
        Args:
            email_ids: List of email IDs to delete
            namespace: Pinecone namespace
            
        Returns:
            Tuple of (dense_result, sparse_result)
        """
        try:
            dense_index = self.get_dense_index()
            sparse_index = self.get_sparse_index()
            
            dense_result = dense_index.delete(ids=email_ids, namespace=namespace)
            sparse_result = sparse_index.delete(ids=email_ids, namespace=namespace)
            
            logger.info(f"Deleted {len(email_ids)} emails from both indexes in namespace '{namespace}'")
            return dense_result, sparse_result
            
        except Exception as e:
            logger.error(f"Error deleting emails: {str(e)}")
            raise
    
    def get_index_stats(self, namespace: str = "emails") -> Dict[str, Any]:
        """
        Get statistics about both Pinecone indexes.
        
        Args:
            namespace: Pinecone namespace
            
        Returns:
            Combined index statistics
        """
        try:
            dense_index = self.get_dense_index()
            sparse_index = self.get_sparse_index()
            
            dense_stats = dense_index.describe_index_stats()
            sparse_stats = sparse_index.describe_index_stats()
            
            dense_namespace_stats = dense_stats.namespaces.get(namespace, {})
            sparse_namespace_stats = sparse_stats.namespaces.get(namespace, {})
            
            dense_vectors = dense_namespace_stats.get('vector_count', 0)
            sparse_vectors = sparse_namespace_stats.get('vector_count', 0)
            
            logger.info(f"Dense index has {dense_vectors} vectors, sparse index has {sparse_vectors} vectors in namespace '{namespace}'")
            
            return {
                "dense_vectors": dense_vectors,
                "sparse_vectors": sparse_vectors,
                "dense_stats": dense_stats,
                "sparse_stats": sparse_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            raise