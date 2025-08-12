"""Email vectorization service that combines Gmail data with Pinecone storage."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from .pinecone_client import PineconeClient, EmailVector
from .embeddings import EmbeddingService
from ..gmail.client import EmailMessage

logger = logging.getLogger(__name__)


class EmailVectorizer:
    """Service for vectorizing emails and storing in Pinecone."""
    
    def __init__(self, pinecone_api_key: Optional[str] = None):
        """
        Initialize email vectorizer.
        
        Args:
            pinecone_api_key: Pinecone API key
        """
        self.pinecone_client = PineconeClient(api_key=pinecone_api_key)
        self.embedding_service = EmbeddingService()
        
        logger.info("Initialized email vectorizer")
    
    def setup_indexes(self) -> bool:
        """
        Set up Pinecone indexes for dense and sparse vectors.
        
        Returns:
            bool: True if successful
        """
        try:
            return self.pinecone_client.create_indexes()
        except Exception as e:
            logger.error(f"Error setting up indexes: {str(e)}")
            raise
    
    def prepare_email_text(self, email: EmailMessage) -> str:
        """
        Prepare email text for embedding by combining relevant fields.
        Uses conditional approach: full body when available, snippet as fallback.
        
        Args:
            email: EmailMessage object
            
        Returns:
            Combined text for embedding
        """
        text_parts = []
        
        # Add subject (with higher weight by including twice)
        if email.subject and email.subject != "No Subject":
            text_parts.append(email.subject)
            text_parts.append(email.subject)  # Double weight for subject
        
        # Conditional approach: use body if available, otherwise use snippet
        if email.body and email.body != "<!DOCTYPE html>...</html>":
            # Use full body when available
            text_parts.append(email.body)
        else:
            # Fallback to snippet when body is unavailable or just HTML placeholder
            if email.snippet:
                text_parts.append(email.snippet)
        
        return " ".join(text_parts)
    
    def create_email_metadata(self, email: EmailMessage) -> Dict[str, Any]:
        """
        Create metadata dict for email.
        
        Args:
            email: EmailMessage object
            
        Returns:
            Metadata dictionary
        """
        return {
            "subject": email.subject,
            "sender": email.sender,
            "recipient": email.recipient,
            "date": email.date.isoformat(),
            "snippet": email.snippet[:200] if email.snippet else "",  # Truncate for metadata
            "labels": email.labels,
            "thread_id": email.thread_id,
            "has_body": bool(email.body and email.body != "<!DOCTYPE html>...</html>"),
            "used_snippet": not bool(email.body and email.body != "<!DOCTYPE html>...</html>"),
            "indexed_at": datetime.now().isoformat()
        }
    
    def vectorize_emails(self, emails: List[EmailMessage]) -> Tuple[List[EmailVector], List[str]]:
        """
        Vectorize a list of emails.
        
        Args:
            emails: List of EmailMessage objects
            
        Returns:
            Tuple of (email_vectors, skipped_email_ids)
        """
        try:
            if not emails:
                return [], []
            
            logger.info(f"Vectorizing {len(emails)} emails")
            
            # Prepare texts for batch embedding
            email_texts = []
            valid_emails = []
            skipped_ids = []
            
            for email in emails:
                email_text = self.prepare_email_text(email)
                if email_text.strip():
                    email_texts.append(email_text)
                    valid_emails.append(email)
                else:
                    skipped_ids.append(email.id)
                    logger.warning(f"Skipping email {email.id} - no meaningful text")
            
            if not email_texts:
                logger.warning("No valid emails to vectorize")
                return [], skipped_ids
            
            # Fit sparse vectorizer if needed (first time or if not cached)
            if not self.embedding_service.load_sparse_vectorizer():
                logger.info("Fitting sparse vectorizer on email corpus")
                self.embedding_service.fit_sparse_vectorizer(email_texts)
            
            # Generate embeddings in batch
            dense_embeddings, sparse_embeddings = self.embedding_service.batch_generate_embeddings(email_texts)
            
            # Create EmailVector objects
            email_vectors = []
            for i, email in enumerate(valid_emails):
                metadata = self.create_email_metadata(email)
                
                email_vector = EmailVector(
                    id=email.id,
                    dense_values=dense_embeddings[i],
                    sparse_values=sparse_embeddings[i],
                    metadata=metadata
                )
                email_vectors.append(email_vector)
            
            logger.info(f"Successfully vectorized {len(email_vectors)} emails")
            return email_vectors, skipped_ids
            
        except Exception as e:
            logger.error(f"Error vectorizing emails: {str(e)}")
            raise
    
    def store_email_vectors(self, email_vectors: List[EmailVector], 
                          namespace: str = "emails") -> Dict[str, Any]:
        """
        Store email vectors in Pinecone indexes.
        
        Args:
            email_vectors: List of EmailVector objects
            namespace: Pinecone namespace
            
        Returns:
            Storage results
        """
        try:
            if not email_vectors:
                return {"dense_upserted": 0, "sparse_upserted": 0}
            
            logger.info(f"Storing {len(email_vectors)} email vectors to Pinecone")
            
            # Store dense vectors
            dense_result = self.pinecone_client.upsert_dense_vectors(email_vectors, namespace)
            
            # Store sparse vectors
            sparse_result = self.pinecone_client.upsert_sparse_vectors(email_vectors, namespace)
            
            return {
                "dense_upserted": dense_result.get("upserted_count", 0),
                "sparse_upserted": sparse_result.get("upserted_count", 0),
                "dense_result": dense_result,
                "sparse_result": sparse_result
            }
            
        except Exception as e:
            logger.error(f"Error storing email vectors: {str(e)}")
            raise
    
    def process_and_store_emails(self, emails: List[EmailMessage], 
                               namespace: str = "emails") -> Dict[str, Any]:
        """
        Complete pipeline: vectorize emails and store in Pinecone.
        
        Args:
            emails: List of EmailMessage objects
            namespace: Pinecone namespace
            
        Returns:
            Processing results
        """
        try:
            # Ensure indexes exist
            self.setup_indexes()
            
            # Filter out duplicates by checking existing vectors
            unique_emails = self.filter_duplicate_emails(emails, namespace)
            
            if not unique_emails:
                logger.info("No new emails to process (all duplicates)")
                return {
                    "total_emails": len(emails),
                    "new_emails": 0,
                    "skipped_duplicates": len(emails),
                    "vectorized": 0,
                    "stored": 0
                }
            
            # Vectorize emails
            email_vectors, skipped_ids = self.vectorize_emails(unique_emails)
            
            # Store vectors
            storage_results = self.store_email_vectors(email_vectors, namespace)
            
            return {
                "total_emails": len(emails),
                "new_emails": len(unique_emails),
                "skipped_duplicates": len(emails) - len(unique_emails),
                "vectorized": len(email_vectors),
                "skipped_no_text": len(skipped_ids),
                "dense_stored": storage_results.get("dense_upserted", 0),
                "sparse_stored": storage_results.get("sparse_upserted", 0)
            }
            
        except Exception as e:
            logger.error(f"Error in process_and_store_emails: {str(e)}")
            raise
    
    def filter_duplicate_emails(self, emails: List[EmailMessage], 
                              namespace: str = "emails") -> List[EmailMessage]:
        """
        Filter out emails that already exist in Pinecone.
        
        Args:
            emails: List of EmailMessage objects
            namespace: Pinecone namespace
            
        Returns:
            List of emails that don't exist in Pinecone
        """
        try:
            if not emails:
                return []
            
            # Get existing email IDs from Pinecone by querying stats
            # This is a simple approach - for large datasets, you might want to batch query
            unique_emails = []
            
            for email in emails:
                # Try to query for this specific email ID
                try:
                    # Query dense index for this email ID
                    dense_index = self.pinecone_client.get_dense_index()
                    result = dense_index.fetch(ids=[email.id], namespace=namespace)
                    
                    if email.id not in result.vectors:
                        # Email doesn't exist, add to unique list
                        unique_emails.append(email)
                    else:
                        logger.debug(f"Skipping duplicate email: {email.id}")
                        
                except Exception as e:
                    # If error fetching, assume it doesn't exist and include it
                    logger.debug(f"Error checking email {email.id}, including it: {str(e)}")
                    unique_emails.append(email)
            
            logger.info(f"Filtered {len(emails)} emails to {len(unique_emails)} unique emails")
            return unique_emails
            
        except Exception as e:
            logger.error(f"Error filtering duplicate emails: {str(e)}")
            # If error, return all emails to be safe
            return emails
    
    def search_similar_emails(self, query_text: str, top_k: int = 10,
                            use_dense: bool = True, use_sparse: bool = True,
                            namespace: str = "emails") -> Dict[str, Any]:
        """
        Search for similar emails using dense and/or sparse search.
        
        Args:
            query_text: Search query
            top_k: Number of results
            use_dense: Use dense (semantic) search
            use_sparse: Use sparse (keyword) search
            namespace: Pinecone namespace
            
        Returns:
            Search results
        """
        try:
            results = {}
            
            if use_dense:
                # Generate dense embedding for query
                dense_embedding = self.embedding_service.generate_dense_embedding(query_text)
                dense_results = self.pinecone_client.query_dense_similarity(
                    dense_vector=dense_embedding,
                    top_k=top_k,
                    namespace=namespace
                )
                results["dense"] = dense_results
            
            if use_sparse:
                # Generate sparse embedding for query
                sparse_embedding = self.embedding_service.generate_sparse_embedding(query_text)
                if sparse_embedding:  # Only search if we have sparse features
                    sparse_results = self.pinecone_client.query_sparse_similarity(
                        sparse_vector=sparse_embedding,
                        top_k=top_k,
                        namespace=namespace
                    )
                    results["sparse"] = sparse_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar emails: {str(e)}")
            raise