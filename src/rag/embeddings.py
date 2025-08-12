"""Embedding service for generating dense and sparse embeddings from email text."""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating dense and sparse embeddings."""
    
    def __init__(self, 
                 dense_model_name: str = "all-MiniLM-L6-v2",
                 sparse_max_features: int = 10000):
        """
        Initialize embedding service.
        
        Args:
            dense_model_name: Name of sentence transformer model
            sparse_max_features: Maximum features for TF-IDF vectorizer
        """
        self.dense_model_name = dense_model_name
        self.sparse_max_features = sparse_max_features
        
        # Initialize dense model
        self.dense_model = None
        
        # Initialize sparse vectorizer
        self.sparse_vectorizer = None
        self.vectorizer_fitted = False
        
        # Cache directory for models
        self.cache_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized embedding service - Dense: {dense_model_name}, Sparse features: {sparse_max_features}")
    
    def _get_dense_model(self):
        """Lazy load dense model."""
        if self.dense_model is None:
            logger.info(f"Loading dense model: {self.dense_model_name}")
            self.dense_model = SentenceTransformer(self.dense_model_name)
        return self.dense_model
    
    def _get_sparse_vectorizer(self):
        """Get or initialize sparse vectorizer."""
        if self.sparse_vectorizer is None:
            self.sparse_vectorizer = TfidfVectorizer(
                max_features=self.sparse_max_features,
                stop_words='english',
                ngram_range=(1, 2),  # Unigrams and bigrams
                lowercase=True,
                token_pattern=r'[a-zA-Z][a-zA-Z0-9]*',  # Alphanumeric tokens
                min_df=1,  # Minimum document frequency
                max_df=0.95  # Maximum document frequency
            )
        return self.sparse_vectorizer
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs (already handled in Gmail client, but just in case)
        text = re.sub(r'http[s]?://\S+', '', text)
        
        return text.strip()
    
    def generate_dense_embedding(self, text: str) -> List[float]:
        """
        Generate dense embedding for text using sentence transformer.
        
        Args:
            text: Input text
            
        Returns:
            List of float values representing dense embedding
        """
        try:
            processed_text = self._preprocess_text(text)
            if not processed_text:
                # Return zero vector if no text
                model = self._get_dense_model()
                return [0.0] * model.get_sentence_embedding_dimension()
            
            model = self._get_dense_model()
            embedding = model.encode(processed_text, normalize_embeddings=True)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating dense embedding: {str(e)}")
            raise
    
    def fit_sparse_vectorizer(self, texts: List[str]) -> None:
        """
        Fit the sparse vectorizer on a corpus of texts.
        
        Args:
            texts: List of texts to fit vectorizer
        """
        try:
            logger.info(f"Fitting sparse vectorizer on {len(texts)} texts")
            
            processed_texts = [self._preprocess_text(text) for text in texts]
            processed_texts = [text for text in processed_texts if text]  # Remove empty strings
            
            if not processed_texts:
                logger.warning("No valid texts to fit vectorizer")
                return
            
            vectorizer = self._get_sparse_vectorizer()
            vectorizer.fit(processed_texts)
            self.vectorizer_fitted = True
            
            # Save fitted vectorizer
            vectorizer_path = os.path.join(self.cache_dir, "tfidf_vectorizer.pkl")
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            
            logger.info(f"Sparse vectorizer fitted with {len(vectorizer.vocabulary_)} features")
            
        except Exception as e:
            logger.error(f"Error fitting sparse vectorizer: {str(e)}")
            raise
    
    def load_sparse_vectorizer(self) -> bool:
        """
        Load fitted sparse vectorizer from cache.
        
        Returns:
            bool: True if successfully loaded
        """
        try:
            vectorizer_path = os.path.join(self.cache_dir, "tfidf_vectorizer.pkl")
            
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.sparse_vectorizer = pickle.load(f)
                self.vectorizer_fitted = True
                logger.info(f"Loaded sparse vectorizer with {len(self.sparse_vectorizer.vocabulary_)} features")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading sparse vectorizer: {str(e)}")
            return False
    
    def generate_sparse_embedding(self, text: str) -> Dict[str, float]:
        """
        Generate sparse embedding for text using TF-IDF.
        
        Args:
            text: Input text
            
        Returns:
            Dict mapping feature indices to values
        """
        try:
            processed_text = self._preprocess_text(text)
            if not processed_text:
                return {}
            
            vectorizer = self._get_sparse_vectorizer()
            
            if not self.vectorizer_fitted:
                # Try to load from cache first
                if not self.load_sparse_vectorizer():
                    logger.warning("Sparse vectorizer not fitted. Call fit_sparse_vectorizer first.")
                    return {}
            
            # Transform text to sparse vector
            sparse_vector = vectorizer.transform([processed_text])
            
            # Convert to dictionary format
            sparse_dict = {}
            if sparse_vector.nnz > 0:  # Check if there are non-zero elements
                coo = sparse_vector.tocoo()
                for idx, value in zip(coo.col, coo.data):
                    if value > 0:  # Only include positive values
                        sparse_dict[str(idx)] = float(value)
            
            return sparse_dict
            
        except Exception as e:
            logger.error(f"Error generating sparse embedding: {str(e)}")
            raise
    
    def generate_embeddings(self, text: str) -> Tuple[List[float], Dict[str, float]]:
        """
        Generate both dense and sparse embeddings for text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (dense_embedding, sparse_embedding)
        """
        dense_embedding = self.generate_dense_embedding(text)
        sparse_embedding = self.generate_sparse_embedding(text)
        
        return dense_embedding, sparse_embedding
    
    def batch_generate_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts
            
        Returns:
            Tuple of (list_of_dense_embeddings, list_of_sparse_embeddings)
        """
        try:
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate dense embeddings in batch
            model = self._get_dense_model()
            dense_embeddings = model.encode(processed_texts, normalize_embeddings=True, show_progress_bar=True)
            
            # Generate sparse embeddings
            sparse_embeddings = []
            for text in processed_texts:
                sparse_emb = self.generate_sparse_embedding(text)
                sparse_embeddings.append(sparse_emb)
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return dense_embeddings.tolist(), sparse_embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {str(e)}")
            raise