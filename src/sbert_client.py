import pickle
import os
import logging
import numpy as np
from hashlib import md5
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Configure logger for this module
logger = logging.getLogger(__name__)

class SBERTClient:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_file: str = "embeddings_cache.pkl", similarity_threshold: float = 0.85):
        """
        Initialize SBERT client for intelligent duplicate detection
        
        Args:
            model_name: SBERT model to use (default: all-MiniLM-L6-v2)
            cache_file: File path for persistent embedding cache
            similarity_threshold: Threshold for considering tasks as duplicates (0.0-1.0)
        """
        self.model_name = model_name
        self.cache_file = cache_file
        self.similarity_threshold = similarity_threshold
        
        # Load the SBERT model
        self.model = SentenceTransformer(model_name)
        
        # Load or initialize embedding cache
        self.embedding_cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, np.ndarray]:
        """Load embedding cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache file: {str(e)}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save embedding cache to file"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache file: {str(e)}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key"""
        return md5(text.encode('utf-8')).hexdigest()
    
    def _ensure_numpy_array(self, embedding: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array if needed"""
        if isinstance(embedding, torch.Tensor):
            return embedding.cpu().numpy()
        return embedding
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available"""
        text_hash = self._get_text_hash(text)
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Compute embedding and cache it
        embedding = self.model.encode(text)
        embedding_np = self._ensure_numpy_array(embedding)
        self.embedding_cache[text_hash] = embedding_np
        self._save_cache()
        return embedding_np
    
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts, using cache efficiently"""
        # Separate cached and uncached texts
        cached_embeddings = {}
        uncached_texts = []
        uncached_hashes = []
        
        for text in texts:
            text_hash = self._get_text_hash(text)
            if text_hash in self.embedding_cache:
                cached_embeddings[text] = self.embedding_cache[text_hash]
            else:
                uncached_texts.append(text)
                uncached_hashes.append(text_hash)
        
        # Batch encode only uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts)
            
            # Cache the new embeddings
            for text, embedding, text_hash in zip(uncached_texts, new_embeddings, uncached_hashes):
                embedding_np = self._ensure_numpy_array(embedding)
                self.embedding_cache[text_hash] = embedding_np
                cached_embeddings[text] = embedding_np
            
            # Save cache after batch update
            self._save_cache()
        
        # Return embeddings in the same order as input texts
        return [cached_embeddings[text] for text in texts]
    
    def check_duplicate_tasks(self, new_tasks: List[str], existing_tasks: List[str]) -> Dict[str, bool]:
        """
        Check if any new tasks are duplicates of existing tasks using SBERT embeddings
        
        Args:
            new_tasks: List of new task strings to check
            existing_tasks: List of existing task strings to compare against
            
        Returns:
            Dict mapping new task to boolean (True if duplicate, False if unique)
        """
        logger.debug(f"Checking {len(new_tasks)} new tasks against {len(existing_tasks)} existing tasks")
        
        if not existing_tasks:
            # No existing tasks, so all new tasks are unique
            logger.debug("No existing tasks found, all new tasks are unique")
            return {task: False for task in new_tasks}
            
        if not new_tasks:
            # No new tasks to check
            logger.debug("No new tasks to check")
            return {}
        
        try:
            # Get embeddings for all tasks efficiently
            logger.debug("Computing embeddings for all tasks")
            all_tasks = existing_tasks + new_tasks
            all_embeddings = self.get_embeddings_batch(all_tasks)
            
            # Split embeddings
            existing_embeddings = all_embeddings[:len(existing_tasks)]
            new_embeddings = all_embeddings[len(existing_tasks):]
            
            # Convert to numpy arrays for similarity calculation
            existing_embeddings_array = np.array(existing_embeddings)
            new_embeddings_array = np.array(new_embeddings)
            
            # Calculate cosine similarity between new and existing tasks
            logger.debug("Computing cosine similarity matrix")
            similarity_matrix = cosine_similarity(new_embeddings_array, existing_embeddings_array)
            

            
            # Determine duplicates based on similarity threshold
            duplicate_results = {}
            duplicates_found = 0
            for i, new_task in enumerate(new_tasks):
                # Find maximum similarity with any existing task
                max_similarity = np.max(similarity_matrix[i])
                is_duplicate = max_similarity >= self.similarity_threshold
                duplicate_results[new_task] = is_duplicate
                
                if is_duplicate:
                    duplicates_found += 1
                    # Find which existing task is most similar
                    most_similar_idx = np.argmax(similarity_matrix[i])
                    most_similar_task = existing_tasks[most_similar_idx]
            
            # Log summary table of results
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("=== DUPLICATE DETECTION SUMMARY ===")
                logger.debug("New Task".ljust(40) + " | Max Similarity | Most Similar Existing Task".ljust(40) + " | Status")
                logger.debug("-" * 100)
                
                for i, new_task in enumerate(new_tasks):
                    max_similarity = np.max(similarity_matrix[i])
                    most_similar_idx = np.argmax(similarity_matrix[i])
                    most_similar_task = existing_tasks[most_similar_idx]
                    is_duplicate = max_similarity >= self.similarity_threshold
                    
                    # Truncate task names for readability
                    truncated_new = new_task[:40] + "..." if len(new_task) > 40 else new_task
                    truncated_existing = most_similar_task[:40] + "..." if len(most_similar_task) > 40 else most_similar_task
                    
                    status = "DUPLICATE" if is_duplicate else "UNIQUE"
                    similarity_str = f"{max_similarity:.3f}"
                    
                    row = truncated_new.ljust(40) + " | " + similarity_str.ljust(14) + " | " + truncated_existing.ljust(40) + " | " + status
                    logger.debug(row)
                
                logger.debug("-" * 100)
                logger.debug("=== END DUPLICATE DETECTION SUMMARY ===")
            
            logger.info(f"Duplicate detection completed: {duplicates_found} duplicates found out of {len(new_tasks)} tasks")
            return duplicate_results
            
        except Exception as e:
            logger.error(f"Error checking duplicates with SBERT: {str(e)}")
            # Fallback to simple text comparison
            return self._fallback_duplicate_check(new_tasks, existing_tasks)
    
    def _fallback_duplicate_check(self, new_tasks: List[str], existing_tasks: List[str]) -> Dict[str, bool]:
        """Fallback to simple text comparison if SBERT fails"""
        duplicate_results = {}
        
        for new_task in new_tasks:
            is_duplicate = False
            
            for existing_task in existing_tasks:
                # Simple case-insensitive comparison
                if new_task.lower().strip() == existing_task.lower().strip():
                    is_duplicate = True
                    break
            
            duplicate_results[new_task] = is_duplicate
        
        return duplicate_results
    
    def cleanup_cache(self, max_age_days: int = 30):
        """Remove old cache entries (placeholder for future implementation)"""
        # TODO: Implement cache cleanup based on age
        pass 