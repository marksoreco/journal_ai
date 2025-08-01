# OCR Engine Configuration
# Available options:
# - "GPT4oOCRAdapter": Uses OpenAI's GPT-4o for OCR with confidence scoring
# - "PaddleOCRAdapter": Uses PaddleOCR (if available) - removed from current implementation
# - Future engines can be added here (e.g., "TesseractOCRAdapter", "GoogleVisionOCRAdapter")
OCR_ENGINE = "GPT4oOCRAdapter"

# Task Confidence Threshold Configuration
# Range: 0.0 to 1.0
# - Lower values (0.5-0.7): More strict - review more tasks
# - Higher values (0.8-0.9): Less strict - review fewer tasks
# - Default: 0.9 (90% confidence threshold)
TASK_CONFIDENCE_THRESHOLD = 0.9

# SBERT Configuration for Intelligent Duplicate Detection
# - SBERT_ENABLED: Whether to use SBERT for duplicate detection
# - SBERT_MODEL: SBERT model to use (default: all-MiniLM-L6-v2)
# - SBERT_SIMILARITY_THRESHOLD: Threshold for considering tasks as duplicates (0.0-1.0)
# - SBERT_CACHE_FILE: File path for persistent embedding cache
SBERT_ENABLED = True
SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_SIMILARITY_THRESHOLD = 0.8
SBERT_CACHE_FILE = "embeddings_cache.pkl"

# Logging Configuration
# - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# - LOG_FILE: Optional file path to write logs to (None for console only)
LOG_LEVEL = "DEBUG"
LOG_FILE = None