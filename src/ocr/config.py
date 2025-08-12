# OCR Engine Configuration
# Available options:
# - "GPT4oOCRAdapter": Uses OpenAI's GPT-4o for OCR with confidence scoring
# - "PaddleOCRAdapter": Uses PaddleOCR (if available) - removed from current implementation
# - Future engines can be added here (e.g., "TesseractOCRAdapter", "GoogleVisionOCRAdapter")
OCR_ENGINE = "GPT4oOCRAdapter"