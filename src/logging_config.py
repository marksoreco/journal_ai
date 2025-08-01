import logging
import sys
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """
    Setup logging configuration for the Journal AI application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        log_format: Format string for log messages
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)

    # Get the OpenAI logger and set its level
    openai_logger = logging.getLogger("openai")
    openai_logger.setLevel(logging.WARNING) # Or logging.WARNING, logging.ERROR to suppress debug

    # You might also want to control the httpx logger, as it also logs request/response details
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING) # Or logging.WARNING, logging.ERROR
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {level}")
    if log_file:
        logger.info(f"Logs will also be written to: {log_file}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name) 