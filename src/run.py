#!/usr/bin/env python3
"""
Entry point for the Journal AI FastAPI application.
"""

import uvicorn
from .logging_config import setup_logging
from .main import app

# Setup logging before starting the application
from .config import LOG_LEVEL, LOG_FILE
setup_logging(level=LOG_LEVEL, log_file=LOG_FILE)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 