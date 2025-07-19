#!/usr/bin/env python3
"""
Entry point for the Journal AI FastAPI application.
"""

import uvicorn
from .main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 