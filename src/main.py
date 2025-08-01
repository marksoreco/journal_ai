import json
import logging
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
from fastapi import UploadFile, File, HTTPException, Form
from typing import List
import os
from .config import OCR_ENGINE
from .ocr.base import BaseOCR
import importlib
from dotenv import load_dotenv
from .todoist_client import TodoistClient
from .ocr_factory import OCRFactory

# Configure logger for this module
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Setup logging
from .logging_config import setup_logging
from .config import LOG_LEVEL, LOG_FILE
setup_logging(level=LOG_LEVEL, log_file=LOG_FILE)

# Initialize OCR factory
ocr_factory = OCRFactory()

app = FastAPI()

@app.get("/", response_class=JSONResponse)
def system_status():
    return {"status": "ok", "message": "Journal AI is running"}

@app.get("/config", response_class=JSONResponse)
def get_config():
    """Get application configuration for the frontend"""
    from .config import TASK_CONFIDENCE_THRESHOLD
    return {
        "task_confidence_threshold": TASK_CONFIDENCE_THRESHOLD
    }

@app.get("/ui", response_class=FileResponse)
def serve_index():
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html"))

@app.post("/upload-to-todoist")
async def upload_to_todoist(
    task_data: dict
):
    """
    Upload tasks to Todoist from extracted OCR data
    """
    logger.info("Received request to upload tasks to Todoist")
    try:
        todoist_client = TodoistClient()
        result = todoist_client.upload_tasks_from_ocr(task_data)
        logger.info(f"Successfully uploaded tasks to Todoist: {result.get('message', 'Unknown result')}")
        return result
        
    except Exception as e:
        logger.error(f"Error uploading to Todoist: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading to Todoist: {str(e)}")

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    category: str = Form(...)
):
    logger.info(f"Received image upload request: {file.filename}, category: {category}")
    
    allowed_types = [
        "image/jpeg",
        "image/png"
    ]
    if file.content_type not in allowed_types:
        logger.warning(f"Invalid image type received: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid image type. Allowed types: jpg, jpeg, and png.")
    
    allowed_categories = {"Day", "Week", "Month"}
    if category not in allowed_categories:
        logger.warning(f"Invalid category received: {category}")
        raise HTTPException(status_code=400, detail="Invalid category. Must be Day, Week, or Month.")

    # Save the uploaded file to disk
    base_dir = os.path.abspath(os.path.dirname(__file__))
    upload_dir = os.path.join(os.path.dirname(base_dir), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filename = file.filename or "uploaded_image"
    file_path = os.path.join(upload_dir, filename)
    
    logger.info(f"Saving uploaded file to: {file_path}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # OCR processing
    logger.info("Starting OCR processing")
    ocr_engine = ocr_factory.get_current_engine()
    ocr_text = ocr_engine.extract_text(file_path)
    logger.info("OCR processing completed successfully")
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "category": category,
        "ocr_text": ocr_text,
        "message": "Image uploaded and processed successfully."
    }