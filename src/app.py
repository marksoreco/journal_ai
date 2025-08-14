import json
import logging
import atexit
from fastapi import FastAPI, Depends
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
from fastapi import UploadFile, File, HTTPException, Form
from typing import List
import os
from dotenv import load_dotenv
from .todoist.todoist_client import TodoistClient
from .ocr.ocr_factory import OCRFactory
from .agents.tools.page_detector import PageTypeDetector
from .auth_routes import router as auth_router
from .gmail.auth import get_gmail_service
from .gmail.client import GmailClient
from datetime import datetime
from pydantic import BaseModel
from .rag.email_vectorizer import EmailVectorizer

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
app.include_router(auth_router)

# Pydantic models
class GmailDataRequest(BaseModel):
    since_date: str
    limit: int

@app.get("/", response_class=JSONResponse)
def system_status():
    return {"status": "ok", "message": "Journal AI is running"}

@app.get("/config", response_class=JSONResponse)
def get_config():
    """Get application configuration for the frontend"""
    from .todoist.config import TASK_CONFIDENCE_THRESHOLD
    from .config import LOG_LEVEL
    return {
        "task_confidence_threshold": TASK_CONFIDENCE_THRESHOLD,
        "log_level": LOG_LEVEL
    }

@app.get("/ui", response_class=FileResponse)
async def serve_index():
    """Serve the main UI - authentication checked by frontend"""
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html"))

@app.post("/upload-to-todoist")
async def upload_to_todoist(
    task_data: dict,
    gmail_service = Depends(get_gmail_service)
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

@app.post("/detect-page-type")
async def detect_page_type(
    file: UploadFile = File(...),
    gmail_service = Depends(get_gmail_service)
):
    logger.info(f"Received page type detection request: {file.filename}")
    
    allowed_types = [
        "image/jpeg",
        "image/png"
    ]
    if file.content_type not in allowed_types:
        logger.warning(f"Invalid image type received: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid image type. Allowed types: jpg, jpeg, and png.")

    # Save the uploaded file temporarily
    base_dir = os.path.abspath(os.path.dirname(__file__))
    upload_dir = os.path.join(os.path.dirname(base_dir), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filename = file.filename or "uploaded_image"
    file_path = os.path.join(upload_dir, filename)
    
    logger.info(f"Saving uploaded file for detection to: {file_path}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Detect page type using PageTypeDetector
    try:
        logger.info(f"Starting page type detection for: {filename}")
        detector = PageTypeDetector()
        detection_result = detector.detect_page_type(file_path)
        
        logger.info(f"Page type detection completed: {detection_result.page_type.value}")
        
        return {
            "page_type": detection_result.page_type.value,
            "reasoning": detection_result.reasoning,
            "visual_indicators": detection_result.visual_indicators,
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"Error during page type detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting page type: {str(e)}")

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    category: str = Form(...),
    gmail_service = Depends(get_gmail_service)
):
    logger.info(f"Received image upload request: {file.filename}, category: {category}")
    
    allowed_types = [
        "image/jpeg",
        "image/png"
    ]
    if file.content_type not in allowed_types:
        logger.warning(f"Invalid image type received: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid image type. Allowed types: jpg, jpeg, and png.")
    
    allowed_categories = {"Daily", "Weekly", "Monthly"}
    if category not in allowed_categories:
        logger.warning(f"Invalid category received: {category}")
        raise HTTPException(status_code=400, detail="Invalid category. Must be Daily, Weekly, or Monthly.")

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
    logger.info(f"Starting OCR processing with category: {category}")
    ocr_engine = ocr_factory.get_current_engine()
    ocr_text = ocr_engine.extract_text(file_path, category)
    logger.info(f"OCR processing completed successfully for category: {category}")
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "category": category,
        "ocr_text": ocr_text,
        "message": "Image uploaded and processed successfully."
    }

@app.post("/gmail/fetch-data")
async def fetch_gmail_data(
    request: GmailDataRequest,
    gmail_service = Depends(get_gmail_service)
):
    """
    Fetch Gmail data since specified date and save to JSON file
    """
    logger.info(f"Received Gmail data request: since_date={request.since_date}, limit={request.limit}")
    
    try:
        # Parse the date
        since_date = datetime.strptime(request.since_date, "%Y-%m-%d")
        
        # Initialize Gmail client
        gmail_client = GmailClient()
        
        # Fetch emails
        logger.info(f"Fetching emails since {since_date} with limit {request.limit}")
        emails = gmail_client.get_emails_since_date(since_date, request.limit)
        
        # Convert EmailMessage objects to dict for JSON serialization
        emails_data = []
        for email in emails:
            emails_data.append({
                "id": email.id,
                "thread_id": email.thread_id,
                "subject": email.subject,
                "sender": email.sender,
                "recipient": email.recipient,
                "date": email.date.isoformat(),
                "body": email.body,
                "snippet": email.snippet,
                "labels": email.labels
            })
        
        # Create filename with date format
        filename = f"gmail_since_{request.since_date}.json"
        
        # Save to gmail downloads directory
        downloads_dir = os.path.join(os.path.dirname(__file__), "gmail", "downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        filepath = os.path.join(downloads_dir, filename)
        
        # Write JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "fetch_date": datetime.now().isoformat(),
                "since_date": request.since_date,
                "limit": request.limit,
                "total_emails": len(emails_data),
                "emails": emails_data
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved {len(emails_data)} emails to {filepath}")
        
        # Upload to Pinecone vector database
        vectorizer_results = {}
        try:
            logger.info("Starting Pinecone vectorization and upload")
            vectorizer = EmailVectorizer()
            vectorizer_results = vectorizer.process_and_store_emails(emails)
            logger.info(f"Pinecone upload results: {vectorizer_results}")
        except Exception as vec_error:
            logger.error(f"Error uploading to Pinecone: {str(vec_error)}")
            # Continue without failing the entire request
            vectorizer_results = {"error": str(vec_error)}
        
        return {
            "message": f"Successfully fetched {len(emails_data)} emails and saved to {filename}",
            "filename": filename,
            "total_emails": len(emails_data),
            "since_date": request.since_date,
            "limit": request.limit,
            "pinecone_upload": vectorizer_results
        }
        
    except ValueError as e:
        logger.error(f"Invalid date format: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    except Exception as e:
        logger.error(f"Error fetching Gmail data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching Gmail data: {str(e)}")