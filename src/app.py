import logging
import json
from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException, Request
import os
import pathlib
import base64
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware

from .auth_routes import router as auth_router
from .chat.routes import router as chat_router
from .gmail.auth import get_gmail_service
from .gmail.client import GmailClient
from datetime import datetime
from pydantic import BaseModel
from .rag.email_vectorizer import EmailVectorizer
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

# Configure logger for this module
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Setup logging
from .logging_config import setup_logging
from .config import LOG_LEVEL, LOG_FILE
setup_logging(level=LOG_LEVEL, log_file=LOG_FILE)

# Set up Google credentials handling for cloud deployment
def _write_google_credentials_from_env():
    creds = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if not creds:
        return  # local dev can still use a file at GOOGLE_CREDENTIALS_PATH

    # ✅ write to /tmp by default on Cloud Run
    target = pathlib.Path(os.getenv("GOOGLE_CREDENTIALS_PATH", "/tmp/google_client.json"))
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        text = creds.strip()
        if not text:
            return
        if not text.lstrip().startswith("{"):
            text = base64.b64decode(text).decode("utf-8")
        json.loads(text)  # validate
        target.write_text(text)
        logging.getLogger(__name__).info("Wrote Google OAuth credentials to %s", target)
    except Exception as e:
        logging.getLogger(__name__).exception("Failed to materialize GOOGLE_CREDENTIALS_JSON: %s", e)

_write_google_credentials_from_env()

app = FastAPI()

# Add session middleware for OAuth state management
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY"),
    session_cookie="journal_ai_session",
    max_age=3600,  # 1 hour
    same_site="lax"
)

# Add proxy headers middleware to trust Cloud Run headers
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Mount static files (only chat-related)
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(auth_router)
app.include_router(chat_router)

# Gmail data request model
class GmailDataRequest(BaseModel):
    since_date: str
    limit: int

@app.get("/", response_class=JSONResponse)
def system_status():
    return {"status": "ok", "message": "Journal AI is running"}

# Classic UI routes removed

@app.get("/chat", response_class=FileResponse)
async def serve_chat():
    """Serve the chat UI - authentication checked by frontend"""
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "chat.html"))

# Classic UI endpoints removed - all processing now handled by chat interface

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