"""Gmail authentication service."""

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import pickle
import os
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

OAUTH_SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]

def _compute_redirect_uri(request: Request, path="/auth/google/callback") -> str:
    if (fixed := os.getenv("GOOGLE_REDIRECT_URI")):  # optional manual override
        return fixed
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host  = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}{path}"

def get_gmail_service():
    """Get or refresh Gmail service with authentication."""
    creds = None
    try:
        # Get token file path in gmail directory
        token_path = os.path.join(os.path.dirname(__file__), "token.pkl")
        
        # Load existing credentials if available
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # Check for logout lock
        lock_file = os.path.join(os.path.dirname(__file__), "logout.lock")
        if os.path.exists(lock_file):
            logger.info("Logout lock detected, not refreshing credentials")
            raise HTTPException(status_code=401, detail="Authentication required. Visit /auth/google")
        
        # Refresh or validate credentials
        if creds and creds.valid:
            return build('gmail', 'v1', credentials=creds)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
            return build('gmail', 'v1', credentials=creds)
        
        # No valid creds; need to initiate OAuth flow
        raise HTTPException(status_code=401, detail="Authentication required. Visit /auth/google")
    except Exception as e:
        logger.error(f"Error in get_gmail_service: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize Gmail service")