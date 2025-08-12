from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
import os
import time
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
import logging
from fastapi import APIRouter
from .gmail.auth import OAUTH_SCOPES

logger = logging.getLogger(__name__)

router = APIRouter()

# Load environment variables from .env file
load_dotenv()

OAUTH_REDIRECT_URI = os.getenv('GOOGLE_REDIRECT_URI')
GOOGLE_CREDENTIALS_PATH = os.getenv('GOOGLE_CREDENTIALS_PATH')

# Session timeout configuration (in seconds)
# Default: 24 hours, can be overridden with GOOGLE_SESSION_TIMEOUT env var
SESSION_TIMEOUT = int(os.getenv('GOOGLE_SESSION_TIMEOUT', 24 * 60 * 60))

# Log configuration for debugging
logger.info(f"OAuth Redirect URI: {OAUTH_REDIRECT_URI}")
logger.info(f"Google Credentials Path: {GOOGLE_CREDENTIALS_PATH}")

@router.get("/auth/google")
async def auth_google(t: str = None):
    try:
        logger.info(f"OAuth initiation requested. Timestamp: {t}")
        
        # Force clear any existing tokens to ensure fresh flow
        token_path = os.path.join(os.path.dirname(__file__), "gmail", "token.pkl")
        if os.path.exists(token_path):
            os.remove(token_path)
            logger.info("Removed existing token file for fresh OAuth flow")
        
        # Use credentials path from env or fallback to default
        credentials_file = GOOGLE_CREDENTIALS_PATH or os.path.join(os.path.dirname(__file__), 'gmail', 'credentials.json')
        flow = InstalledAppFlow.from_client_secrets_file(
            credentials_file,
            scopes=OAUTH_SCOPES,
            redirect_uri=OAUTH_REDIRECT_URI
        )
        auth_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'  # Force consent screen to appear
        )
        # Store state in session or DB for verification (simplified here)
        state_file = os.path.join(os.path.dirname(__file__), "gmail", "state.txt")
        with open(state_file, 'w') as f:
            f.write(state)
        logger.info(f"OAuth flow initiated. Redirecting to: {auth_url}")
        return RedirectResponse(url=auth_url)
    except Exception as e:
        logger.error(f"Error initiating OAuth: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start OAuth flow")

@router.get("/auth/google/callback")
async def auth_google_callback(code: str, state: str):
    try:
        # Verify state (prevent CSRF; simplified here)
        state_file = os.path.join(os.path.dirname(__file__), "gmail", "state.txt")
        with open(state_file, 'r') as f:
            saved_state = f.read()
        if state != saved_state:
            raise HTTPException(status_code=400, detail="Invalid state parameter")
        
        # Exchange code for tokens
        flow = InstalledAppFlow.from_client_secrets_file(
            GOOGLE_CREDENTIALS_PATH,
            scopes=OAUTH_SCOPES,
            redirect_uri=OAUTH_REDIRECT_URI
        )
        flow.fetch_token(code=code)
        creds = flow.credentials
        
        # Save credentials
        token_path = os.path.join(os.path.dirname(__file__), "gmail", "token.pkl")
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
        
        # Remove logout lock file if it exists
        lock_file = os.path.join(os.path.dirname(__file__), "gmail", "logout.lock")
        if os.path.exists(lock_file):
            os.remove(lock_file)
            logger.info("Removed logout lock file after successful login")
        
        # Clean up state
        state_file = os.path.join(os.path.dirname(__file__), "gmail", "state.txt")
        if os.path.exists(state_file):
            os.remove(state_file)
        
        # Redirect to the main UI after successful authentication
        return RedirectResponse(url='/ui')
    except Exception as e:
        logger.error(f"Error in OAuth callback: {str(e)}")
        raise HTTPException(status_code=500, detail="OAuth callback failed")

@router.post("/auth/logout")
async def logout():
    """Logout user by clearing stored credentials"""
    try:
        token_path = os.path.join(os.path.dirname(__file__), "gmail", "token.pkl")
        logger.info(f"Logout requested. Looking for token file at: {token_path}")
        
        # Remove token.pkl file
        if os.path.exists(token_path):
            try:
                os.remove(token_path)
                logger.info(f"Token file removed successfully: {token_path}")
                
                # Verify removal
                if os.path.exists(token_path):
                    logger.error(f"Token file still exists after removal: {token_path}")
                    # Try force removal
                    try:
                        import stat
                        os.chmod(token_path, stat.S_IWRITE)
                        os.remove(token_path)
                        logger.info(f"Token file force removed: {token_path}")
                    except Exception as force_e:
                        logger.error(f"Force removal failed: {force_e}")
                else:
                    logger.info("Token file successfully verified as removed")
            except Exception as e:
                logger.error(f"Error removing token file: {e}")
        else:
            logger.info("Token file not found")
        
        # Create a lock file to prevent token recreation during logout
        lock_file = os.path.join(os.path.dirname(__file__), "gmail", "logout.lock")
        try:
            with open(lock_file, 'w') as f:
                f.write(str(time.time()))
            logger.info(f"Created logout lock file: {lock_file}")
            
            # Remove lock file after 5 minutes
            def remove_lock_file():
                import time
                time.sleep(300)  # 5 minutes
                if os.path.exists(lock_file):
                    try:
                        os.remove(lock_file)
                        logger.info("Removed logout lock file after timeout")
                    except:
                        pass
            
            import threading
            timer = threading.Timer(300, remove_lock_file)
            timer.daemon = True
            timer.start()
        except Exception as e:
            logger.error(f"Error creating lock file: {e}")
        
        # Remove state.txt file
        state_file = os.path.join(os.path.dirname(__file__), "gmail", "state.txt")
        if os.path.exists(state_file):
            try:
                os.remove(state_file)
                logger.info(f"State file removed successfully: {state_file}")
            except Exception as e:
                logger.error(f"Error removing state file: {e}")
        else:
            logger.info("State file not found")
        
        return JSONResponse({"status": "Logout successful", "message": "Credentials cleared"})
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        # Even if there's an error, return success to allow user to proceed
        return JSONResponse({"status": "Logout successful", "message": "Logout completed"})

@router.get("/auth/status")
async def auth_status():
    """Check if Gmail authentication is valid without requiring authentication"""
    try:
        token_path = os.path.join(os.path.dirname(__file__), "gmail", "token.pkl")
        logger.info(f"Auth status check requested. Looking for token at: {token_path}")
        
        # Check if token file exists and is valid
        if os.path.exists(token_path):
            logger.info(f"Token file found at {token_path}")
            
            # Check if token file is recent (within configured timeout)
            token_age = time.time() - os.path.getmtime(token_path)
            
            if token_age > SESSION_TIMEOUT:
                logger.info(f"Token file is too old ({token_age:.0f} seconds, max: {SESSION_TIMEOUT}), removing it")
                os.remove(token_path)
                return {"authenticated": False, "message": f"Session expired - token file too old ({token_age:.0f}s > {SESSION_TIMEOUT}s)"}
            
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
            
            logger.info(f"Credentials loaded: valid={getattr(creds, 'valid', 'N/A')}, expired={getattr(creds, 'expired', 'N/A')}")
            
            # Check if credentials are valid
            if creds and creds.valid:
                logger.info("Credentials are valid")
                return {"authenticated": True, "message": "Valid credentials found"}
            elif creds and creds.expired and creds.refresh_token:
                # Don't refresh expired credentials during status check - just return not authenticated
                logger.info("Credentials expired, not refreshing during status check")
                # Remove expired token file
                os.remove(token_path)
                return {"authenticated": False, "message": "Credentials expired"}
            else:
                logger.info("Credentials are invalid")
                # Remove invalid token file
                os.remove(token_path)
                return {"authenticated": False, "message": "Invalid credentials"}
        else:
            logger.info(f"Token file not found at {token_path}")
            return {"authenticated": False, "message": "No credentials found"}
    except Exception as e:
        logger.error(f"Error checking auth status: {str(e)}")
        # Remove token file if there's an error reading it
        if os.path.exists(token_path):
            try:
                os.remove(token_path)
                logger.info("Removed corrupted token file")
            except:
                pass
        return {"authenticated": False, "message": f"Error: {str(e)}"}