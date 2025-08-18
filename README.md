# Journal AI - Conversational Journal Processing System

A comprehensive FastAPI application for intelligent journal processing through a conversational AI interface. Features specialized OCR capabilities for Daily, Weekly, and Monthly journal pages, task management integration with Todoist, low-confidence item review system, and Gmail-based RAG (Retrieval-Augmented Generation) system.

## Features

- **Conversational AI Interface**: Modern chat-based interface for all journal processing workflows
- **Intelligent Journal OCR Processing**: Specialized OCR adapters for Day/Week/Month journal pages using GPT-4o vision
- **Low-Confidence Item Review**: Interactive prefill editing system for uncertain OCR results before Todoist upload
- **Streaming Function Execution**: Real-time progress updates during OCR processing and task operations
- **Vector Database Storage**: Store journal OCR data in Pinecone with deterministic IDs for searchable history
- **Todoist Integration**: Automatically create tasks from extracted text with intelligent duplicate detection  
- **Gmail RAG System**: Process emails, generate embeddings, and store in Pinecone vector database
- **Session-Based OAuth Authentication**: Secure Gmail API integration with persistent session management
- **SBERT Intelligence**: Advanced semantic similarity for duplicate detection using sentence transformers
- **Session Management**: Persistent chat sessions with processing state tracking
- **Cloud Run Deployment**: Production-ready deployment with proper proxy header handling

## Setup

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Create a `.env` file in the project root:**

```bash
# Required API Keys
OPENAI_API_KEY=your-openai-api-key-here
TODOIST_API_TOKEN=your-todoist-api-token-here
PINECONE_API_KEY=your-pinecone-api-key-here

# Session Management (Required for OAuth)
SESSION_SECRET_KEY=your-32-character-random-secret-key

# Pinecone Configuration (optional - defaults will be used)
PINECONE_DENSE_INDEX=email-dense-index
PINECONE_SPARSE_INDEX=email-sparse-index

# Gmail OAuth Configuration (optional - auto-detected in production)
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
GOOGLE_CREDENTIALS_PATH=src/gmail/google_client.json

# Logging Configuration
LOG_LEVEL=DEBUG
LOG_FILE=logs/app.log

# Optional: Milvus Configuration (for future use)
MILVUS_USERNAME=your-username
MILVUS_PASSWORD=your-password
MILVUS_URI=your-milvus-uri
```

3. **Set up Gmail API credentials:**
   - Create a Google Cloud project and enable Gmail API
   - Download credentials and place as `src/gmail/google_client.json`
   - For Cloud Run deployment, set `GOOGLE_CREDENTIALS_JSON` environment variable with base64-encoded credentials

4. **Generate Session Secret Key:**
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

5. **Run the application:**

```bash
# Using the run.py entry point
python -m src.run

# Or using uvicorn directly
uvicorn src.app:app --reload
```

6. Visit [http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat) to access the conversational interface.

## Project Structure

```
journal_ai/
├── src/
│   ├── app.py              # Main FastAPI application with session middleware
│   ├── config.py           # Core logging configuration
│   ├── run.py              # Application entry point
│   ├── auth_routes.py      # Session-based OAuth authentication routes
│   ├── mcp_server.py       # MCP (Model Context Protocol) server
│   ├── chat/               # Conversational AI system
│   │   ├── routes.py       # Chat API endpoints
│   │   ├── chat_service.py # OpenAI integration & function calling
│   │   ├── function_tools.py # Chat function implementations (journal OCR, Pinecone storage, Todoist, Gmail)
│   │   ├── models.py       # Chat data models
│   │   ├── session_manager.py # Session state management
│   │   └── ocr_formatter.py   # OCR results formatting
│   ├── agents/             # AI agents and tools
│   │   ├── journal_processing_agent.py # LangChain agent workflow
│   │   └── tools/
│   │       └── page_detector.py # Page type detection
│   ├── gmail/              # Gmail API integration
│   │   ├── auth.py         # Gmail authentication with updated OAuth scopes
│   │   ├── client.py       # Gmail API client
│   │   ├── google_client.json # OAuth credentials (not in git)
│   │   ├── downloads/      # Downloaded email data
│   │   └── token.pkl       # OAuth token (not in git)
│   ├── ocr/                # Intelligent OCR processing system
│   │   ├── base.py         # Base OCR interface
│   │   ├── config.py       # OCR configuration
│   │   ├── gpt4o_ocr.py    # OCR dispatcher (routes to specialized adapters)
│   │   ├── daily_ocr.py    # Daily page OCR adapter
│   │   ├── weekly_ocr.py   # Weekly page OCR adapter  
│   │   ├── monthly_ocr.py  # Monthly page OCR adapter with date parsing
│   │   └── ocr_factory.py  # OCR engine factory
│   ├── rag/                # RAG system components
│   │   ├── embeddings.py   # Embedding service
│   │   ├── email_vectorizer.py # Email processing pipeline
│   │   ├── pinecone_client.py  # Pinecone vector database
│   │   └── models/         # Cached TF-IDF models
│   └── todoist/            # Todoist integration
│       ├── config.py       # Todoist-specific configuration
│       ├── sbert_client.py # SBERT embedding client
│       ├── todoist_client.py # Todoist API client
│       └── cache/          # SBERT embedding cache
├── static/
│   ├── chat.html          # Conversational AI interface
│   └── chat.css           # Chat interface styling
├── test/                   # Comprehensive test suite
│   ├── test_basic.py       # Basic functionality tests
│   ├── test_duplicate_detection.py # SBERT and Todoist integration tests
│   ├── test_ocr_integration.py # OCR adapter integration tests
│   ├── test_monthly_date_parsing.py # Date parsing validation tests
│   ├── run_tests.py        # Test runner
│   └── images/             # Test image files
├── uploads/                # Uploaded images
│   └── chat/               # Chat-uploaded files
├── logs/                   # Application logs (created automatically)
└── requirements.txt        # Python dependencies
```

## Core Features

### 1. Conversational AI Interface
- **Modern Chat Interface**: Clean, responsive chat UI for all interactions
- **File Upload Integration**: Drag & drop or browse for journal page images
- **Real-time Streaming**: Live progress updates during processing operations
- **Session Persistence**: Maintains conversation history and processing state
- **Authentication Integration**: Seamless OAuth login/logout experience

### 2. Session-Based OAuth Authentication
- **Secure Session Management**: Encrypted session cookies with configurable timeouts
- **Cloud Run Compatible**: Works across ephemeral instances without file persistence issues
- **Automatic Redirect URI Detection**: Dynamically computes redirect URIs for different environments
- **Session State Tracking**: Maintains OAuth state and authentication status across requests
- **Graceful Error Handling**: Proper session cleanup and error recovery
- **Updated OAuth Scopes**: Includes OpenID Connect scopes for enhanced security

### 3. Intelligent Journal OCR Processing
- **Automated Page Detection**: AI determines if page is Daily, Weekly, or Monthly
- **Specialized Page Adapters**: Dedicated OCR processing for each journal type
- **GPT-4o Vision**: Advanced multimodal processing for handwritten and printed text
- **Structured Extraction**: Category-specific parsing with confidence scoring
- **Pre-formatted Output**: Consistent markdown formatting matching original UI logic

### 4. Low-Confidence Item Review System
- **Confidence Detection**: Identifies OCR items below threshold (default 0.9)
- **Interactive Prefill Editing**: Auto-populates user input with detected text
- **Streamlined Workflow**: Simple press Enter to accept or edit then press Enter
- **Progress Tracking**: Shows current item number and total items to review
- **Index-based Matching**: Reliable item updates using array positions
- **Todoist Integration**: Seamless transition from review to task creation

### 5. Journal Data Storage System
- **Vector Database Storage**: Automatic storage of OCR data in Pinecone for future search
- **Deterministic IDs**: Same journal page always gets same ID (based on page type + date hash)
- **Separate Journal Index**: Dedicated sparse vector index for journal data (`journal-sparse-index`)
- **Smart Page Type Detection**: Auto-detects Daily/Weekly/Monthly from data structure
- **Metadata Storage**: Stores page type, date, content type, and upload timestamp
- **Workflow Integration**: Storage occurs before Todoist upload with optional low-confidence review

### 6. Advanced Function Calling System  
- **Streaming Functions**: Real-time progress updates during long operations
- **Session State Management**: Tracks processing states across conversations
- **Function Result Handling**: Proper data flow between streaming functions
- **Error Recovery**: Graceful handling of processing failures

### 7. Gmail RAG System
- **OAuth Integration**: Secure Gmail authentication through chat interface
- **Email Processing**: Fetch, clean, and vectorize email content
- **Pinecone Storage**: Vector database integration with duplicate prevention
- **Date Filtering**: Configurable date ranges for email fetching

### 8. Todoist Integration
- **Intelligent Duplicate Detection**: SBERT-based semantic similarity analysis
- **Task Creation**: Automatic task creation from OCR-extracted items
- **Priority Support**: Handles both priority tasks and regular to-do items
- **Due Date Parsing**: Smart date extraction from journal pages

### 9. SBERT Intelligence
- **Sentence Embeddings**: Advanced semantic understanding using transformers
- **Persistent Caching**: Performance optimization for repeated operations
- **Configurable Thresholds**: Adjustable similarity detection settings
- **Fallback Support**: Graceful degradation to text-based comparison

## Chat Interface Usage

### Starting a Conversation
1. Navigate to `/chat`
2. Login with Google OAuth (session-based, no file persistence issues)
3. Start typing or upload a journal image
4. Follow the conversational prompts

### Sample Interactions
```
User: "Hi! I have a journal page to process"
AI: "Hello! I'd be happy to help you process your journal page. Please upload the image and I'll analyze it for you."

User: [uploads daily page image]
AI: "I can see this is a Daily journal page. Let me process it for you..."
[Processing updates stream in real-time]
AI: "Processing complete! I found 3 priority tasks and 5 to-do items. Store journal data in database?"

User: "Yes"
AI: "I found 2 low-confidence items in italics. Would you like to review these before storing in database?"

User: "Yes, let me review them"
AI: "Item 1 of 2 (Confidence: 75%): [pre-fills with detected text]"
User: [edits text and presses Enter]
AI: "Item 2 of 2 (Confidence: 80%): [pre-fills with next item]"
User: [accepts as-is by pressing Enter]
AI: "Review complete! Storing in database... ✅ Journal data stored with ID: daily_2018-11-12_a1b2c3d4e5f6. Would you like to upload any tasks to Todoist?"

User: "Yes, upload to Todoist"
AI: "Uploading to Todoist... Successfully created 8 tasks, skipped 2 duplicates."
```

## API Endpoints

### Core Application
- `GET /` - System status
- `GET /chat` - Conversational interface

### Authentication (Session-Based)
- `GET /auth/google` - Initiate Gmail OAuth (stores state in session)
- `GET /auth/google/callback` - OAuth callback (retrieves state from session)
- `POST /auth/logout` - Clear authentication and session data
- `GET /auth/status` - Check auth status with session validation

### Chat System
- `POST /chat/message` - Send chat message (non-streaming)
- `GET /chat/message/stream` - Streaming chat with progress updates
- `POST /chat/upload` - Upload files for processing
- `GET /chat/session/{session_id}/history` - Get conversation history
- `DELETE /chat/session/{session_id}` - Delete chat session
- `POST /chat/clear-session` - Clear current session

### Gmail Integration
- `POST /gmail/fetch-data` - Fetch and process Gmail data

## Configuration

### Session Settings
- `SESSION_SECRET_KEY` - 32+ character random string for session encryption (required)
- Session timeout: 1 hour of inactivity
- Secure cookies with sameSite="lax"

### OCR Settings (`src/ocr/config.py`)
- `OCR_ENGINE` - OCR engine selection (default: "GPT4oOCRAdapter")

### Todoist Settings (`src/todoist/config.py`)
- `TASK_CONFIDENCE_THRESHOLD` - Low-confidence review threshold (0.0-1.0, default: 0.9)
- `SBERT_ENABLED` - Enable semantic duplicate detection (default: True)
- `SBERT_MODEL` - Sentence transformer model (default: "all-MiniLM-L6-v2")
- `SBERT_SIMILARITY_THRESHOLD` - Duplicate detection threshold (0.0-1.0, default: 0.8)
- `SBERT_CACHE_FILE` - Embedding cache path (default: "cache/embeddings_cache.pkl")

### Core Settings (`src/config.py`)
- `LOG_LEVEL` - Logging level (default: "DEBUG")
- `LOG_FILE` - Log file path (default: "logs/app.log")

### Chat Settings
- Session timeout: 1 hour of inactivity
- File upload limits: 10MB per file, image formats only
- Streaming timeout: 2 minutes for function calls

## Deployment

### Local Development
```bash
# Start with auto-reload
uvicorn src.app:app --reload

# Or use the run script
python -m src.run
```

### Google Cloud Run Deployment
The application is configured for Cloud Run deployment with:

- **Session Middleware**: Works across ephemeral instances
- **Proxy Headers**: Properly handles Cloud Run's proxy headers
- **Environment Variables**: Configured for Cloud Run environment
- **OAuth Redirect URIs**: Auto-detected for production URLs

**Required Environment Variables for Cloud Run:**
- `SESSION_SECRET_KEY` - Session encryption key
- `GOOGLE_CREDENTIALS_JSON` - Base64-encoded Google OAuth credentials
- `OPENAI_API_KEY` - OpenAI API key
- `PINECONE_API_KEY` - Pinecone API key
- `TODOIST_API_TOKEN` - Todoist API token

## Testing

Run the comprehensive test suite:

```bash
# Install pytest (if not already installed)
pip install pytest

# Run all tests with the built-in test runner
python test/run_tests.py

# Or run with pytest directly
python -m pytest test/ -v

# Run specific test files
python -m pytest test/test_basic.py -v
python -m pytest test/test_duplicate_detection.py -v
python -m pytest test/test_ocr_integration.py -v
python -m pytest test/test_monthly_date_parsing.py -v
```

**Comprehensive Test Coverage (37 tests total):**
- **Basic Setup** (3 tests): Configuration, imports, file structure
- **Duplicate Detection** (23 tests): TodoistClient and SBERTClient functionality
- **OCR Integration** (4 tests): OCR adapter routing and method signatures  
- **Monthly Date Parsing** (10 tests): Comprehensive date format validation

## Development Notes

### Architecture Highlights
- **Conversational First**: All functionality accessible through natural language
- **Streaming Architecture**: Real-time feedback for long-running operations
- **Session Management**: Stateful conversations with processing context
- **Modular Functions**: Clean separation between chat logic and core functionality
- **Error Resilience**: Graceful handling of processing failures and API errors
- **Cloud-Native**: Designed for Cloud Run deployment with proper session handling

### Security
- **Session-Based OAuth**: Encrypted session cookies with secure timeouts
- **Environment Variable Configuration**: All secrets stored as environment variables
- **Secure File Upload Handling**: File type validation and size limits
- **API Key Management**: Centralized API key configuration
- **Session Isolation**: State isolation between different user sessions

### Performance
- **Streaming Responses**: Immediate feedback for long operations
- **Intelligent Caching**: Embeddings and models cached for performance
- **Efficient Duplicate Detection**: Optimized algorithms for task deduplication
- **Session Cleanup**: Automatic memory management and session expiration
- **Optimized OCR Pipeline**: Efficient processing of journal pages

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated and dependencies installed
2. **OAuth Failures**: Check `SESSION_SECRET_KEY` is set and `google_client.json` path is correct
3. **Session Issues**: Verify session middleware is properly configured
4. **API Errors**: Verify API keys in `.env` file
5. **Chat Issues**: Clear browser cache and check browser console for errors
6. **Cloud Run Issues**: Ensure all required environment variables are set

### Debug Mode
Set `LOG_LEVEL="DEBUG"` to see detailed processing information including:
- Chat function execution traces
- OCR processing steps and confidence scores
- Todoist upload decisions and duplicate detection
- Session state management and OAuth flow
- Streaming function progress
- Error stack traces

### Monitoring
- Check server logs for backend errors (`logs/app.log`)
- Use browser developer tools for frontend issues
- Monitor API rate limits for OpenAI, Todoist, and Gmail
- Watch file upload sizes and processing times
- Monitor session creation and expiration

### OAuth Troubleshooting
- **Session Secret Missing**: Ensure `SESSION_SECRET_KEY` is set in environment
- **Redirect URI Mismatch**: Check Google Cloud Console OAuth configuration
- **Scope Issues**: Verify OAuth scopes include required Gmail and OpenID scopes
- **Token Expiration**: Clear session and re-authenticate if tokens expire