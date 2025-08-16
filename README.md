# Journal AI - Conversational Journal Processing System

A comprehensive FastAPI application for intelligent journal processing through a conversational AI interface. Features specialized OCR capabilities for Daily, Weekly, and Monthly journal pages, task management integration with Todoist, low-confidence item review system, and Gmail-based RAG (Retrieval-Augmented Generation) system.

## Features

- **Conversational AI Interface**: Modern chat-based interface for all journal processing workflows
- **Intelligent Journal OCR Processing**: Specialized OCR adapters for Day/Week/Month journal pages using GPT-4o vision
- **Low-Confidence Item Review**: Interactive prefill editing system for uncertain OCR results before Todoist upload
- **Streaming Function Execution**: Real-time progress updates during OCR processing and task operations
- **Todoist Integration**: Automatically create tasks from extracted text with intelligent duplicate detection
- **Gmail RAG System**: Process emails, generate embeddings, and store in Pinecone vector database
- **OAuth Authentication**: Secure Gmail API integration with token management
- **SBERT Intelligence**: Advanced semantic similarity for duplicate detection using sentence transformers
- **Session Management**: Persistent chat sessions with processing state tracking

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

# Gmail OAuth Configuration  
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
GOOGLE_CREDENTIALS_PATH=src/gmail/credentials.json

# Optional: Milvus Configuration (for future use)
MILVUS_USERNAME=your-username
MILVUS_PASSWORD=your-password
MILVUS_URI=your-milvus-uri
```

3. **Set up Gmail API credentials:**
   - Create a Google Cloud project and enable Gmail API
   - Download credentials and place as `src/gmail/credentials.json`

4. **Run the application:**

```bash
# Using the run.py entry point
python -m src.run

# Or using uvicorn directly
uvicorn src.app:app --reload
```

5. Visit [http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat) to access the conversational interface.

## Project Structure

```
journal_ai/
├── src/
│   ├── app.py              # Main FastAPI application
│   ├── config.py           # Core logging configuration
│   ├── run.py              # Application entry point
│   ├── auth_routes.py      # OAuth authentication routes
│   ├── mcp_server.py       # MCP (Model Context Protocol) server
│   ├── chat/               # Conversational AI system
│   │   ├── routes.py       # Chat API endpoints
│   │   ├── chat_service.py # OpenAI integration & function calling
│   │   ├── function_tools.py # Chat function implementations
│   │   ├── models.py       # Chat data models
│   │   ├── session_manager.py # Session state management
│   │   └── ocr_formatter.py   # OCR results formatting
│   ├── agents/             # AI agents and tools
│   │   ├── journal_processing_agent.py # LangChain agent workflow
│   │   └── tools/
│   │       └── page_detector.py # Page type detection
│   ├── gmail/              # Gmail API integration
│   │   ├── auth.py         # Gmail authentication
│   │   ├── client.py       # Gmail API client
│   │   ├── credentials.json # OAuth credentials (not in git)
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
└── requirements.txt        # Python dependencies
```

## Core Features

### 1. Conversational AI Interface
- **Modern Chat Interface**: Clean, responsive chat UI for all interactions
- **File Upload Integration**: Drag & drop or browse for journal page images
- **Real-time Streaming**: Live progress updates during processing operations
- **Session Persistence**: Maintains conversation history and processing state
- **Authentication Integration**: Seamless OAuth login/logout experience

### 2. Intelligent Journal OCR Processing
- **Automated Page Detection**: AI determines if page is Daily, Weekly, or Monthly
- **Specialized Page Adapters**: Dedicated OCR processing for each journal type
- **GPT-4o Vision**: Advanced multimodal processing for handwritten and printed text
- **Structured Extraction**: Category-specific parsing with confidence scoring
- **Pre-formatted Output**: Consistent markdown formatting matching original UI logic

### 3. Low-Confidence Item Review System
- **Confidence Detection**: Identifies OCR items below threshold (default 0.9)
- **Interactive Prefill Editing**: Auto-populates user input with detected text
- **Streamlined Workflow**: Simple press Enter to accept or edit then press Enter
- **Progress Tracking**: Shows current item number and total items to review
- **Index-based Matching**: Reliable item updates using array positions
- **Todoist Integration**: Seamless transition from review to task creation

### 4. Advanced Function Calling System
- **Streaming Functions**: Real-time progress updates during long operations
- **Session State Management**: Tracks processing states across conversations
- **Function Result Handling**: Proper data flow between streaming functions
- **Error Recovery**: Graceful handling of processing failures

### 5. Gmail RAG System
- **OAuth Integration**: Secure Gmail authentication through chat interface
- **Email Processing**: Fetch, clean, and vectorize email content
- **Pinecone Storage**: Vector database integration with duplicate prevention
- **Date Filtering**: Configurable date ranges for email fetching

### 6. Todoist Integration
- **Intelligent Duplicate Detection**: SBERT-based semantic similarity analysis
- **Task Creation**: Automatic task creation from OCR-extracted items
- **Priority Support**: Handles both priority tasks and regular to-do items
- **Due Date Parsing**: Smart date extraction from journal pages

### 7. SBERT Intelligence
- **Sentence Embeddings**: Advanced semantic understanding using transformers
- **Persistent Caching**: Performance optimization for repeated operations
- **Configurable Thresholds**: Adjustable similarity detection settings
- **Fallback Support**: Graceful degradation to text-based comparison

## Chat Interface Usage

### Starting a Conversation
1. Navigate to `/chat`
2. Login with Google OAuth
3. Start typing or upload a journal image
4. Follow the conversational prompts

### Sample Interactions
```
User: "Hi! I have a journal page to process"
AI: "Hello! I'd be happy to help you process your journal page. Please upload the image and I'll analyze it for you."

User: [uploads daily page image]
AI: "I can see this is a Daily journal page. Let me process it for you..."
[Processing updates stream in real-time]
AI: "Processing complete! I found 3 priority tasks and 5 to-do items. Would you like to upload any tasks to Todoist?"

User: "Yes, upload to Todoist"
AI: "I found 2 low-confidence items in italics. Would you like to review these before uploading?"

User: "Yes, let me review them"
AI: "Item 1 of 2 (Confidence: 75%): [pre-fills with detected text]"
User: [edits text and presses Enter]
AI: "Item 2 of 2 (Confidence: 80%): [pre-fills with next item]"
User: [accepts as-is by pressing Enter]
AI: "Review complete! Uploading to Todoist... Successfully created 8 tasks, skipped 2 duplicates."
```

## API Endpoints

### Core Application
- `GET /` - System status
- `GET /chat` - Conversational interface

### Authentication
- `GET /auth/google` - Initiate Gmail OAuth
- `GET /auth/google/callback` - OAuth callback
- `POST /auth/logout` - Clear authentication
- `GET /auth/status` - Check auth status

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
- `LOG_FILE` - Log file path (default: None for console)

### Chat Settings
- Session timeout: 1 hour of inactivity
- File upload limits: 10MB per file, image formats only
- Streaming timeout: 2 minutes for function calls

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

### Security
- OAuth tokens stored securely (excluded from git)
- Environment variable configuration
- Secure file upload handling
- API key management
- Session-based state isolation

### Performance
- Streaming responses for immediate feedback
- Intelligent caching for embeddings and models
- Efficient duplicate detection algorithms
- Session cleanup and memory management
- Optimized OCR processing pipeline

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated and dependencies installed
2. **OAuth Failures**: Check `credentials.json` path and redirect URI configuration
3. **API Errors**: Verify API keys in `.env` file
4. **Chat Issues**: Clear browser cache and check browser console for errors
5. **Session Problems**: Use "Clear Session" button or delete session via API

### Debug Mode
Set `LOG_LEVEL="DEBUG"` to see detailed processing information including:
- Chat function execution traces
- OCR processing steps and confidence scores
- Todoist upload decisions and duplicate detection
- Session state management
- Streaming function progress
- Error stack traces

### Monitoring
- Check server logs for backend errors
- Use browser developer tools for frontend issues
- Monitor API rate limits for OpenAI, Todoist, and Gmail
- Watch file upload sizes and processing times