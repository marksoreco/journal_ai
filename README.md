# Journal AI - FastAPI Application

A comprehensive FastAPI application for image OCR processing, task management integration with Todoist, and Gmail-based RAG (Retrieval-Augmented Generation) system with intelligent duplicate detection.

## Features

- **Image OCR Processing**: Upload images and extract text using GPT-4o
- **Todoist Integration**: Automatically create tasks from extracted text with intelligent duplicate detection
- **Gmail RAG System**: Process emails, generate embeddings, and store in Pinecone vector database
- **OAuth Authentication**: Secure Gmail API integration with token management
- **SBERT Intelligence**: Advanced semantic similarity for duplicate detection using sentence transformers
- **Modular Architecture**: Clean separation of concerns across packages

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

5. Visit [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui) to access the web interface.

## Project Structure

```
journal_ai/
├── src/
│   ├── app.py              # Main FastAPI application
│   ├── config.py           # Core logging configuration
│   ├── run.py              # Application entry point
│   ├── auth_routes.py      # OAuth authentication routes
│   ├── gmail/              # Gmail API integration
│   │   ├── auth.py         # Gmail authentication
│   │   ├── client.py       # Gmail API client
│   │   ├── credentials.json # OAuth credentials (not in git)
│   │   ├── downloads/      # Downloaded email data
│   │   └── token.pkl       # OAuth token (not in git)
│   ├── ocr/                # OCR processing
│   │   ├── base.py         # Base OCR interface
│   │   ├── config.py       # OCR configuration
│   │   ├── gpt4o_ocr.py    # GPT-4o OCR implementation
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
│   └── index.html          # Web UI
├── test/                   # Test suite
│   ├── test_basic.py       # Basic functionality tests
│   └── test_duplicate_detection.py # Comprehensive SBERT tests
├── uploads/                # Uploaded images
└── requirements.txt        # Python dependencies
```

## Core Features

### 1. Image OCR Processing
- Upload images through web interface
- GPT-4o powered text extraction
- Structured task identification
- Category-based organization (Day/Week/Month)

### 2. Gmail RAG System
- OAuth-based Gmail authentication
- Email fetching with date filtering
- Content cleaning (HTML removal, URL shortening)
- Dual embedding generation (dense + sparse)
- Pinecone vector storage with duplicate prevention
- Semantic search capabilities

### 3. Todoist Integration
- Intelligent duplicate detection using SBERT
- Semantic similarity analysis
- Automatic task creation with confidence scoring
- Fallback to simple text comparison when needed

### 4. SBERT Intelligence
- Sentence transformer embeddings (all-MiniLM-L6-v2)
- Persistent caching for performance
- Configurable similarity thresholds
- Comprehensive similarity matrix logging

## API Endpoints

### Core Application
- `GET /` - System status
- `GET /ui` - Web interface
- `GET /config` - Application configuration

### Authentication
- `GET /auth/google` - Initiate Gmail OAuth
- `GET /auth/google/callback` - OAuth callback
- `POST /auth/logout` - Clear authentication
- `GET /auth/status` - Check auth status

### OCR & Tasks
- `POST /upload-image` - Upload image for OCR processing
- `POST /upload-to-todoist` - Create Todoist tasks from OCR data

### Gmail & RAG
- `POST /gmail/fetch-data` - Fetch and process Gmail data

## Configuration

### OCR Settings (`src/ocr/config.py`)
- `OCR_ENGINE` - OCR engine selection (default: "GPT4oOCRAdapter")

### Todoist Settings (`src/todoist/config.py`)
- `TASK_CONFIDENCE_THRESHOLD` - Task review threshold (0.0-1.0, default: 0.9)
- `SBERT_ENABLED` - Enable semantic duplicate detection (default: True)
- `SBERT_MODEL` - Sentence transformer model (default: "all-MiniLM-L6-v2")
- `SBERT_SIMILARITY_THRESHOLD` - Duplicate detection threshold (0.0-1.0, default: 0.8)
- `SBERT_CACHE_FILE` - Embedding cache path (default: "cache/embeddings_cache.pkl")

### Core Settings (`src/config.py`)
- `LOG_LEVEL` - Logging level (default: "DEBUG")
- `LOG_FILE` - Log file path (default: None for console)

## Testing

Run the comprehensive test suite:

```bash
# Install pytest
pip install pytest

# Run all tests
python -m pytest test/ -v

# Run specific test files
python -m pytest test/test_basic.py -v
python -m pytest test/test_duplicate_detection.py -v
```

**Test Coverage:**
- **Basic Setup** (3 tests): Configuration, imports, file structure
- **Duplicate Detection** (20 tests): TodoistClient and SBERTClient functionality
  - SBERT initialization and fallback behavior
  - Semantic similarity analysis
  - API integration testing
  - Cache management
  - Mock response handling

## Development Notes

### File Organization
- Modular package structure with clear separation of concerns
- Configuration files co-located with related functionality  
- Automatic cache directory creation and management
- Comprehensive `.gitignore` for temporary files

### Security
- OAuth tokens stored in package directories (excluded from git)
- Environment variable configuration
- Secure credential handling
- API key management

### Performance
- Intelligent caching for embeddings and models
- Batch processing for vector operations
- Efficient duplicate detection algorithms
- Minimal memory footprint for large datasets

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated and dependencies installed
2. **OAuth Failures**: Check `credentials.json` path and redirect URI configuration
3. **API Errors**: Verify API keys in `.env` file
4. **Cache Issues**: Delete cache files in `src/todoist/cache/` and `src/rag/models/`

### Debug Mode
Set `LOG_LEVEL="DEBUG"` to see detailed processing information including:
- Similarity matrices for duplicate detection
- Email processing steps
- Vector generation and storage details
- API call traces