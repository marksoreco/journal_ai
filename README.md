# Journal AI - FastAPI Application

A comprehensive FastAPI application for intelligent journal processing with specialized OCR capabilities for Daily, Weekly, and Monthly journal pages, task management integration with Todoist, and Gmail-based RAG (Retrieval-Augmented Generation) system.

## Features

- **Intelligent Journal OCR Processing**: Specialized OCR adapters for Day/Week/Month journal pages using GPT-4o vision
- **Monthly Check-In Processing**: Advanced parsing of wellness ratings with comprehensive date format support
- **Interactive UI Dialogs**: Month verification and manual data entry with real-time validation
- **Todoist Integration**: Automatically create tasks from extracted text with intelligent duplicate detection
- **Gmail RAG System**: Process emails, generate embeddings, and store in Pinecone vector database
- **OAuth Authentication**: Secure Gmail API integration with token management
- **SBERT Intelligence**: Advanced semantic similarity for duplicate detection using sentence transformers
- **Comprehensive Date Parsing**: Support for multiple date formats using dateparser library
- **Modular Architecture**: Clean separation of concerns with extensive test coverage

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
│   ├── mcp_server.py       # MCP (Model Context Protocol) server
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
│   └── index.html          # Enhanced web UI with interactive dialogs
├── test/                   # Comprehensive test suite
│   ├── test_basic.py       # Basic functionality tests
│   ├── test_duplicate_detection.py # SBERT and Todoist integration tests
│   ├── test_ocr_integration.py # OCR adapter integration tests
│   ├── test_monthly_date_parsing.py # Date parsing validation tests
│   ├── run_tests.py        # Test runner
│   └── images/             # Test image files
├── uploads/                # Uploaded images
└── requirements.txt        # Python dependencies (includes dateparser)
```

## Core Features

### 1. Intelligent Journal OCR Processing
- **Specialized Page Adapters**: Dedicated OCR processing for Daily, Weekly, and Monthly journal pages
- **Smart Routing**: Automatic dispatching based on page type selection (radio buttons)
- **GPT-4o Vision**: Advanced multimodal processing for handwritten and printed text
- **Structured Extraction**: Category-specific parsing with confidence scoring
- **Monthly Check-In Processing**: 
  - Wellness ratings extraction (1-10 scale) for 7 categories
  - Advanced date parsing supporting multiple formats ("Sept 2023", "3/23", "March", etc.)
  - Interactive verification dialogs with manual override capabilities

### 2. Interactive User Interface
- **Enhanced Web UI**: Modern interface with modal dialogs and real-time validation
- **Month Verification Dialog**: User confirmation and editing of detected month/year
- **Monthly Check-In Dialog**: Manual entry and verification of wellness ratings
- **Page Type Selection**: Radio buttons for Daily/Weekly/Monthly journal processing
- **Real-time Feedback**: Progress indicators and error handling

### 3. Advanced Date Processing
- **Comprehensive Format Support**: Handles "September 2023", "Sept 2023", "3/2023", "3/23", "03/2023", "March"
- **Dateparser Integration**: Robust parsing using Python dateparser library
- **Smart Year Logic**: Intelligent 2-digit year handling (20xx/19xx)
- **Confidence Scoring**: Quality assessment for parsing accuracy
- **Fallback Handling**: Graceful defaults for invalid inputs

### 4. Gmail RAG System
- OAuth-based Gmail authentication
- Email fetching with date filtering
- Content cleaning (HTML removal, URL shortening)
- Dual embedding generation (dense + sparse)
- Pinecone vector storage with duplicate prevention
- Semantic search capabilities

### 5. Todoist Integration
- Intelligent duplicate detection using SBERT
- Semantic similarity analysis
- Automatic task creation with confidence scoring
- Fallback to simple text comparison when needed

### 6. SBERT Intelligence
- Sentence transformer embeddings (all-MiniLM-L6-v2)
- Persistent caching for performance
- Configurable similarity thresholds
- Comprehensive similarity matrix logging

### 7. MCP (Model Context Protocol) Integration
- **MCP Server**: Standardized protocol for LLM integrations
- **Context-Aware Processing**: Enhanced model interactions
- **Extensible Architecture**: Support for future AI integrations

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

### Journal OCR & Tasks
- `POST /upload-image` - Upload image for intelligent OCR processing (supports category parameter)
- `POST /upload-to-todoist` - Create Todoist tasks from OCR data with duplicate detection

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
  - SBERT initialization and fallback behavior
  - Semantic similarity analysis
  - API integration testing
  - Cache management
  - Mock response handling
- **OCR Integration** (4 tests): OCR adapter routing and method signatures
  - Category-based dispatching
  - Adapter interface compliance
  - Factory pattern validation
- **Monthly Date Parsing** (10 tests): Comprehensive date format validation
  - Full month names with year ("September 2023", "March 2024")
  - Abbreviated formats ("Sept 2023", "Mar 2024")
  - Numeric formats ("3/2023", "03/23", "12/24")
  - Month-only inputs with smart year detection
  - Edge cases and error handling
  - Integration with OCR post-processing logic
  - Confidence scoring validation

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
- Optimized date parsing with fallback mechanisms
- Frontend/backend separation for responsive UI

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated and dependencies installed
2. **OAuth Failures**: Check `credentials.json` path and redirect URI configuration
3. **API Errors**: Verify API keys in `.env` file
4. **Cache Issues**: Delete cache files in `src/todoist/cache/` and `src/rag/models/`

### Debug Mode
Set `LOG_LEVEL="DEBUG"` to see detailed processing information including:
- OCR adapter routing decisions
- Date parsing attempts and fallbacks
- Similarity matrices for duplicate detection
- Email processing steps
- Vector generation and storage details
- API call traces
- UI dialog interactions