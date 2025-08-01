# Journal AI - FastAPI Application

A FastAPI application for image upload and OCR text extraction using GPT-4o.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your API keys:

```bash
OPENAI_API_KEY=your-openai-api-key-here
TODOIST_API_TOKEN=your-todoist-api-token-here
```

## SBERT Integration (Optional)

For intelligent duplicate task detection, the system uses SBERT (Sentence Transformers) with the `all-MiniLM-L6-v2` model:

1. **Automatic Installation**: The required packages are installed via `requirements.txt`
2. **Model Download**: The SBERT model is automatically downloaded on first use (~80MB)
3. **Persistent Caching**: Embeddings are cached to avoid recomputation
4. **Configurable Threshold**: Adjust `SBERT_SIMILARITY_THRESHOLD` in `config.py` (default: 0.85)
5. **Debug Visualization**: When `LOG_LEVEL` is set to "DEBUG", similarity matrices are displayed as tables

If SBERT is not available, the system will automatically fall back to simple text comparison for duplicate detection.

## Testing

Run the test suite to verify the duplicate detection functionality:

```bash
# Run all tests
python test/run_tests.py

# Run specific test file
python -m unittest test.test_basic
python -m unittest test.test_duplicate_detection

# Run with verbose output
python -m unittest test.test_duplicate_detection -v
```

The tests include:
- Basic setup verification
- TodoistClient initialization with/without SBERT
- Intelligent duplicate detection with mock responses
- Fallback behavior when SBERT is unavailable
- Error handling for API failures
- Task upload scenarios with duplicate filtering

3. Run the application:

```bash
# Option 1: Using the run.py entry point
python -m src.run

# Option 2: Using uvicorn directly
uvicorn src.main:app --reload
```

## Project Structure

```
journal_ai/
├── src/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── config.py        # Configuration settings
│   ├── run.py           # Application entry point
│   ├── todoist_client.py # Todoist API integration
│   ├── ocr_factory.py   # OCR engine factory
│   └── ocr/
│       ├── __init__.py
│       ├── base.py      # Base OCR interface
│       └── gpt4o_ocr.py
├── static/
│   └── index.html       # Web UI
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables
```

## Endpoints

- `/` - Returns system status as JSON.
- `/ui` - Opens the Journal AI web UI for image upload.
- `/upload-image` - Accepts image uploads (POST).
- `/upload-to-todoist` - Uploads extracted tasks to Todoist (POST).
- `/config` - Returns application configuration (GET).

## Configuration

The application can be configured by modifying `src/config.py`:

- `OCR_ENGINE` - Which OCR engine to use (e.g., "GPT4oOCRAdapter")
- `TASK_CONFIDENCE_THRESHOLD` - Confidence threshold for task review (0.0-1.0, default: 0.9)
- `SBERT_ENABLED` - Whether to use SBERT for intelligent duplicate detection (default: True)
- `SBERT_MODEL` - SBERT model to use (default: "all-MiniLM-L6-v2")
- `SBERT_SIMILARITY_THRESHOLD` - Threshold for considering tasks as duplicates (0.0-1.0, default: 0.85)
- `SBERT_CACHE_FILE` - File path for persistent embedding cache (default: "embeddings_cache.pkl")
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, default: "INFO")
- `LOG_FILE` - Optional file path for log output (None for console only, default: None)

### Logging Configuration

The application uses Python's logging module with configurable levels. When `LOG_LEVEL` is set to "DEBUG", the system will display:

- Detailed similarity matrices between new and existing tasks
- Summary tables showing duplicate detection results  
- Step-by-step processing information

Example debug output includes similarity tables like:
```
=== SIMILARITY MATRIX ===
New Task                       | Check emails         | Complete project set... | 
Check emails!                  | 0.813                | 0.048                | 
Setup project                  | -0.000               | 0.879*               | 
* = Above similarity threshold (0.85)
```

4. Visit [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui) in your browser to use the Journal AI UI.
