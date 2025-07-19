# Journal AI - FastAPI Application

A FastAPI application for image upload and OCR text extraction using GPT-4o.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

3. Run the application:

```bash
# Option 1: Using the run.py entry point
python -m src.run

# Option 2: Using uvicorn directly
python -m uvicorn src.main:app --reload
```

## Project Structure

```
journal_ai/
├── src/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── config.py        # Configuration settings
│   ├── run.py           # Application entry point
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

4. Visit [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui) in your browser to use the Journal AI UI.
