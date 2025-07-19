import json
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
from fastapi import UploadFile, File, HTTPException, Form
from typing import List
import os
from .config import OCR_ENGINE
from .ocr.base import BaseOCR
import importlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OCR engine factory

def get_ocr_engine():
    if OCR_ENGINE == "GPT4oOCRAdapter":
        from .ocr.gpt4o_ocr import GPT4oOCRAdapter
        return GPT4oOCRAdapter()
    # Add more engines here as needed
    else:
        raise ValueError(f"Unknown OCR_ENGINE: {OCR_ENGINE}")

ocr_engine = get_ocr_engine()

app = FastAPI()

@app.get("/", response_class=JSONResponse)
def system_status():
    return {"status": "ok", "message": "Journal AI is running"}

@app.get("/ui", response_class=FileResponse)
def serve_index():
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html"))

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    category: str = Form(...)
):
    allowed_types = [
        "image/jpeg",
        "image/png"
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid image type. Allowed types: jpg, jpeg, and png.")
    allowed_categories = {"Day", "Week", "Month"}
    if category not in allowed_categories:
        raise HTTPException(status_code=400, detail="Invalid category. Must be Day, Week, or Month.")

    # Save the uploaded file to disk
    if '__file__' in globals():
        base_dir = os.path.abspath(os.path.dirname(__file__))
    else:
        base_dir = os.getcwd()
    upload_dir = os.path.join(os.path.dirname(base_dir), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filename = file.filename or "uploaded_image"
    file_path = os.path.join(upload_dir, filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # OCR processing
    ocr_text = ocr_engine.extract_text(file_path)
    output_json = {
        "filename": file.filename,
        "content_type": file.content_type,
        "category": category,
        "ocr_text": ocr_text,
        "message": "Image uploaded and processed successfully."
    }
    print(json.dumps(output_json))
    return output_json