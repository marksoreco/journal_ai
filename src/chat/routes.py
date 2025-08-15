"""
Chat API routes
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List
import os
import uuid
import json
import asyncio

from .models import ChatRequest, ChatResponse, ChatMessage
from .session_manager import session_manager
from .chat_service import chat_service
from ..gmail.auth import get_gmail_service

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/message", response_model=ChatResponse)
async def send_chat_message(
    request: ChatRequest,
    gmail_service = Depends(get_gmail_service)
) -> ChatResponse:
    """Send a message to the chat and get AI response"""
    try:
        # Get or create session
        session = session_manager.get_or_create_session(request.session_id)
        
        # Add user message to session
        user_message = ChatMessage(
            type="user",
            content=request.message
        )
        session.add_message(user_message)
        
        # Process message with OpenAI and function calling
        ai_response_content = await chat_service.process_message(session, request.message)
        
        # Add AI response to session
        ai_message = ChatMessage(
            type="assistant",
            content=ai_response_content
        )
        session.add_message(ai_message)
        
        logger.info(f"Chat exchange in session {session.session_id}")
        
        return ChatResponse(
            response=ai_response_content,
            session_id=session.session_id,
            message_id=ai_message.id
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.get("/session/{session_id}/history")
async def get_chat_history(
    session_id: str,
    limit: Optional[int] = 20,
    gmail_service = Depends(get_gmail_service)
):
    """Get chat history for a session"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    recent_messages = session.get_recent_messages(limit)
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": msg.id,
                "type": msg.type,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in recent_messages
        ]
    }

@router.delete("/session/{session_id}")
async def delete_chat_session(
    session_id: str,
    gmail_service = Depends(get_gmail_service)
):
    """Delete a chat session"""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deleted successfully"}

@router.post("/upload")
async def upload_file(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    gmail_service = Depends(get_gmail_service)
):
    """Upload files for chat processing"""
    try:
        logger.info(f"Upload request received: {len(files)} files, session_id: {session_id}")
        
        # Get or create session
        session = session_manager.get_or_create_session(session_id)
        logger.info(f"Using session: {session.session_id}")
        
        uploaded_files = []
        
        for file in files:
            logger.info(f"Processing file: {file.filename}, type: {file.content_type}")
            # Validate file type
            allowed_types = ["image/jpeg", "image/png", "image/jpg"]
            if file.content_type not in allowed_types:
                logger.warning(f"Skipping unsupported file type: {file.content_type}")
                continue
                
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            logger.info(f"Generated file ID: {file_id}")
            
            # Save file
            base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            upload_dir = os.path.join(base_dir, "uploads", "chat")
            logger.info(f"Upload directory: {upload_dir}")
            
            try:
                os.makedirs(upload_dir, exist_ok=True)
                logger.info(f"Created upload directory: {upload_dir}")
            except Exception as e:
                logger.error(f"Failed to create upload directory: {str(e)}")
                raise
            
            file_path = os.path.join(upload_dir, f"{file_id}_{file.filename}")
            logger.info(f"Saving file to: {file_path}")
            
            try:
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                logger.info(f"Successfully saved file: {file_path} ({len(content)} bytes)")
            except Exception as e:
                logger.error(f"Failed to save file: {str(e)}")
                raise
            
            # Store file info in session
            file_info = {
                "file_id": file_id,
                "filename": file.filename,
                "content_type": file.content_type,
                "path": file_path,
                "size": len(content)
            }
            
            session.uploaded_files[file_id] = file_info
            uploaded_files.append(file_info)
            
            logger.info(f"File uploaded successfully: {file.filename} -> {file_id}")
        
        result = {
            "success": True,
            "message": f"Uploaded {len(uploaded_files)} file(s)",
            "files": [{"file_id": f["file_id"], "filename": f["filename"]} for f in uploaded_files],
            "session_id": session.session_id
        }
        
        logger.info(f"Upload complete: {result}")
        return result
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/message/stream")
async def stream_chat_message(
    message: str,
    session_id: Optional[str] = None,
    gmail_service = Depends(get_gmail_service)
):
    """Stream chat message with real-time progress updates"""
    async def generate_stream():
        try:
            # Get or create session
            session = session_manager.get_or_create_session(session_id)
            
            # Add user message to session
            user_message = ChatMessage(
                type="user",
                content=message
            )
            session.add_message(user_message)
            
            # Send initial response
            yield f"data: {json.dumps({'type': 'start', 'content': 'Processing your request...'})}\n\n"
            
            # Process message with streaming progress
            async for progress_data in chat_service.process_message_stream(session, message):
                yield f"data: {json.dumps(progress_data)}\n\n"
                
        except Exception as e:
            logger.error(f"Error in streaming chat endpoint: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache", 
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@router.post("/clear-session")
async def clear_chat_session(
    session_id: Optional[str] = None,
    gmail_service = Depends(get_gmail_service)
):
    """Clear chat session to start fresh (useful for testing formatting changes)"""
    if session_id:
        success = session_manager.delete_session(session_id)
        if success:
            return {"message": "Session cleared successfully", "session_id": session_id}
        else:
            return {"message": "Session not found", "session_id": session_id}
    else:
        return {"message": "No session_id provided"}

@router.get("/health")
async def chat_health():
    """Health check for chat service"""
    return {"status": "ok", "service": "chat"}