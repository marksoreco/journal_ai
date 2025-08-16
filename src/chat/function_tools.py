"""
Function tool definitions and execution handlers for chat
"""
import json
import logging
import os
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import asyncio
from ..agents.journal_processing_agent import JournalProcessingAgent
from ..todoist.todoist_client import TodoistClient
from ..gmail.client import GmailClient
from ..agents.tools.page_detector import PageTypeDetector
from .models import ChatSession
from .ocr_formatter import OCRFormatter

logger = logging.getLogger(__name__)

def get_function_tools() -> List[Dict]:
    """Get OpenAI function tool definitions"""
    return [
        {
            "name": "process_journal_image",
            "description": "Process a journal page image with OCR to extract structured content like tasks, events, and notes",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "ID of the uploaded image file to process"
                    },
                    "page_type": {
                        "type": "string",
                        "enum": ["Daily", "Weekly", "Monthly"],
                        "description": "Type of journal page (optional - will auto-detect if not provided)"
                    }
                },
                "required": ["file_id"]
            }
        },
        {
            "name": "upload_to_todoist",
            "description": "Upload tasks from processed journal data to Todoist with confidence review",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_data": {
                        "type": "string",
                        "description": "JSON string of task data from journal processing"
                    },
                    "review_low_confidence": {
                        "type": "boolean",
                        "description": "Whether to review low confidence items (default: true)",
                        "default": True
                    }
                },
                "required": ["task_data"]
            }
        },
        {
            "name": "fetch_gmail_data",
            "description": "Fetch Gmail emails and store them in the vector database for future queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "since_date": {
                        "type": "string",
                        "description": "Fetch emails since this date (YYYY-MM-DD format)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of emails to fetch (default: 50)",
                        "default": 50
                    }
                },
                "required": ["since_date"]
            }
        },
        {
            "name": "detect_page_type", 
            "description": "Detect the type of journal page from an uploaded image",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "ID of the uploaded image file to analyze"
                    }
                },
                "required": ["file_id"]
            }
        },
        {
            "name": "process_edited_item",
            "description": "Process user's edited or accepted low-confidence item text",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_text": {
                        "type": "string",
                        "description": "The final text for the item (either original or user-edited)"
                    }
                },
                "required": ["item_text"]
            }
        },
        {
            "name": "start_review_from_session",
            "description": "Start reviewing low-confidence items using data stored from recent journal processing",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]


async def execute_function_call_stream(session: ChatSession, function_name: str, function_args: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """Execute a function call with streaming progress updates"""
    try:
        if function_name == "process_journal_image":
            async for progress_data in _process_journal_image_stream(session, **function_args):
                yield progress_data
        elif function_name == "upload_to_todoist":
            async for progress_data in _upload_to_todoist_stream(session, **function_args):
                yield progress_data
        elif function_name == "fetch_gmail_data":
            async for progress_data in _fetch_gmail_data_stream(session, **function_args):
                yield progress_data
        elif function_name == "detect_page_type":
            async for progress_data in _detect_page_type_stream(session, **function_args):
                yield progress_data
        elif function_name == "process_edited_item":
            async for progress_data in _process_edited_item_stream(session, **function_args):
                yield progress_data
        elif function_name == "start_review_from_session":
            async for progress_data in _start_review_from_session_stream(session, **function_args):
                yield progress_data
        else:
            yield {"type": "error", "content": f"Unknown function: {function_name}"}
            
    except Exception as e:
        logger.error(f"Error executing streaming function {function_name}: {str(e)}")
        yield {"type": "error", "content": f"Function execution failed: {str(e)}"}

async def _process_journal_image(session: ChatSession, file_id: str, page_type: Optional[str] = None) -> Dict[str, Any]:
    """Process journal image using the existing LangChain agent with progress updates"""
    try:
        # Get file path from session
        if file_id not in session.uploaded_files:
            return {"error": f"File {file_id} not found in session"}
        
        file_info = session.uploaded_files[file_id]
        file_path = file_info["path"]
        
        # Step 1: Start processing
        progress_updates = []
        progress_updates.append("🔄 Starting journal image processing...")
        
        # Step 2: Page type detection
        progress_updates.append("🔍 Detecting page type (Daily, Weekly, or Monthly)...")
        try:
            from ..agents.tools.page_detector import PageTypeDetector
            detector = PageTypeDetector()
            detection_result = detector.detect_page_type(file_path)
            detected_page_type = detection_result.page_type.value
            progress_updates.append(f"✅ Detected page type: {detected_page_type}")
            progress_updates.append(f"📋 Reasoning: {detection_result.reasoning}")
        except Exception as e:
            progress_updates.append(f"❌ Page detection failed: {str(e)}")
            return {
                "success": False,
                "error": f"Page detection failed: {str(e)}",
                "progress": progress_updates
            }
        
        # Step 3: OCR Processing
        progress_updates.append("📝 Processing image with OCR to extract content...")
        try:
            from ..ocr.gpt4o_ocr import GPT4oOCRAdapter
            ocr_adapter = GPT4oOCRAdapter()
            ocr_result = ocr_adapter.extract_text(file_path, detected_page_type)
            progress_updates.append("✅ OCR processing completed successfully")
            
            # Count extracted items if possible
            if isinstance(ocr_result, dict):
                item_count = 0
                for section, items in ocr_result.items():
                    if isinstance(items, list):
                        item_count += len(items)
                if item_count > 0:
                    progress_updates.append(f"📊 Extracted {item_count} items from {len(ocr_result)} sections")
            
        except Exception as e:
            progress_updates.append(f"❌ OCR processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"OCR processing failed: {str(e)}",
                "progress": progress_updates
            }
        
        # Step 4: Complete processing
        progress_updates.append("🎉 Journal processing completed successfully!")
        
        # Store results in session
        # CACHING DISABLED: Do not store processing state
        # session.processing_states[file_id] = "completed"
        logger.info("Caching disabled - processing completion not stored")
        
        # Get confidence threshold for formatting
        from ..todoist.config import TASK_CONFIDENCE_THRESHOLD
        
        # Format results using server-side formatter (same logic as classic UI)  
        try:
            logger.info(f"About to format {detected_page_type} page with OCR result type: {type(ocr_result)}")
            
            # Parse OCR result if it's a JSON string
            if isinstance(ocr_result, str):
                try:
                    import json
                    ocr_data = json.loads(ocr_result)
                    logger.info(f"Successfully parsed JSON OCR result with keys: {list(ocr_data.keys())}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse OCR result as JSON: {e}")
                    logger.error(f"Raw OCR result: {ocr_result}")
                    raise Exception(f"OCR result is not valid JSON: {e}")
            else:
                ocr_data = ocr_result
                logger.info(f"OCR result is already parsed, keys: {list(ocr_data.keys()) if isinstance(ocr_data, dict) else 'Not a dict'}")
            
            formatter = OCRFormatter(confidence_threshold=TASK_CONFIDENCE_THRESHOLD)
            formatted_content = formatter.format_ocr_results(detected_page_type, ocr_data)
            
            # Detect low-confidence items for potential review
            low_confidence_items = formatter.detect_low_confidence_items(ocr_data)
            logger.info(f"Successfully formatted {detected_page_type} page with {len(formatted_content)} characters")
            logger.info(f"Detected {len(low_confidence_items)} low-confidence items")
        except Exception as format_error:
            logger.error(f"Formatting error: {str(format_error)}")
            logger.error(f"OCR result that caused error: {ocr_result}")
            # Fallback to basic formatting if server-side formatting fails
            formatted_content = f"**{detected_page_type} Page**\n\nProcessing completed but formatting failed: {str(format_error)}\n\nRaw data available for manual review."
            low_confidence_items = []  # Empty list if formatting fails
        
        # Store low-confidence review data in session for later access
        if low_confidence_items:
            session.processing_states['pending_review'] = {
                'low_confidence_items': low_confidence_items,
                'ocr_data': ocr_data,
                'has_items': True
            }
        else:
            session.processing_states['pending_review'] = {
                'has_items': False
            }

        return {
            "success": True,
            "message": "Journal page processed successfully",
            "page_type": detected_page_type,
            "ocr_data": ocr_result,
            "formatted_content": formatted_content,
            "confidence_threshold": TASK_CONFIDENCE_THRESHOLD,
            "low_confidence_items": low_confidence_items,
            "progress": progress_updates,
            "filename": file_info["filename"]
        }
            
    except Exception as e:
        logger.error(f"Error in process_journal_image: {str(e)}")
        progress_updates.append(f"❌ Processing failed: {str(e)}")
        return {
            "error": str(e),
            "progress": progress_updates
        }

async def _process_journal_image_stream(session: ChatSession, file_id: str, page_type: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
    """Process journal image with real-time streaming progress updates"""
    try:
        # Get file path from session
        if file_id not in session.uploaded_files:
            yield {"type": "error", "content": f"File {file_id} not found in session"}
            return
        
        file_info = session.uploaded_files[file_id]
        file_path = file_info["path"]
        
        # CACHING DISABLED: Clear all processing states to force fresh processing
        session.processing_states.clear()
        logger.info("Cleared all cached processing states to force fresh processing")
        
        # Send initial processing message with filename
        filename = file_info["filename"]
        yield {"type": "progress", "content": f"🔄 Processing {filename}..."}
        await asyncio.sleep(0.2)  # Small delay for visual effect
        
        # Step 1: Page type detection
        yield {"type": "progress", "content": "🔍 Detecting page type (Daily, Weekly, or Monthly)..."}
        await asyncio.sleep(0.1)  # Small delay for visual effect
        
        try:
            from ..agents.tools.page_detector import PageTypeDetector
            detector = PageTypeDetector()
            detection_result = detector.detect_page_type(file_path)
            detected_page_type = detection_result.page_type.value
            
            yield {"type": "progress", "content": f"✅ Detected page type: {detected_page_type}"}
            await asyncio.sleep(0.5)
            
        except Exception as e:
            yield {"type": "progress", "content": f"❌ Page detection failed: {str(e)}"}
            return
        
        # Step 3: OCR Processing
        yield {"type": "progress", "content": "📝 Processing image with OCR to extract content..."}
        # Note: OCR processing is the time-consuming part (30-60 seconds)
        
        try:
            from ..ocr.gpt4o_ocr import GPT4oOCRAdapter
            ocr_adapter = GPT4oOCRAdapter()
            ocr_result = ocr_adapter.extract_text(file_path, detected_page_type)
            
            yield {"type": "progress", "content": "✅ OCR processing completed successfully"}
            
            # Count extracted items if possible
            if isinstance(ocr_result, dict):
                item_count = 0
                for section, items in ocr_result.items():
                    if isinstance(items, list):
                        item_count += len(items)
                if item_count > 0:
                    yield {"type": "progress", "content": f"📊 Extracted {item_count} items from {len(ocr_result)} sections"}
            
            # Add preparing results message
            yield {"type": "progress", "content": "🔄 Preparing results for display..."}
            
        except Exception as e:
            yield {"type": "progress", "content": f"❌ OCR processing failed: {str(e)}"}
            return
        
        # Format results using server-side formatter (same logic as classic UI)
        try:
            from ..todoist.config import TASK_CONFIDENCE_THRESHOLD
            logger.info(f"About to format {detected_page_type} page with OCR result type: {type(ocr_result)}")
            
            # Parse OCR result if it's a JSON string
            if isinstance(ocr_result, str):
                try:
                    import json
                    ocr_data = json.loads(ocr_result)
                    logger.info(f"Successfully parsed JSON OCR result with keys: {list(ocr_data.keys())}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse OCR result as JSON: {e}")
                    logger.error(f"Raw OCR result: {ocr_result}")
                    raise Exception(f"OCR result is not valid JSON: {e}")
            else:
                ocr_data = ocr_result
                logger.info(f"OCR result is already parsed, keys: {list(ocr_data.keys()) if isinstance(ocr_data, dict) else 'Not a dict'}")
            
            formatter = OCRFormatter(confidence_threshold=TASK_CONFIDENCE_THRESHOLD)
            formatted_content = formatter.format_ocr_results(detected_page_type, ocr_data)
            
            # Detect low-confidence items for potential review
            low_confidence_items = formatter.detect_low_confidence_items(ocr_data)
            logger.info(f"Successfully formatted {detected_page_type} page with {len(formatted_content)} characters")
            logger.info(f"Detected {len(low_confidence_items)} low-confidence items")
        except Exception as format_error:
            logger.error(f"Formatting error: {str(format_error)}")
            logger.error(f"OCR result that caused error: {ocr_result}")
            # Fallback to basic formatting if server-side formatting fails
            formatted_content = f"**{detected_page_type} Page**\n\nProcessing completed but formatting failed: {str(format_error)}\n\nRaw data available for manual review."
            low_confidence_items = []  # Empty list if formatting fails
        
        # Store results in session for potential low-confidence review
        try:
            session.processing_states['pending_review'] = {
                'has_items': len(low_confidence_items) > 0,
                'low_confidence_items': low_confidence_items,
                'ocr_data': ocr_data,
                'page_type': detected_page_type
            }
            yield {"type": "progress", "content": "💾 Results saved to session"}
        except Exception as e:
            yield {"type": "progress", "content": f"⚠️ Could not save to session: {str(e)}"}
        
        yield {"type": "progress", "content": "🎉 Journal processing completed successfully!"}
        
        # Yield the final result for AI to process
        yield {
            "type": "function_result",
            "data": {
                "success": True,
                "formatted_content": formatted_content,
                "page_type": detected_page_type,
                "low_confidence_items": low_confidence_items,
                "ocr_data": ocr_data
            }
        }
        
    except Exception as e:
        logger.error(f"Error in streaming process_journal_image: {str(e)}")
        yield {"type": "progress", "content": f"❌ Processing failed: {str(e)}"}

def _apply_reviewed_items_to_ocr_data(original_data: Dict[str, Any], reviewed_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply reviewed low-confidence items back to the original OCR data"""
    try:
        # Create a copy to avoid modifying the original
        import copy
        updated_data = copy.deepcopy(original_data)
        
        # Create index-based mapping instead of text-based
        logger.info(f"Processing {len(reviewed_items)} reviewed items for OCR data update")
        
        for review_item in reviewed_items:
            section = review_item.get('section')  # e.g., 'prepare_priority'
            item_index = review_item.get('item_index')
            field_name = review_item.get('field_name', 'task')
            new_text = review_item.get('text')
            
            logger.info(f"Processing review item: section='{section}', index={item_index}, field='{field_name}'")
            logger.info(f"  Original -> New: '{review_item.get('original_item', {}).get(field_name, 'N/A')}' -> '{new_text}'")
            
            # Use direct index-based update
            if section and item_index is not None and updated_data.get(section):
                section_items = updated_data[section]
                if isinstance(section_items, list) and item_index < len(section_items):
                    target_item = section_items[item_index]
                    if isinstance(target_item, dict):
                        old_text = target_item.get(field_name, 'N/A')
                        target_item[field_name] = new_text
                        target_item['confidence'] = review_item.get('confidence', 1.0)
                        logger.info(f"✅ Updated {section}[{item_index}].{field_name}: '{old_text}' -> '{new_text}'")
                    else:
                        logger.error(f"❌ Target item is not a dict: {target_item}")
                else:
                    logger.error(f"❌ Invalid index {item_index} for section '{section}' (length: {len(section_items) if isinstance(section_items, list) else 'not a list'})")
            else:
                logger.error(f"❌ Missing data: section='{section}', index={item_index}, section_exists={section in updated_data if updated_data else False}")
        
        logger.info("Index-based OCR data update completed")
        
        return updated_data
        
    except Exception as e:
        logger.error(f"Error applying reviewed items: {str(e)}")
        return original_data  # Return original if update fails

async def _upload_to_todoist(session: ChatSession, task_data: str, review_low_confidence: bool = True) -> Dict[str, Any]:
    """Upload tasks to Todoist with progress updates"""
    try:
        progress_updates = []
        progress_updates.append("📋 Starting Todoist upload process...")
        
        # Check if there are reviewed items from a completed review session
        review_state = session.get_low_confidence_review_state()
        has_reviewed_items = (review_state and 
                             not review_state.get('active') and 
                             review_state.get('reviewed_items'))
        
        # Parse task data
        progress_updates.append("🔍 Parsing task data...")
        if isinstance(task_data, str):
            parsed_data = json.loads(task_data)
        else:
            parsed_data = task_data
        
        # Apply reviewed items if available
        if has_reviewed_items:
            progress_updates.append("📝 Applying reviewed low-confidence items...")
            original_ocr_data = review_state['ocr_data']
            reviewed_items = review_state['reviewed_items']
            parsed_data = _apply_reviewed_items_to_ocr_data(original_ocr_data, reviewed_items)
            progress_updates.append(f"✅ Applied {len(reviewed_items)} reviewed items")
            
            # Clear the review state
            session.clear_low_confidence_review_state()
        
        # Count tasks to upload
        task_count = 0
        if isinstance(parsed_data, dict):
            for section, items in parsed_data.items():
                if isinstance(items, list):
                    task_count += len(items)
        
        if task_count > 0:
            progress_updates.append(f"📊 Found {task_count} potential tasks to upload")
        
        # Use existing Todoist client
        progress_updates.append("🚀 Uploading tasks to Todoist...")
        todoist_client = TodoistClient()
        result = todoist_client.upload_tasks_from_ocr(parsed_data)
        
        # Add completion message
        created_count = result.get("created_count", 0)
        skipped_count = result.get("skipped_count", 0)
        
        if created_count > 0:
            progress_updates.append(f"✅ Successfully created {created_count} new tasks in Todoist")
        if skipped_count > 0:
            progress_updates.append(f"⏭️ Skipped {skipped_count} duplicate tasks")
        
        progress_updates.append("🎉 Todoist upload completed!")
        
        return {
            "success": True,
            "message": result.get("message", "Tasks uploaded successfully"),
            "created_count": created_count,
            "skipped_count": skipped_count,
            "total_tasks": result.get("total_tasks", 0),
            "progress": progress_updates
        }
        
    except Exception as e:
        logger.error(f"Error in upload_to_todoist: {str(e)}")
        progress_updates.append(f"❌ Upload failed: {str(e)}")
        return {
            "error": str(e),
            "progress": progress_updates
        }

async def _upload_to_todoist_stream(session: ChatSession, task_data: str, review_low_confidence: bool = True) -> AsyncGenerator[Dict[str, Any], None]:
    """Upload tasks to Todoist with real-time streaming progress updates"""
    try:
        yield {"type": "progress", "content": "📋 Starting Todoist upload process..."}
        await asyncio.sleep(0.2)
        
        # Check if there are reviewed items from a completed review session
        review_state = session.get_low_confidence_review_state()
        has_reviewed_items = (review_state and 
                             not review_state.get('active') and 
                             review_state.get('reviewed_items'))
        
        # Parse task data
        yield {"type": "progress", "content": "🔍 Parsing task data..."}
        await asyncio.sleep(0.3)
        
        if isinstance(task_data, str):
            parsed_data = json.loads(task_data)
        else:
            parsed_data = task_data
        
        # Apply reviewed items if available
        if has_reviewed_items:
            yield {"type": "progress", "content": "📝 Applying reviewed low-confidence items..."}
            await asyncio.sleep(0.3)
            original_ocr_data = review_state['ocr_data']
            reviewed_items = review_state['reviewed_items']
            parsed_data = _apply_reviewed_items_to_ocr_data(original_ocr_data, reviewed_items)
            yield {"type": "progress", "content": f"✅ Applied {len(reviewed_items)} reviewed items"}
            await asyncio.sleep(0.2)
            
            # Clear the review state
            session.clear_low_confidence_review_state()
        
        # Count tasks to upload
        task_count = 0
        if isinstance(parsed_data, dict):
            for section, items in parsed_data.items():
                if isinstance(items, list):
                    task_count += len(items)
        
        if task_count > 0:
            yield {"type": "progress", "content": f"📊 Found {task_count} potential tasks to upload"}
            await asyncio.sleep(0.3)
        
        # Use existing Todoist client
        yield {"type": "progress", "content": "🚀 Uploading tasks to Todoist..."}
        await asyncio.sleep(0.5)
        
        # Debug: Log what we're actually uploading
        if has_reviewed_items:
            logger.info("=== UPLOADING WITH REVIEWED ITEMS ===")
            for section, items in parsed_data.items():
                if isinstance(items, list) and items:
                    logger.info(f"Section '{section}': {len(items)} items")
                    for i, item in enumerate(items):
                        if isinstance(item, dict):
                            text = item.get('task') or item.get('item') or item.get('value', 'No text')
                            confidence = item.get('confidence', 'No confidence')
                            logger.info(f"  Item {i+1}: '{text}' (confidence: {confidence})")
        
        todoist_client = TodoistClient()
        result = todoist_client.upload_tasks_from_ocr(parsed_data)
        
        # Add completion message
        created_count = result.get("created_count", 0)
        skipped_count = result.get("skipped_count", 0)
        
        if created_count > 0:
            yield {"type": "progress", "content": f"✅ Successfully created {created_count} new tasks in Todoist"}
            await asyncio.sleep(0.3)
        if skipped_count > 0:
            yield {"type": "progress", "content": f"⏭️ Skipped {skipped_count} duplicate tasks"}
            await asyncio.sleep(0.3)
        
        yield {"type": "progress", "content": "🎉 Todoist upload completed!"}
        
        # Yield the final result for AI to process
        yield {
            "type": "function_result",
            "data": {
                "success": True,
                "message": result.get("message", "Tasks uploaded successfully"),
                "created_count": created_count,
                "skipped_count": skipped_count,
                "total_tasks": result.get("total_tasks", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in streaming upload_to_todoist: {str(e)}")
        yield {"type": "progress", "content": f"❌ Upload failed: {str(e)}"}
        yield {
            "type": "function_result", 
            "data": {
                "error": str(e),
                "success": False
            }
        }

async def _fetch_gmail_data(session: ChatSession, since_date: str, limit: int = 50) -> Dict[str, Any]:
    """Fetch Gmail data and store in vector database with progress updates"""
    try:
        progress_updates = []
        progress_updates.append("📧 Starting Gmail data fetch...")
        
        # Parse date
        progress_updates.append(f"📅 Fetching emails since {since_date} (limit: {limit})")
        since_date_obj = datetime.strptime(since_date, "%Y-%m-%d")
        
        # Use existing Gmail client
        progress_updates.append("🔍 Connecting to Gmail API...")
        gmail_client = GmailClient()
        emails = gmail_client.get_emails_since_date(since_date_obj, limit)
        
        progress_updates.append(f"📬 Found {len(emails)} emails")
        
        # Convert to serializable format
        progress_updates.append("🔄 Processing email data...")
        emails_data = []
        for email in emails:
            emails_data.append({
                "id": email.id,
                "subject": email.subject,
                "sender": email.sender,
                "date": email.date.isoformat(),
                "snippet": email.snippet
            })
        
        # Upload to Pinecone (existing functionality)
        progress_updates.append("🗂️ Storing emails in vector database...")
        from ..rag.email_vectorizer import EmailVectorizer
        try:
            vectorizer = EmailVectorizer()
            vectorizer_results = vectorizer.process_and_store_emails(emails)
            progress_updates.append("✅ Successfully stored emails in Pinecone vector database")
        except Exception as vec_error:
            logger.error(f"Vector upload failed: {str(vec_error)}")
            progress_updates.append(f"❌ Vector database storage failed: {str(vec_error)}")
            vectorizer_results = {"error": str(vec_error)}
        
        progress_updates.append("🎉 Gmail data fetch completed!")
        
        return {
            "success": True,
            "message": f"Fetched {len(emails_data)} emails since {since_date}",
            "total_emails": len(emails_data),
            "emails_sample": emails_data[:5],  # Show first 5 as sample
            "pinecone_upload": vectorizer_results,
            "progress": progress_updates
        }
        
    except Exception as e:
        logger.error(f"Error in fetch_gmail_data: {str(e)}")
        progress_updates.append(f"❌ Gmail fetch failed: {str(e)}")
        return {
            "error": str(e),
            "progress": progress_updates
        }

async def _fetch_gmail_data_stream(session: ChatSession, since_date: str, limit: int = 50) -> AsyncGenerator[Dict[str, Any], None]:
    """Fetch Gmail data with real-time streaming progress updates"""
    try:
        yield {"type": "progress", "content": "📧 Starting Gmail data fetch..."}
        await asyncio.sleep(0.3)
        
        # Parse date
        yield {"type": "progress", "content": f"📅 Fetching emails since {since_date} (limit: {limit})"}
        await asyncio.sleep(0.3)
        
        since_date_obj = datetime.strptime(since_date, "%Y-%m-%d")
        
        # Use existing Gmail client
        yield {"type": "progress", "content": "🔍 Connecting to Gmail API..."}
        await asyncio.sleep(0.5)
        
        gmail_client = GmailClient()
        emails = gmail_client.get_emails_since_date(since_date_obj, limit)
        
        yield {"type": "progress", "content": f"📬 Found {len(emails)} emails"}
        await asyncio.sleep(0.3)
        
        # Convert to serializable format
        yield {"type": "progress", "content": "🔄 Processing email data..."}
        await asyncio.sleep(0.5)
        
        # Upload to Pinecone (existing functionality)
        yield {"type": "progress", "content": "🗂️ Storing emails in vector database..."}
        await asyncio.sleep(1.0)
        
        from ..rag.email_vectorizer import EmailVectorizer
        try:
            vectorizer = EmailVectorizer()
            vectorizer_results = vectorizer.process_and_store_emails(emails)
            yield {"type": "progress", "content": "✅ Successfully stored emails in Pinecone vector database"}
        except Exception as vec_error:
            logger.error(f"Vector upload failed: {str(vec_error)}")
            yield {"type": "progress", "content": f"❌ Vector database storage failed: {str(vec_error)}"}
        
        yield {"type": "progress", "content": "🎉 Gmail data fetch completed!"}
        
    except Exception as e:
        logger.error(f"Error in streaming fetch_gmail_data: {str(e)}")
        yield {"type": "progress", "content": f"❌ Gmail fetch failed: {str(e)}"}

async def _detect_page_type_stream(session: ChatSession, file_id: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Detect page type with real-time streaming progress updates"""
    try:
        yield {"type": "progress", "content": "🔍 Starting page type detection..."}
        await asyncio.sleep(0.2)
        
        # Get file path from session
        if file_id not in session.uploaded_files:
            yield {"type": "error", "content": f"File {file_id} not found in session"}
            return
        
        file_info = session.uploaded_files[file_id]
        file_path = file_info["path"]
        
        yield {"type": "progress", "content": f"📄 Analyzing image: {file_info['filename']}"}
        await asyncio.sleep(0.3)
        
        # Use existing page type detector
        yield {"type": "progress", "content": "🤖 Using AI to detect page layout and type..."}
        await asyncio.sleep(1.0)
        
        detector = PageTypeDetector()
        result = detector.detect_page_type(file_path)
        
        yield {"type": "progress", "content": f"✅ Detected page type: {result.page_type.value}"}
        
    except Exception as e:
        logger.error(f"Error in streaming detect_page_type: {str(e)}")
        yield {"type": "progress", "content": f"❌ Page type detection failed: {str(e)}"}

async def _detect_page_type(session: ChatSession, file_id: str) -> Dict[str, Any]:
    """Detect page type from uploaded image with progress updates"""
    try:
        progress_updates = []
        progress_updates.append("🔍 Starting page type detection...")
        
        # Get file path from session
        if file_id not in session.uploaded_files:
            return {"error": f"File {file_id} not found in session"}
        
        file_info = session.uploaded_files[file_id]
        file_path = file_info["path"]
        
        progress_updates.append(f"📄 Analyzing image: {file_info['filename']}")
        
        # Use existing page type detector
        progress_updates.append("🤖 Using AI to detect page layout and type...")
        detector = PageTypeDetector()
        result = detector.detect_page_type(file_path)
        
        progress_updates.append(f"✅ Detected page type: {result.page_type.value}")
        progress_updates.append(f"📋 Detection reasoning: {result.reasoning}")
        
        return {
            "success": True,
            "page_type": result.page_type.value,
            "reasoning": result.reasoning,
            "visual_indicators": result.visual_indicators,
            "filename": file_info["filename"],
            "progress": progress_updates
        }
        
    except Exception as e:
        logger.error(f"Error in detect_page_type: {str(e)}")
        progress_updates.append(f"❌ Page type detection failed: {str(e)}")
        return {
            "error": str(e),
            "progress": progress_updates
        }




async def _process_edited_item(session: ChatSession, item_text: str) -> Dict[str, Any]:
    """Process a user-edited or accepted low-confidence item"""
    try:
        progress_updates = []
        
        # Get current review state
        review_state = session.get_low_confidence_review_state()
        if not review_state or not review_state.get('active'):
            return {
                "error": "No active review session found"
            }
        
        current_index = review_state['current_index']
        current_item = review_state['items'][current_index]
        
        # Check if user edited the text or kept it as-is
        original_text = current_item['text']
        user_edited = item_text.strip() != original_text.strip()
        
        if user_edited:
            progress_updates.append(f"✏️ Item updated to: {item_text}")
            # Create edited item with high confidence since user reviewed it
            edited_item = current_item.copy()
            edited_item['text'] = item_text.strip()
            edited_item['confidence'] = 1.0
            review_state['reviewed_items'].append(edited_item)
        else:
            progress_updates.append("✅ Item accepted as-is")
            # Keep original item
            review_state['reviewed_items'].append(current_item)
        
        # Move to next item
        old_index = review_state['current_index']
        review_state['current_index'] += 1
        logger.info(f"Review progress: moved from item {old_index + 1} to item {review_state['current_index'] + 1} of {len(review_state['items'])}")
        
        # Check if we're done with all items
        if review_state['current_index'] >= len(review_state['items']):
            review_state['active'] = False
            progress_updates.append("🎉 All items reviewed! Ready for Todoist upload.")
            
            # Clear pending review data since review is complete
            if 'pending_review' in session.processing_states:
                del session.processing_states['pending_review']
                logger.info("Cleared pending_review data after completing review")
            
            return {
                "success": True,
                "message": "Review completed",
                "review_complete": True,
                "reviewed_items_count": len(review_state['reviewed_items']),
                "progress": progress_updates
            }
        else:
            # More items to review - send next item for prefill
            next_index = review_state['current_index']
            next_item = review_state['items'][next_index]
            
            progress_updates.append(f"➡️ Item {next_index + 1} of {len(review_state['items'])}")
            
            return {
                "success": True,
                "message": f"Item processed. Continuing with item {next_index + 1}:",
                "review_complete": False,
                "next_prefill_item": {
                    "text": next_item["text"],
                    "confidence": next_item["confidence"],
                    "section": next_item["section"],
                    "index": next_index + 1
                },
                "total_items": len(review_state['items']),
                "progress": progress_updates
            }
        
    except Exception as e:
        logger.error(f"Error processing edited item: {str(e)}")
        return {
            "error": str(e)
        }

async def _process_edited_item_stream(session: ChatSession, item_text: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Process edited item with streaming progress"""
    try:
        yield {"type": "progress", "content": "📝 Processing item..."}
        await asyncio.sleep(0.2)
        
        # Get current review state
        review_state = session.get_low_confidence_review_state()
        if not review_state or not review_state.get('active'):
            yield {"type": "progress", "content": "❌ No active review session found"}
            return
        
        current_index = review_state['current_index']
        current_item = review_state['items'][current_index]
        
        # Check if user edited the text
        original_text = current_item['text']
        user_edited = item_text.strip() != original_text.strip()
        
        if user_edited:
            yield {"type": "progress", "content": "✏️ Item updated"}
            edited_item = current_item.copy()
            edited_item['text'] = item_text.strip()
            edited_item['confidence'] = 1.0
            # Preserve the original_item reference for OCR data updating
            edited_item['original_item'] = current_item['original_item']
            review_state['reviewed_items'].append(edited_item)
            
            # Debug logging
            logger.info(f"EDITED ITEM - Original: '{original_text}' -> New: '{item_text.strip()}'")
            logger.info(f"Original item keys: {list(current_item['original_item'].keys())}")
        else:
            yield {"type": "progress", "content": "✅ Item accepted as-is"}
            review_state['reviewed_items'].append(current_item)
            logger.info(f"ACCEPTED ITEM - Text: '{original_text}'")
        
        await asyncio.sleep(0.3)
        
        # Move to next item
        review_state['current_index'] += 1
        
        # Check if we're done
        if review_state['current_index'] >= len(review_state['items']):
            review_state['active'] = False
            yield {"type": "progress", "content": "🎉 All items reviewed! Ready for Todoist upload."}
            
            # Clear pending review data since review is complete
            if 'pending_review' in session.processing_states:
                del session.processing_states['pending_review']
            
            # Yield completion result
            yield {
                "type": "function_result",
                "data": {
                    "success": True,
                    "message": f"All {len(review_state['reviewed_items'])} items reviewed! Ready to proceed with Todoist upload.",
                    "review_complete": True,
                    "reviewed_items_count": len(review_state['reviewed_items'])
                }
            }
        else:
            next_index = review_state['current_index']
            next_item = review_state['items'][next_index]
            yield {"type": "progress", "content": f"➡️ Item {next_index + 1} of {len(review_state['items'])}"}
            
            # Yield continuation result for next prefill
            yield {
                "type": "function_result", 
                "data": {
                    "success": True,
                    "message": f"Item processed. Continuing with item {next_index + 1}:",
                    "review_complete": False,
                    "next_prefill_item": {
                        "text": next_item["text"],
                        "confidence": next_item["confidence"],
                        "section": next_item["section"],
                        "index": next_index + 1
                    },
                    "total_items": len(review_state['items'])
                }
            }
        
    except Exception as e:
        logger.error(f"Error in process edited item stream: {str(e)}")
        yield {"type": "progress", "content": f"❌ Processing failed: {str(e)}"}

async def _start_review_from_session(session: ChatSession) -> Dict[str, Any]:
    """Start review using low-confidence items stored in session"""
    try:
        progress_updates = []
        progress_updates.append("🔍 Starting low-confidence item review...")
        
        # Get stored review data from session
        pending_review = session.processing_states.get('pending_review')
        if not pending_review or not pending_review.get('has_items'):
            return {
                "success": False,
                "error": "No low-confidence items found in recent processing",
                "progress": progress_updates
            }
        
        items = pending_review.get('low_confidence_items', [])
        ocr_data = pending_review.get('ocr_data', {})
        
        if not items:
            progress_updates.append("✅ No low-confidence items to review")
            return {
                "success": True,
                "message": "No low-confidence items found",
                "progress": progress_updates
            }
        
        # Set up review state in session
        session.set_low_confidence_review_state(items, ocr_data)
        
        # Get first item to review
        first_item = items[0]
        progress_updates.append(f"📝 Ready to review {len(items)} low-confidence items")
        
        return {
            "success": True,
            "message": f"I found {len(items)} low-confidence items that need review. For each of the following items, make any necessary edits, if any, then press Enter to accept.",
            "total_items": len(items),
            "start_prefill_editing": True,
            "current_item": {
                "text": first_item["text"],
                "confidence": first_item["confidence"],
                "section": first_item["section"],
                "index": 1
            },
            "progress": progress_updates
        }
        
    except Exception as e:
        logger.error(f"Error starting review from session: {str(e)}")
        return {
            "error": str(e)
        }

async def _start_review_from_session_stream(session: ChatSession) -> AsyncGenerator[Dict[str, Any], None]:
    """Start review from session with streaming progress"""
    try:
        yield {"type": "progress", "content": "🔍 Starting low-confidence item review..."}
        await asyncio.sleep(0.2)
        
        # Get stored review data from session
        pending_review = session.processing_states.get('pending_review')
        if not pending_review or not pending_review.get('has_items'):
            yield {"type": "progress", "content": "❌ No low-confidence items found in recent processing"}
            return
        
        items = pending_review.get('low_confidence_items', [])
        ocr_data = pending_review.get('ocr_data', {})
        
        if not items:
            yield {"type": "progress", "content": "✅ No low-confidence items to review"}
            return
        
        # Set up review state in session
        session.set_low_confidence_review_state(items, ocr_data)
        
        yield {"type": "progress", "content": f"📝 Ready to review {len(items)} low-confidence items"}
        
        # Yield the final result data for chat service to handle prefill
        first_item = items[0]
        yield {
            "type": "function_result",
            "data": {
                "success": True,
                "message": f"I found {len(items)} low-confidence items that need review. For each of the following items, make any necessary edits, if any, then press Enter to accept.",
                "total_items": len(items),
                "start_prefill_editing": True,
                "current_item": {
                    "text": first_item["text"],
                    "confidence": first_item["confidence"],
                    "section": first_item["section"],
                    "index": 1
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error starting review from session stream: {str(e)}")
        yield {"type": "progress", "content": f"❌ Review start failed: {str(e)}"}