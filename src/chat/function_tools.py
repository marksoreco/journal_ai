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
            "name": "route_user_intent",
            "description": "Route user request to appropriate specialized agent (journal, gmail, or search)",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_message": {
                        "type": "string",
                        "description": "The user's message to analyze for intent"
                    },
                    "session_context": {
                        "type": "string", 
                        "description": "Current session context for routing decisions"
                    }
                },
                "required": ["user_message"]
            }
        },
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
            "name": "upload_journal_to_pinecone",
            "description": "Upload processed journal page OCR data to Pinecone vector database as separate section chunks for future search",
            "parameters": {
                "type": "object",
                "properties": {
                    "ocr_data": {
                        "type": "string",
                        "description": "JSON string of OCR data from journal processing"
                    }
                },
                "required": ["ocr_data"]
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
        if function_name == "route_user_intent":
            async for progress_data in _route_user_intent_stream(session, **function_args):
                yield progress_data
        elif function_name == "process_journal_image":
            async for progress_data in _process_journal_image_stream(session, **function_args):
                yield progress_data
        elif function_name == "upload_journal_to_pinecone":
            logger.info(f"Dispatching to upload_journal_to_pinecone with args: {function_args}")
            async for progress_data in _upload_journal_to_pinecone_stream(session, **function_args):
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

#         return {
#             "success": True,
#             "message": "Journal page processed successfully",
#             "page_type": detected_page_type,
#             "ocr_data": ocr_result,
#             "formatted_content": formatted_content,
#             "confidence_threshold": TASK_CONFIDENCE_THRESHOLD,
#             "low_confidence_items": low_confidence_items,
#             "progress": progress_updates,
#             "filename": file_info["filename"]
#         }
            

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
        
            if isinstance(ocr_data, dict):
                for key, value in ocr_data.items():
                    if isinstance(value, list):
                        logger.info(f"  {key}: {len(value)} items")
                    else:
                        logger.info(f"  {key}: {type(value)} - {str(value)[:100]}...")
            
            # Store results in session for potential low-confidence review
            try:
                if low_confidence_items and len(low_confidence_items) > 0:
                    session.processing_states['pending_review'] = {
                        'low_confidence_items': low_confidence_items,
                        'ocr_data': ocr_data,
                        'has_items': True,
                        'item_count': len(low_confidence_items),
                        'page_type': detected_page_type
                    }
                    logger.info(f"Stored {len(low_confidence_items)} low-confidence items for potential review")
                else:
                    session.processing_states['pending_review'] = {
                        'has_items': False,
                        'item_count': 0,
                        'page_type': detected_page_type
                    }
                    logger.info("No low-confidence items found - review not needed")
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

async def _upload_to_todoist_stream(session: ChatSession, task_data: str, review_low_confidence: bool = True) -> AsyncGenerator[Dict[str, Any], None]:
    """Upload tasks to Todoist with real-time streaming progress updates"""
    try:
        yield {"type": "progress", "content": "📋 Starting Todoist upload process..."}
        await asyncio.sleep(0.2)
        
        # Check if there's final OCR data from Pinecone storage
        final_ocr_data = session.processing_states.get('final_ocr_data')
        
        # Check if there are reviewed items from a completed review session
        review_state = session.get_low_confidence_review_state()
        has_reviewed_items = (review_state and 
                             not review_state.get('active') and 
                             review_state.get('reviewed_items'))
        
        # Parse task data - prefer final OCR data from Pinecone storage
        yield {"type": "progress", "content": "🔍 Parsing task data..."}
        await asyncio.sleep(0.3)
        
        if final_ocr_data:
            parsed_data = final_ocr_data
            yield {"type": "progress", "content": "📝 Using final OCR data from Pinecone storage"}
        elif isinstance(task_data, str):
            parsed_data = json.loads(task_data)
        else:
            parsed_data = task_data
        
        # Apply reviewed items if available (this would update the final OCR data)
        if has_reviewed_items:
            yield {"type": "progress", "content": "📝 Applying reviewed low-confidence items..."}
            await asyncio.sleep(0.3)
            original_ocr_data = review_state['ocr_data']
            reviewed_items = review_state['reviewed_items']
            parsed_data = _apply_reviewed_items_to_ocr_data(original_ocr_data, reviewed_items)
            yield {"type": "progress", "content": f"✅ Applied {len(reviewed_items)} reviewed items"}
            await asyncio.sleep(0.2)
            
            # Update the final OCR data with reviewed items
            session.processing_states['final_ocr_data'] = parsed_data
            
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

async def _start_review_from_session_stream(session: ChatSession) -> AsyncGenerator[Dict[str, Any], None]:
    """Start review from session with streaming progress"""
    try:
        yield {"type": "progress", "content": "🔍 Starting low-confidence item review..."}
        await asyncio.sleep(0.2)
        
        # Get stored review data from session
        logger.debug(f"Session processing states keys: {list(session.processing_states.keys())}")
        pending_review = session.processing_states.get('pending_review')
        logger.debug(f"Pending review data: {pending_review}")
        
        if not pending_review:
            logger.warning("No pending_review data found in session")
            yield {"type": "progress", "content": "❌ No pending review data found in session"}
            return
            
        has_items = pending_review.get('has_items', False)
        items = pending_review.get('low_confidence_items', [])

        # Validate consistency between has_items flag and actual items
        if has_items and not items:
            logger.warning("has_items=True but low_confidence_items is empty - correcting flag")
            pending_review['has_items'] = False
            has_items = False
        elif not has_items and items:
            logger.warning("has_items=False but low_confidence_items has content - correcting flag")  
            pending_review['has_items'] = True
            has_items = True

        logger.debug(f"Has items flag: {has_items}, Actual items count: {len(items)}")
        
        if not has_items:
            yield {"type": "progress", "content": "❌ No low-confidence items found in recent processing"}
            return
        
        ocr_data = pending_review.get('ocr_data', {})
        
        logger.debug(f"Found {len(items)} low-confidence items in session")
        logger.debug(f"Items: {items}")
        
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

async def _route_user_intent_stream(session: ChatSession, user_message: str, session_context: str = None) -> AsyncGenerator[Dict[str, Any], None]:
    """Route user intent to appropriate agent with streaming progress"""
    try:
        yield {"type": "progress", "content": "🤖 Analyzing your request..."}
        
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Simple intent classification using gpt-3.5-turbo
        system_prompt = """You are an intent router. Classify user messages into exactly one category:
- "journal": Processing journal pages, OCR, tasks, Todoist uploads
- "gmail": Email downloads, Gmail API, email processing  
- "search": Searching data, RAG queries, information retrieval
- "general": Greetings, help, other general conversation

Respond with only JSON: {"intent": "category", "confidence": 0.9}"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        result = json.loads(response.choices[0].message.content)
        intent = result.get("intent", "general")
        
        yield {"type": "progress", "content": f"📋 Intent detected: {intent}"}
        
        # Store routing decision in session
        session.processing_states["current_agent"] = intent
        
        yield {
            "type": "function_result", 
            "data": {
                "success": True,
                "intent": intent,
                "confidence": result.get("confidence", 0.8),
                "next_agent": intent,
                "message": f"Routing to {intent} agent..."
            }
        }
        
    except Exception as e:
        logger.error(f"Error in routing: {str(e)}")
        yield {"type": "progress", "content": f"❌ Routing failed: {str(e)}"}

def _detect_page_type_from_data(ocr_dict: Dict[str, Any]) -> str:
    """Detect page type from OCR data structure"""
    logger.info(f"Detecting page type from OCR data. Keys: {list(ocr_dict.keys())}")
    
    if ocr_dict.get('page_type'):
        page_type = ocr_dict['page_type'].lower()
        logger.info(f"Found page_type in data: {page_type}")
        return page_type
    elif 'month' in ocr_dict:
        logger.info("Found 'month' key, detecting as monthly")
        return "monthly"
    elif 'week_of' in ocr_dict or 'week' in ocr_dict or 'Weekly Priorities' in ocr_dict:
        logger.info("Found 'week' key, detecting as weekly")
        return "weekly" 
    elif 'date' in ocr_dict:
        logger.info("Found 'date' key, detecting as daily")
        return "daily"
    elif 'items' in ocr_dict and len(ocr_dict.keys()) == 1:
        # This is a simplified OCR structure - assume daily for now
        logger.info("Found simplified 'items' structure, assuming daily page")
        return "daily"
    else:
        logger.warning(f"No page type indicators found. Available keys: {list(ocr_dict.keys())}")
        return "unknown"

def _extract_date_from_data(ocr_dict: Dict[str, Any]) -> str:
    """Extract date from OCR data based on page type"""
    page_type = _detect_page_type_from_data(ocr_dict)
    logger.info(f"Extracting date for page type: {page_type}")
    
    if page_type == "monthly" and ocr_dict.get('month'):
        date_str = str(ocr_dict['month'])
        logger.info(f"Found monthly date: {date_str}")
    elif page_type == "weekly" and ocr_dict.get('week_of'):
        date_str = str(ocr_dict['week_of'])
        logger.info(f"Found weekly date (week_of): {date_str}")
    elif page_type == "weekly" and ocr_dict.get('week'):
        date_str = str(ocr_dict['week'])
        logger.info(f"Found weekly date (week): {date_str}")
    elif ocr_dict.get('date'):
        if isinstance(ocr_dict['date'], dict):
            date_str = ocr_dict['date'].get('value', 'unknown')
            logger.info(f"Found daily date (dict): {date_str}")
        else:
            date_str = str(ocr_dict['date'])
            logger.info(f"Found daily date (string): {date_str}")
    elif page_type == "daily" and 'items' in ocr_dict:
        # For simplified OCR structure, use current date as fallback
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Using current date for simplified OCR structure: {date_str}")
    else:
        logger.warning(f"No date found for page type {page_type}. Available keys: {list(ocr_dict.keys())}")
        date_str = "unknown"
    
    # Clean up date string
    date_str = date_str.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    logger.info(f"Final cleaned date: {date_str}")
    return date_str

def _create_section_chunks(ocr_dict: Dict[str, Any], page_type: str, date_str: str) -> List[Dict[str, Any]]:
    """Create section chunks from OCR data"""
    chunks = []
    logger.info(f"Creating section chunks for page type: {page_type}, date: {date_str}")
    logger.info(f"OCR dict keys: {list(ocr_dict.keys())}")
    
    for section_name, section_items in ocr_dict.items():
        logger.info(f"Processing section: {section_name}, type: {type(section_items)}")
        
        # Handle different data types for section_items
        if isinstance(section_items, list):
            # List of items - process each item
            logger.info(f"Section {section_name} has {len(section_items)} items")
            
            # Extract text from each item
            section_texts = []
            for i, item in enumerate(section_items):
                if isinstance(item, dict):
                    # Handle different field names for text content
                    text = item.get('text') or item.get('task') or item.get('item') or item.get('value', str(item))
                    logger.info(f"Item {i}: {text[:50]}...")
                else:
                    text = str(item)
                    logger.info(f"Item {i}: {text[:50]}...")
                section_texts.append(text)
            
            # Concatenate into single text
            section_content = ", ".join(section_texts)
            items_count = len(section_items)
            
        elif isinstance(section_items, str):
            # String value - use directly
            logger.info(f"Section {section_name} is a string: {section_items[:100]}...")
            section_content = section_items
            items_count = 1
            
        elif isinstance(section_items, dict):
            # Dictionary value - convert to string
            logger.info(f"Section {section_name} is a dict: {str(section_items)[:100]}...")
            section_content = str(section_items)
            items_count = 1
            
        else:
            # Other types - convert to string
            logger.info(f"Section {section_name} is {type(section_items)}: {str(section_items)[:100]}...")
            section_content = str(section_items)
            items_count = 1
        
        # Skip empty or whitespace-only content
        if not section_content or not section_content.strip():
            logger.info(f"Skipping section {section_name} - empty content")
            continue
        
        # Generate unique ID
        section_id = _generate_section_id(date_str, page_type, section_name)
        
        chunk = {
            "id": section_id,
            "content": section_content,
            "metadata": {
                "page_type": page_type,
                "date": date_str,
                "section_name": section_name,
                "items_count": items_count,
                "content_type": "journal_section",
                "chunk_text": section_content,
                "upload_timestamp": datetime.now().isoformat()
            }
        }
        chunks.append(chunk)
        logger.info(f"Created chunk for section {section_name} with ID: {section_id}")
    
    logger.info(f"Created {len(chunks)} total chunks")
    return chunks

def _generate_section_id(date: str, page_type: str, section_name: str) -> str:
    """Generate deterministic ID for section"""
    import hashlib
    id_string = f"{date}_{page_type}_{section_name}"
    return hashlib.md5(id_string.encode()).hexdigest()[:16]

def _store_chunk_in_pinecone(chunk: Dict[str, Any]) -> bool:
    """Store single chunk in Pinecone with sparse embeddings"""
    try:
        from pinecone import Pinecone, ServerlessSpec
        import re
        from collections import Counter
        
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index_name = "journal-sparse-index"
        
        # Create index if needed
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=10000,  # Sparse dimension
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        index = pc.Index(index_name)
        
        # Generate sparse embedding from text content
        text_content = chunk['content']
        sparse_vector = _create_sparse_embedding(text_content)
        
        # Upsert to Pinecone
        index.upsert([{
            "id": chunk['id'],
            "values": sparse_vector,
            "metadata": chunk['metadata']
        }])
        
        logger.info(f"Successfully stored chunk {chunk['id']} for section {chunk['metadata']['section_name']} with sparse embedding")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store chunk {chunk['id']}: {str(e)}")
        return False

def _create_sparse_embedding(text: str) -> List[float]:
    """Create sparse embedding from text using TF-IDF-like approach"""
    import re
    from collections import Counter
    import hashlib
    
    # Clean and tokenize text
    text = text.lower()
    # Remove punctuation but keep important separators
    text = re.sub(r'[^\w\s\-]', ' ', text)
    # Split into words
    words = text.split()
    
    # Create word frequency counter
    word_counts = Counter(words)
    
    # Create sparse vector
    sparse_dimension = 10000
    sparse_vector = [0.0] * sparse_dimension
    
    # Hash each word to a position and set frequency-based value
    for word, count in word_counts.items():
        # Hash word to position
        hash_obj = hashlib.md5(word.encode())
        position = int(hash_obj.hexdigest(), 16) % sparse_dimension
        
        # Set value based on frequency (log-scaled to prevent dominance)
        import math
        value = math.log(1 + count) / 10.0  # Normalize to reasonable range
        sparse_vector[position] = value
    
    return sparse_vector

async def _upload_journal_to_pinecone_stream(session: ChatSession, ocr_data: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Upload journal OCR data to Pinecone as section chunks"""
    try:
        logger.info("Starting section-based journal to Pinecone upload")
        yield {"type": "progress", "content": "📊 Preparing journal data for section-based storage..."}
        
        # Get the full OCR data from session instead of using the parameter
        logger.debug(f"Session processing states keys: {list(session.processing_states.keys())}")
        pending_review = session.processing_states.get('pending_review')
        logger.debug(f"Pending review data: {pending_review}")
        
        if pending_review and pending_review.get('ocr_data'):
            ocr_dict = pending_review['ocr_data']
            logger.info(f"Using full OCR data from session with keys: {list(ocr_dict.keys())}")
        else:
            # Fallback to parameter if session data not available
            import json
            try:
                ocr_dict = json.loads(ocr_data)
                logger.info(f"Using OCR data from parameter with keys: {list(ocr_dict.keys())}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.debug(f"Problematic JSON: {ocr_data}")
                logger.debug(f"Error position: char {e.pos}, line {e.lineno}, column {e.colno}")
                # Try to fix common JSON issues
                try:
                    # Use ast.literal_eval as a fallback for malformed JSON
                    import ast
                    ocr_dict = ast.literal_eval(ocr_data)
                    logger.info("Successfully parsed using ast.literal_eval")
                except (ValueError, SyntaxError) as e2:
                    logger.error(f"ast.literal_eval also failed: {e2}")
                    # Last resort: try to manually fix the JSON
                    try:
                        # Remove problematic characters and try again
                        import re
                        # Remove or escape problematic quotes
                        fixed_ocr_data = re.sub(r"'([^']*)'", r'"\1"', ocr_data)
                        ocr_dict = json.loads(fixed_ocr_data)
                        logger.debug("Successfully parsed after manual JSON fixing")
                    except json.JSONDecodeError as e3:
                        logger.error(f"Manual JSON fixing also failed: {e3}")
                        # Try with explicit UTF-8 encoding
                        try:
                            ocr_dict = json.loads(ocr_data)
                            logger.debug("Successfully parsed with UTF-8 encoding")
                        except (json.JSONDecodeError, TypeError) as e4:
                            logger.error(f"UTF-8 parsing also failed: {e4}")
                            # Last resort: try to clean Unicode characters
                            try:
                                # Replace problematic Unicode characters with boolean values
                                cleaned_ocr_data = ocr_data.replace('✅', 'true').replace('⬜', 'false')
                                cleaned_ocr_data = cleaned_ocr_data.replace('✔️', 'true').replace('❌', 'false')
                                cleaned_ocr_data = cleaned_ocr_data.replace('✓', 'true').replace('✗', 'false')
                                cleaned_ocr_data = cleaned_ocr_data.replace('☑', 'true').replace('☐', 'false')
                                cleaned_ocr_data = cleaned_ocr_data.replace('✔', 'true')
                                ocr_dict = json.loads(cleaned_ocr_data)
                                logger.debug("Successfully parsed after Unicode character replacement with booleans")
                            except json.JSONDecodeError as e5:
                                logger.error(f"Unicode replacement also failed: {e5}")
                                # Final attempt: try to manually fix the specific error location
                                try:
                                    logger.debug(f"Attempting manual fix at position {e5.pos}")
                                    # Get the character at the error position
                                    if e5.pos < len(ocr_data):
                                        problem_char = ocr_data[e5.pos]
                                        logger.debug(f"Problem character at position {e5.pos}: '{problem_char}' (ord: {ord(problem_char)})")
                                        # Try to replace the problematic character
                                        fixed_ocr_data = ocr_data[:e5.pos] + '"' + ocr_data[e5.pos+1:]
                                        ocr_dict = json.loads(fixed_ocr_data)
                                        logger.debug("Successfully parsed after manual character replacement")
                                    else:
                                        logger.debug(f"Error position {e5.pos} is beyond string length {len(ocr_data)}")
                                        # Try to add missing closing braces
                                        try:
                                            logger.debug("Attempting to add missing closing braces")
                                            # Count opening and closing braces
                                            open_braces = ocr_data.count('{')
                                            close_braces = ocr_data.count('}')
                                            missing_braces = open_braces - close_braces
                                            logger.debug(f"Open braces: {open_braces}, Close braces: {close_braces}, Missing: {missing_braces}")
                                            
                                            if missing_braces > 0:
                                                fixed_ocr_data = ocr_data + '}' * missing_braces
                                                ocr_dict = json.loads(fixed_ocr_data)
                                                logger.debug(f"Successfully parsed after adding {missing_braces} closing braces")
                                            else:
                                                raise Exception(f"JSON parsing failed at position {e5.pos}")
                                        except Exception as e7:
                                            logger.error(f"Adding closing braces also failed: {e7}")
                                            raise Exception(f"JSON parsing failed at position {e5.pos}")
                                except Exception as e6:
                                    logger.error(f"Manual character replacement also failed: {e6}")
                                    raise Exception(f"Failed to parse OCR data as JSON after all attempts: {e}")
            logger.warning("Session OCR data not found - using simplified parameter data")
        
        from datetime import datetime
        
        # Extract page info
        page_type = _detect_page_type_from_data(ocr_dict)
        date_str = _extract_date_from_data(ocr_dict)
        
        yield {"type": "progress", "content": f"🔍 Detected {page_type} page from {date_str}"}
        
        # Create section chunks
        yield {"type": "progress", "content": "📝 Creating section chunks..."}
        section_chunks = _create_section_chunks(ocr_dict, page_type, date_str)
        
        yield {"type": "progress", "content": f"📋 Created {len(section_chunks)} sections"}
        
        # Store each section in Pinecone
        yield {"type": "progress", "content": "🗃️ Storing sections in Pinecone..."}
        
        stored_sections = []
        for chunk in section_chunks:
            section_name = chunk['metadata']['section_name']
            yield {"type": "progress", "content": f"💾 Storing {section_name} section..."}
            
            # Store chunk in Pinecone
            success = _store_chunk_in_pinecone(chunk)
            if success:
                stored_sections.append(section_name)
                yield {"type": "progress", "content": f"✅ Stored {section_name} section"}
            else:
                yield {"type": "progress", "content": f"❌ Failed to store {section_name} section"}
        
        yield {"type": "progress", "content": f"✅ Successfully stored {len(stored_sections)} sections"}
        
        # Store final OCR data in session (Todoist upload is a separate optional step)
        session.processing_states['final_ocr_data'] = ocr_dict
        
        yield {
            "type": "function_result",
            "data": {
                "success": True,
                "total_sections": len(section_chunks),
                "page_type": page_type,
                "date": date_str
            }
        }
        
    except Exception as e:
        logger.error(f"Error in section-based Pinecone upload: {str(e)}")
        yield {"type": "progress", "content": f"❌ Storage failed: {str(e)}"}