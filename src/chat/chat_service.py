"""
Chat service with OpenAI integration and function calling
"""
import json
import logging
import os
from typing import Dict, List, Any, Optional, AsyncGenerator
from openai import OpenAI
from .models import ChatMessage, ChatSession
from .function_tools import get_function_tools, execute_function_call_stream

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o-mini"  # Using same model as the LangChain agent
        self.function_tools = get_function_tools()
        
    async def process_message(self, session: ChatSession, user_message: str) -> str:
        """Process a user message and return AI response"""
        try:
            # Build conversation history for context
            messages = self._build_conversation_context(session)
            
            # Add current user message
            messages.append({
                "role": "user", 
                "content": user_message
            })
            
            # Make OpenAI API call with function calling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=self.function_tools,
                function_call="auto",
                temperature=0.1,
                max_tokens=1000
            )
            
            choice = response.choices[0]
            message = choice.message
            
            # Handle function calls
            if hasattr(message, 'function_call') and message.function_call:
                return await self._handle_function_call(session, message.function_call, messages)
            else:
                # Regular text response
                return message.content or "I'm sorry, I couldn't generate a response."
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return f"I encountered an error: {str(e)}. Please try again."
    
    async def process_message_stream(self, session: ChatSession, user_message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a user message with streaming progress updates"""
        try:
            # Build conversation history for context
            messages = self._build_conversation_context(session)
            
            # Add current user message
            messages.append({
                "role": "user", 
                "content": user_message
            })
            
            # Make OpenAI API call with function calling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=self.function_tools,
                function_call="auto",
                temperature=0.1,
                max_tokens=1000
            )
            
            choice = response.choices[0]
            message = choice.message
            
            # Handle function calls with streaming
            if hasattr(message, 'function_call') and message.function_call:
                async for progress_data in self._handle_function_call_stream(session, message.function_call, messages):
                    yield progress_data
            else:
                # Regular text response
                ai_response = message.content or "I'm sorry, I couldn't generate a response."
                
                # Add AI response to session
                ai_message = ChatMessage(
                    type="assistant",
                    content=ai_response
                )
                session.add_message(ai_message)
                
                yield {
                    "type": "response",
                    "content": ai_response,
                    "session_id": session.session_id,
                    "message_id": ai_message.id
                }
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            yield {
                "type": "error",
                "content": f"I encountered an error: {str(e)}. Please try again."
            }
    
    def _build_conversation_context(self, session: ChatSession, max_messages: int = 10, force_fresh: bool = False) -> List[Dict]:
        """Build conversation context from session history"""
        
        # Build system message with session context
        system_content = """You are Journal AI, an assistant that helps users process journal pages, manage tasks, and access their data.

You have access to several functions:
- process_journal_image: Process uploaded journal page images with OCR
- upload_to_todoist: Upload tasks to Todoist with confidence review
- fetch_gmail_data: Fetch and store Gmail data in vector database
- detect_page_type: Detect the type of journal page
- start_review_from_session: Initialize review using low-confidence items from recent journal processing
- process_edited_item: Process user's edited or accepted item text from prefill editing

Be helpful, conversational, and guide users through these workflows. When they mention uploading images or files, offer to help process them. Ask clarifying questions when needed.

**IMPORTANT for Processing Requests:** When users respond with "Yes", "Y", "Sure", "Please", or similar affirmative responses after a file upload, interpret this as a request to process their journal page. When they respond with "No", "N", "Not now", or similar negative responses, acknowledge their choice and offer to help with other tasks.

**IMPORTANT for OCR Results:** When you receive results from the process_journal_image function, the response will include a "formatted_content" field that contains pre-formatted content using the exact same logic as the classic UI. Simply present this formatted_content as-is without any additional formatting or modification. This ensures consistent formatting that matches the classic UI exactly.

**IMPORTANT for Low-Confidence Items:** When you receive results from process_journal_image, check if there are any "low_confidence_items" in the response. These are items that were detected with low OCR confidence and appear as italics (*text*) in the formatted output. 

When responding to journal processing requests, structure your response in two parts:
1. First, present the pre-formatted content from the "formatted_content" field exactly as provided
2. Then, after a clear break (double line break), ask follow-up questions like "Would you like to upload any tasks to Todoist?" or suggest next steps

**IMPORTANT for Todoist Upload Flow:** 
1. After showing OCR results, FIRST ask: "Would you like to upload any tasks to Todoist?"
2. ONLY if user responds yes to Todoist upload, THEN check if there are low_confidence_items
3. If there are low-confidence items, ask: "Would you like to review low-confidence items (in italics) prior to upload?"
   - If user says yes, use start_review_from_session function to begin the review process
   - If user says no, proceed directly with the Todoist upload using the original data  
4. If there are no low-confidence items, proceed directly with the upload

NEVER ask about reviewing low-confidence items unless the user has first confirmed they want to upload to Todoist.

**IMPORTANT for Low-Confidence Item Review Flow:**
1. When user agrees to review, YOU MUST call the start_review_from_session function - DO NOT try to handle reviews manually
2. This function automatically accesses the stored low-confidence items from recent journal processing
3. The function will automatically handle the prefill editing flow
4. When user responds with item text, call process_edited_item function with their message as item_text
5. Continue until the function returns review_complete=true
6. When review_complete=true, IMMEDIATELY proceed with Todoist upload using upload_to_todoist function
7. NEVER manually ask "keep/edit/skip" - always use the function tools for reviews

**IMPORTANT for Prefill Messages:** When you need to prefill the user's input field with item text:
- Send a message with type "prefill_edit" 
- Include the item text to prefill and any context message
- The frontend will auto-populate the user's input field for easy editing

**IMPORTANT for Processing Prefilled Responses:** 
- After sending a prefill_edit message, the VERY NEXT user message should be treated as their final item text
- IMMEDIATELY call process_edited_item function with their complete message as the item_text
- Do NOT ask questions or make conversation - just process their response with the function
- The function will handle moving to the next item or completing the review
- NEVER respond conversationally during prefill editing - always use the function

This creates a better user experience by separating the factual results from interactive questions and allowing users to review uncertain OCR results before uploading to Todoist."""
        
        # Add information about uploaded files
        if session.uploaded_files:
            file_list = []
            for file_id, file_info in session.uploaded_files.items():
                file_list.append(f"- {file_info['filename']} (ID: {file_id})")
            
            system_content += f"\n\nCurrently uploaded files in this session:\n" + "\n".join(file_list)
            system_content += f"\n\nWhen the user asks to process files, you can use the file IDs above with the process_journal_image function."
        
        # Add information about pending low-confidence reviews
        pending_review = session.processing_states.get('pending_review')
        if pending_review and pending_review.get('has_items'):
            low_confidence_count = len(pending_review.get('low_confidence_items', []))
            system_content += f"\n\n**CURRENT SESSION STATUS**: There are {low_confidence_count} low-confidence items from recent journal processing that can be reviewed before Todoist upload. When user requests Todoist upload, ask if they want to review these items first."
        
        # Add information about active low-confidence review session
        review_state = session.get_low_confidence_review_state()
        if review_state and review_state.get('active'):
            current_index = review_state.get('current_index', 0)
            total_items = len(review_state.get('items', []))
            system_content += f"\n\n**ACTIVE REVIEW SESSION**: Currently reviewing low-confidence item {current_index + 1} of {total_items}. The next user message should be processed with process_edited_item function."
        
        messages = [
            {
                "role": "system",
                "content": system_content
            }
        ]
        
        # Add recent conversation history
        recent_messages = session.get_recent_messages(max_messages)
        for msg in recent_messages:
            if msg.type in ["user", "assistant"]:
                messages.append({
                    "role": msg.type,
                    "content": msg.content
                })
        
        return messages
    
    async def _handle_function_call(self, session: ChatSession, function_call, messages: List[Dict]) -> str:
        """Handle OpenAI function call"""
        try:
            function_name = function_call.name
            function_args = json.loads(function_call.arguments)
            
            logger.info(f"Executing function: {function_name} with args: {function_args}")
            
            # Non-streaming function calls are deprecated - redirect to streaming
            return "This operation requires streaming support. Please try again or refresh the page to enable real-time processing."
            
        except Exception as e:
            logger.error(f"Error handling function call: {str(e)}")
            return f"I had trouble executing that function: {str(e)}"
    
    async def _handle_function_call_stream(self, session: ChatSession, function_call, messages: List[Dict]) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle OpenAI function call with streaming progress"""
        try:
            function_name = function_call.name
            function_args = json.loads(function_call.arguments)
            
            logger.info(f"Executing function: {function_name} with args: {function_args}")
            
            # For prefill functions, collect the result data from streaming
            function_result = None
            if function_name in ["start_review_from_session", "process_edited_item"]:
                # Execute the function with streaming progress and capture result data
                async for progress_data in execute_function_call_stream(session, function_name, function_args):
                    if progress_data.get("type") == "function_result":
                        function_result = progress_data.get("data")
                    else:
                        yield progress_data
                
                # Handle prefill flow with captured result data
                if function_result:
                    if (function_name == "start_review_from_session") and function_result.get("start_prefill_editing"):
                        # Handle start of prefill editing flow
                        current_item = function_result.get("current_item", {})
                        
                        # Send the initial message
                        intro_message = ChatMessage(
                            type="assistant",
                            content=function_result.get("message", "Starting review...")
                        )
                        session.add_message(intro_message)
                        
                        yield {
                            "type": "response",
                            "content": function_result.get("message", "Starting review..."),
                            "session_id": session.session_id,
                            "message_id": intro_message.id
                        }
                        
                        # Send prefill message for first item
                        if current_item.get("text"):
                            prefill_message = ChatMessage(
                                type="assistant",
                                content=f"Item {current_item.get('index', 1)} of {function_result.get('total_items', '?')} (Confidence: {current_item.get('confidence', 0):.0%}):"
                            )
                            session.add_message(prefill_message)
                            
                            yield {
                                "type": "prefill_edit",
                                "content": f"Item {current_item.get('index', 1)} of {function_result.get('total_items', '?')} (Confidence: {current_item.get('confidence', 0):.0%}):",
                                "prefill_text": current_item.get("text", ""),
                                "session_id": session.session_id,
                                "message_id": prefill_message.id
                            }
                        return
                        
                    elif function_name == "process_edited_item" and function_result.get("next_prefill_item"):
                        # Handle continuation of prefill editing flow
                        next_item = function_result.get("next_prefill_item", {})
                        
                        # Send progress message
                        progress_message = ChatMessage(
                            type="assistant",
                            content=function_result.get("message", "Continuing...")
                        )
                        session.add_message(progress_message)
                        
                        yield {
                            "type": "response",
                            "content": function_result.get("message", "Continuing..."),
                            "session_id": session.session_id,
                            "message_id": progress_message.id
                        }
                        
                        # Send prefill message for next item
                        if next_item.get("text"):
                            prefill_message = ChatMessage(
                                type="assistant",
                                content=f"Item {next_item.get('index', 1)} of {function_result.get('total_items', '?')} (Confidence: {next_item.get('confidence', 0):.0%}):"
                            )
                            session.add_message(prefill_message)
                            
                            yield {
                                "type": "prefill_edit",
                                "content": f"Item {next_item.get('index', 1)} of {function_result.get('total_items', '?')} (Confidence: {next_item.get('confidence', 0):.0%}):",
                                "prefill_text": next_item.get("text", ""),
                                "session_id": session.session_id,
                                "message_id": prefill_message.id
                            }
                        return
                        
                    elif function_name == "process_edited_item" and function_result.get("review_complete"):
                        # Review completed - add the function result to conversation context and let AI continue
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "function_call": {
                                "name": function_name,
                                "arguments": function_call.arguments
                            }
                        })
                        
                        messages.append({
                            "role": "function",
                            "name": function_name,
                            "content": json.dumps(function_result)
                        })
                        
                        # Get AI response to decide next action (should trigger Todoist upload)
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            functions=self.function_tools,
                            function_call="auto",
                            temperature=0.1,
                            max_tokens=1000
                        )
                        
                        choice = response.choices[0]
                        message = choice.message
                        
                        # If AI wants to make another function call (Todoist upload), handle it
                        if hasattr(message, 'function_call') and message.function_call:
                            async for progress_data in self._handle_function_call_stream(session, message.function_call, messages):
                                yield progress_data
                        else:
                            # Regular response
                            ai_response = message.content or "Review completed!"
                            completion_message = ChatMessage(
                                type="assistant",
                                content=ai_response
                            )
                            session.add_message(completion_message)
                            
                            yield {
                                "type": "response",
                                "content": ai_response,
                                "session_id": session.session_id,
                                "message_id": completion_message.id
                            }
                        return
                
                return  # Prefill handling complete
            
            # Execute the function with streaming progress for non-prefill functions
            function_result = None
            async for progress_data in execute_function_call_stream(session, function_name, function_args):
                if progress_data.get("type") == "function_result":
                    function_result = progress_data.get("data")
                else:
                    yield progress_data
            
            # If no function result from streaming, create a default success result
            if function_result is None:
                function_result = {"success": True, "message": "Function completed successfully."}
            
            # Add function call and result to conversation context
            messages.append({
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_name,
                    "arguments": function_call.arguments
                }
            })
            
            messages.append({
                "role": "function",
                "name": function_name,
                "content": json.dumps(function_result)
            })
            
            # Get AI response based on function result
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            ai_response = response.choices[0].message.content or "Function completed successfully."
            
            # Check for special handling based on function and result
            if function_name == "process_journal_image":
                logger.info(f"Journal processing response to split: {repr(ai_response)}")
                # Split the response into results and follow-up question
                parts = ai_response.split("\n\n")
                logger.info(f"Split into {len(parts)} parts")
                results_parts = []
                question_parts = []
                
                current_section = results_parts
                for part in parts:
                    # If the part contains a question, switch to question section
                    question_keywords = [
                        "would you like", "what would", "do you want", "should i", "would you prefer",
                        "would you", "do you", "are you", "can i", "shall i", "next steps", "anything else",
                        "upload", "todoist"
                    ]
                    if any(keyword in part.lower() for keyword in question_keywords):
                        logger.info(f"Found question keywords in part: {repr(part)}")
                        current_section = question_parts
                    current_section.append(part)
                
                # Send results first
                if results_parts:
                    results_content = "\n\n".join(results_parts).strip()
                    logger.info(f"Sending results content: {repr(results_content)}")
                    results_message = ChatMessage(
                        type="assistant",
                        content=results_content
                    )
                    session.add_message(results_message)
                    
                    yield {
                        "type": "response",
                        "content": results_content,
                        "session_id": session.session_id,
                        "message_id": results_message.id
                    }
                
                # Send follow-up question as separate message if it exists
                if question_parts:
                    question_content = "\n\n".join(question_parts).strip()
                    logger.info(f"Sending question content: {repr(question_content)}")
                    if question_content:  # Only send if there's actual content
                        question_message = ChatMessage(
                            type="assistant", 
                            content=question_content
                        )
                        session.add_message(question_message)
                        
                        yield {
                            "type": "followup",
                            "content": question_content,
                            "session_id": session.session_id,
                            "message_id": question_message.id
                        }
            else:
                # Regular single response
                ai_message = ChatMessage(
                    type="assistant",
                    content=ai_response
                )
                session.add_message(ai_message)
                
                # Send final response
                yield {
                    "type": "response",
                    "content": ai_response,
                    "session_id": session.session_id,
                    "message_id": ai_message.id
                }
            
        except Exception as e:
            logger.error(f"Error handling function call: {str(e)}")
            yield {
                "type": "error",
                "content": f"I had trouble executing that function: {str(e)}"
            }

# Global chat service instance
chat_service = ChatService()