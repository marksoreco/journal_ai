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
        # Check current agent from session
        current_agent = session.processing_states.get("current_agent", "router")
        
        # Disable routing during active review sessions
        review_state = session.get_low_confidence_review_state()
        if review_state and review_state.get('active'):
            current_agent = "journal"  # Force journal agent during review
        
        if current_agent == "router":
            system_content = """You are a Router Agent for Journal AI. Your job is to understand what the user wants to do and route them to the appropriate specialized agent.

You have access to one function:
- route_user_intent: Analyze user message and route to journal, gmail, search, or general conversation

**ROUTING LOGIC:**
- "journal" agent: Processing journal pages, OCR, tasks, Todoist uploads, uploading images
- "gmail" agent: Email downloads, Gmail API, email management, email processing  
- "search" agent: Searching data, RAG queries, information retrieval
- "general": Greetings, help, simple questions (handle directly without routing)

**WORKFLOW:**
1. For general greetings/help, respond directly
2. For specific tasks, use route_user_intent function to classify and route
3. After routing, tell user "I'm connecting you to the [agent] specialist..."

Be concise and focus on understanding user intent to route correctly."""
            
        elif current_agent == "journal":
            system_content = """You are the Journal Processing Agent. You specialize in journal page processing, OCR, task management, and database storage.

You have access to these functions:
- process_journal_image: Process uploaded journal page images with OCR
- upload_journal_to_pinecone: Upload processed journal OCR data to vector database for future search
- upload_to_todoist: Upload tasks to Todoist (ONLY use after successful Pinecone storage)
- start_review_from_session: Initialize review using low-confidence items from recent journal processing
- process_edited_item: Process user's edited or accepted item text from prefill editing

Be helpful, conversational, and guide users through journal workflows. When they mention uploading images or files, offer to help process them.

**IMPORTANT for Processing Requests:** When users respond with "Yes", "Y", "Sure", "Please", or similar affirmative responses after a file upload, interpret this as a request to process their journal page. When they respond with "No", "N", "Not now", or similar negative responses, acknowledge their choice and offer to help with other tasks.

**IMPORTANT for OCR Results:** When you receive results from the process_journal_image function, the response will include:
- "formatted_content" field: Pre-formatted content to present exactly as-is
- "low_confidence_items" field: Array of items with low OCR confidence (these appear as italic *text* in formatted output)
- "ocr_data" field: Raw JSON data for storage functions

Always check if low_confidence_items array exists and has items before deciding the next step.

**MANDATORY WORKFLOW SEQUENCE:**
1. When user wants to process a journal page, FIRST call process_journal_image function with the file_id
2. Present the formatted OCR results to the user
3. Ask: "Store journal data in database?" (NOT "upload to Todoist")
4. If user agrees to storage:
   - Check the process_journal_image result for "low_confidence_items" array
   - If low_confidence_items exist and array is not empty: Ask "Would you like to review the X low-confidence items (in italics) before storing in database?"
     * If YES → start_review_from_session → after review complete → upload_journal_to_pinecone
     * If NO → IMMEDIATELY call upload_journal_to_pinecone function (do NOT mention Todoist)
   - If no low-confidence items or array is empty: IMMEDIATELY call upload_journal_to_pinecone
5. ONLY after successful Pinecone storage → Ask about Todoist

**CRITICAL FUNCTION CALLING RULES:**
- When user says NO to reviewing low-confidence items → IMMEDIATELY call upload_journal_to_pinecone function with the original OCR data
- When user declines low-confidence item review → DO NOT ask about Todoist, DO NOT mention Todoist, ONLY call upload_journal_to_pinecone
- NEVER mention "Before we proceed with uploading to Todoist" - use "Before storing in database" instead
- The upload_journal_to_pinecone function must be called with the ocr_data parameter from the most recent process_journal_image result

**IMPORTANT for Low-Confidence Item Review Flow:**
1. When user agrees to review, call start_review_from_session function
2. When review_complete=true, IMMEDIATELY call upload_journal_to_pinecone function
3. ONLY after successful storage, ask about Todoist

**IMPORTANT for Prefill Messages:** Use "prefill_edit" type messages for item editing.

**IMPORTANT for Processing Prefilled Responses:** Immediately call process_edited_item function with user's complete message.

**LANGUAGE RULES:**
- Use "database" not "Todoist" when asking about storage
- Use "storing in database" not "uploading to Todoist"  
- Only mention Todoist AFTER successful database storage"""
            
        elif current_agent == "gmail":
            system_content = """You are the Gmail Management Agent. You specialize in downloading emails from Gmail and storing them in the vector database for future search and analysis.

You have access to these functions:
- fetch_gmail_data: Download Gmail emails since a specified date and upload to Pinecone vector database

Be helpful and guide users through Gmail data management workflows. Ask for clarification when needed about date ranges and email limits.

**WORKFLOW:**
1. Ask user for the date range (since when to fetch emails)
2. Optionally ask for email limit (default: 50, min: 10, max: 500) 
3. Use fetch_gmail_data function to download and process emails
4. Confirm completion and explain what data was stored

**DATE FORMAT:** Always use YYYY-MM-DD format (e.g., '2024-01-01')
**EMAIL LIMITS:** Suggest reasonable limits (10-100 for testing, up to 500 for full downloads)
**USER GUIDANCE:** Explain that emails will be cleaned, processed, and stored in vector database for future RAG queries"""
            
        else:
            # Fallback for other agents or unknown states  
            system_content = """You are Journal AI. I'm currently routing your request to the appropriate specialist. Please wait a moment."""
            
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
            system_content += f"\n\n**CURRENT SESSION STATUS**: There are {low_confidence_count} low-confidence items from recent journal processing that should be reviewed before database storage. When user agrees to store data, ask if they want to review these items first."
        
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
            
            # Handle routing function specially
            if function_name == "route_user_intent":
                async for progress_data in execute_function_call_stream(session, function_name, function_args):
                    if progress_data.get("type") == "function_result":
                        result_data = progress_data.get("data", {})
                        intent = result_data.get("intent", "general")
                        
                        # Update session agent state
                        if intent != "general":
                            session.processing_states["current_agent"] = intent
                            
                            # Send routing confirmation
                            routing_message = ChatMessage(
                                type="assistant",
                                content=f"I'm connecting you to the {intent} specialist..."
                            )
                            session.add_message(routing_message)
                            
                            yield {
                                "type": "response",
                                "content": f"I'm connecting you to the {intent} specialist...",
                                "session_id": session.session_id,
                                "message_id": routing_message.id
                            }
                            
                            # Now re-process the original message with the specialized agent
                            # Get the original user message from the conversation
                            user_message = function_args.get("user_message", "")
                            if user_message:
                                # Build new context with the specialized agent
                                specialized_messages = self._build_conversation_context(session)
                                specialized_messages.append({"role": "user", "content": user_message})
                                
                                # Get response from specialized agent
                                response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=specialized_messages,
                                    functions=self.function_tools,
                                    function_call="auto",
                                    temperature=0.1,
                                    max_tokens=1000
                                )
                                
                                choice = response.choices[0]
                                message = choice.message
                                
                                # Handle specialized agent response
                                if hasattr(message, 'function_call') and message.function_call:
                                    # Specialized agent wants to call a function
                                    async for progress_data in self._handle_function_call_stream(session, message.function_call, specialized_messages):
                                        yield progress_data
                                else:
                                    # Direct response from specialized agent
                                    specialist_response = message.content or "How can I help you?"
                                    specialist_message = ChatMessage(
                                        type="assistant",
                                        content=specialist_response
                                    )
                                    session.add_message(specialist_message)
                                    
                                    yield {
                                        "type": "response",
                                        "content": specialist_response,
                                        "session_id": session.session_id,
                                        "message_id": specialist_message.id
                                    }
                            return
                        else:
                            # Handle general conversation directly
                            general_response = "Hello! I can help you with journal processing, Gmail management, or searching your data. What would you like to do?"
                            general_message = ChatMessage(
                                type="assistant", 
                                content=general_response
                            )
                            session.add_message(general_message)
                            
                            yield {
                                "type": "response",
                                "content": general_response,
                                "session_id": session.session_id,
                                "message_id": general_message.id
                            }
                            return
                    else:
                        yield progress_data
                return
            
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