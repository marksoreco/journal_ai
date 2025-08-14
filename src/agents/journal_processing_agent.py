"""
LangChain Agent for Journal Processing Workflow

This agent handles the complete journal image processing pipeline:
1. Page type detection
2. User confirmation
3. OCR processing
4. Optional Todoist integration for daily pages
"""

import logging
import json
import os
from typing import Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel, Field

from ..agents.tools.page_detector import PageTypeDetector
from ..ocr.gpt4o_ocr import GPT4oOCRAdapter
from ..todoist.todoist_client import TodoistClient

# Configure logger
logger = logging.getLogger(__name__)

class PageDetectionInput(BaseModel):
    image_path: str = Field(description="Path to the journal page image")

class OCRProcessingInput(BaseModel):
    image_path: str = Field(description="Path to the journal page image")
    page_type: str = Field(description="Type of page: Daily, Weekly, or Monthly")

class TodoistUploadInput(BaseModel):
    ocr_data: str = Field(description="JSON string containing OCR extracted data")

class UserConfirmationInput(BaseModel):
    detected_page_type: str = Field(description="The detected page type")
    reasoning: str = Field(description="Reasoning for the detection")

def create_page_detection_tool():
    """Create the page type detection tool"""
    def detect_page_type(image_path: str) -> Dict[str, Any]:
        """Detect the type of journal page from an image"""
        try:
            detector = PageTypeDetector()
            result = detector.detect_page_type(image_path)
            
            return {
                "page_type": result.page_type.value,
                "reasoning": result.reasoning,
                "visual_indicators": result.visual_indicators,
                "success": True
            }
        except Exception as e:
            logger.error(f"Page detection failed: {str(e)}")
            return {
                "page_type": "Unknown",
                "reasoning": f"Detection failed: {str(e)}",
                "visual_indicators": {},
                "success": False
            }
    
    return StructuredTool.from_function(
        func=detect_page_type,
        name="detect_page_type",
        description="Detect the type of journal page (Daily, Weekly, Monthly) from an image",
        args_schema=PageDetectionInput
    )

def create_ocr_processing_tool():
    """Create the OCR processing tool"""
    def process_image_ocr(image_path: str, page_type: str) -> Dict[str, Any]:
        """Process journal page image with OCR based on page type"""
        try:
            ocr_adapter = GPT4oOCRAdapter()
            ocr_result = ocr_adapter.extract_text(image_path, page_type)
            
            return {
                "ocr_data": ocr_result,
                "page_type": page_type,
                "success": True
            }
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return {
                "ocr_data": f"OCR processing failed: {str(e)}",
                "page_type": page_type,
                "success": False
            }
    
    return StructuredTool.from_function(
        func=process_image_ocr,
        name="process_image_ocr",
        description="Process journal page image with OCR extraction based on page type",
        args_schema=OCRProcessingInput
    )

def create_todoist_upload_tool():
    """Create the Todoist upload tool"""
    def upload_to_todoist(ocr_data: str) -> Dict[str, Any]:
        """Upload tasks from OCR data to Todoist"""
        try:
            todoist_client = TodoistClient()
            
            # Parse OCR data if it's a string
            if isinstance(ocr_data, str):
                parsed_data = json.loads(ocr_data)
            else:
                parsed_data = ocr_data
                
            result = todoist_client.upload_tasks_from_ocr(parsed_data)
            
            return {
                "success": True,
                "message": result.get("message", "Tasks uploaded successfully"),
                "tasks_added": result.get("tasks_added", 0)
            }
        except Exception as e:
            logger.error(f"Todoist upload failed: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to upload to Todoist: {str(e)}",
                "tasks_added": 0
            }
    
    return StructuredTool.from_function(
        func=upload_to_todoist,
        name="upload_to_todoist",
        description="Upload tasks from OCR data to Todoist",
        args_schema=TodoistUploadInput
    )

def create_user_confirmation_tool():
    """Create a tool that handles user confirmation for page type"""
    def request_user_confirmation(detected_page_type: str, reasoning: str) -> Dict[str, Any]:
        """Request user confirmation for detected page type"""
        # For the initial implementation, we'll auto-confirm the detected type
        # This could be enhanced later to support interactive confirmation
        logger.info(f"Auto-confirming detected page type: {detected_page_type}")
        return {
            "confirmed_page_type": detected_page_type,
            "user_confirmed": True,
            "reasoning": reasoning,
            "confirmation_method": "auto"
        }
    
    return StructuredTool.from_function(
        func=request_user_confirmation,
        name="request_user_confirmation",
        description="Confirm the detected page type (auto-confirms for now)",
        args_schema=UserConfirmationInput
    )

class JournalProcessingAgent:
    """LangChain agent for journal processing workflow"""
    
    def __init__(self):
        """Initialize the journal processing agent"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Store results from processing
        self.last_page_type = None
        self.last_ocr_data = None
        
        # Create tools that can access this agent instance
        self.tools = [
            self._create_page_detection_tool(),
            self._create_user_confirmation_tool(),
            self._create_ocr_processing_tool(),
            create_todoist_upload_tool()
        ]
        
        # Create agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a journal processing assistant that processes journal pages automatically.

Your workflow must follow these steps in order:
1. DETECT: Use detect_page_type to identify the journal page type (Daily, Weekly, or Monthly)
2. CONFIRM: Use request_user_confirmation to confirm the detected page type (this auto-confirms)
3. PROCESS: Use process_image_ocr to extract all text content based on the confirmed page type

Complete ALL steps automatically without waiting for user input. After completing OCR processing, return the extracted content in a structured format. Do NOT upload to Todoist - the user interface handles that separately.

Be efficient and complete all steps in sequence without stopping."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def _create_page_detection_tool(self):
        """Create the page type detection tool that stores results"""
        def detect_page_type(image_path: str) -> Dict[str, Any]:
            """Detect the type of journal page from an image"""
            try:
                detector = PageTypeDetector()
                result = detector.detect_page_type(image_path)
                
                # Store the page type for later use
                self.last_page_type = result.page_type.value
                
                return {
                    "page_type": result.page_type.value,
                    "reasoning": result.reasoning,
                    "visual_indicators": result.visual_indicators,
                    "success": True
                }
            except Exception as e:
                logger.error(f"Page detection failed: {str(e)}")
                return {
                    "page_type": "Unknown",
                    "reasoning": f"Detection failed: {str(e)}",
                    "visual_indicators": {},
                    "success": False
                }
        
        return StructuredTool.from_function(
            func=detect_page_type,
            name="detect_page_type",
            description="Detect the type of journal page (Daily, Weekly, Monthly) from an image",
            args_schema=PageDetectionInput
        )
    
    def _create_user_confirmation_tool(self):
        """Create the user confirmation tool"""
        def request_user_confirmation(detected_page_type: str, reasoning: str) -> Dict[str, Any]:
            """Request user confirmation for detected page type"""
            logger.info(f"Auto-confirming detected page type: {detected_page_type}")
            self.last_page_type = detected_page_type
            return {
                "confirmed_page_type": detected_page_type,
                "user_confirmed": True,
                "reasoning": reasoning,
                "confirmation_method": "auto"
            }
        
        return StructuredTool.from_function(
            func=request_user_confirmation,
            name="request_user_confirmation",
            description="Confirm the detected page type (auto-confirms for now)",
            args_schema=UserConfirmationInput
        )
    
    def _create_ocr_processing_tool(self):
        """Create the OCR processing tool that stores results"""
        def process_image_ocr(image_path: str, page_type: str) -> Dict[str, Any]:
            """Process journal page image with OCR based on page type"""
            try:
                ocr_adapter = GPT4oOCRAdapter()
                ocr_result = ocr_adapter.extract_text(image_path, page_type)
                
                # Store the OCR data for later use
                self.last_ocr_data = ocr_result
                self.last_page_type = page_type
                
                return {
                    "ocr_data": ocr_result,
                    "page_type": page_type,
                    "success": True
                }
            except Exception as e:
                logger.error(f"OCR processing failed: {str(e)}")
                return {
                    "ocr_data": f"OCR processing failed: {str(e)}",
                    "page_type": page_type,
                    "success": False
                }
        
        return StructuredTool.from_function(
            func=process_image_ocr,
            name="process_image_ocr",
            description="Process journal page image with OCR extraction based on page type",
            args_schema=OCRProcessingInput
        )
    
    def process_journal_image(self, image_path: str, chat_history: list = None) -> Dict[str, Any]:
        """Process a journal image through the complete workflow"""
        if chat_history is None:
            chat_history = []
            
        try:
            response = self.agent_executor.invoke({
                "input": f"Please process this journal image: {image_path}. Follow the complete workflow.",
                "chat_history": chat_history
            })
            
            return {
                "success": True,
                "response": response["output"],
                "chat_history": chat_history
            }
            
        except Exception as e:
            logger.error(f"Agent processing failed: {str(e)}")
            return {
                "success": False,
                "response": f"Processing failed: {str(e)}",
                "chat_history": chat_history
            }
    
    def continue_conversation(self, user_input: str, chat_history: list) -> Dict[str, Any]:
        """Continue the conversation with additional user input"""
        try:
            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            return {
                "success": True,
                "response": response["output"],
                "chat_history": chat_history
            }
            
        except Exception as e:
            logger.error(f"Agent conversation failed: {str(e)}")
            return {
                "success": False,
                "response": f"Conversation failed: {str(e)}",
                "chat_history": chat_history
            }

# Example usage
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agent
    agent = JournalProcessingAgent()
    
    # Process an image
    result = agent.process_journal_image("/path/to/journal/image.jpg")
    print(f"Result: {result}")