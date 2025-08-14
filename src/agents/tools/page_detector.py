"""
Page Type Detection Tool for Journal Processing

This module provides automated detection of journal page types (Daily, Weekly, Monthly)
using GPT-4o vision capabilities. Designed to be agent-compatible and modular.
"""

import logging
import base64
import os
import json
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from PIL import Image
import io

# Configure logger
logger = logging.getLogger(__name__)

class PageType(Enum):
    """Enumeration of supported journal page types"""
    DAILY = "Daily"
    WEEKLY = "Weekly" 
    MONTHLY = "Monthly"
    UNKNOWN = "Unknown"

@dataclass
class DetectionResult:
    """Result of page type detection"""
    page_type: PageType
    reasoning: str
    visual_indicators: Dict[str, bool]

class PageTypeDetector:
    """
    Automated page type detection using GPT-4o vision.
    
    Analyzes journal page images to determine if they are Daily, Weekly, or Monthly pages
    based on visual layout, headers, section structure, and content patterns.
    """
    
    def __init__(self):
        """
        Initialize the page type detector.
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
        
    def detect_page_type(self, image_path: str) -> DetectionResult:
        """
        Detect the type of journal page from an image.
        
        Args:
            image_path: Path to the journal page image
            
        Returns:
            DetectionResult with page type, confidence, and reasoning
        """
        logger.info(f"Starting page type detection for image: {image_path}")
        
        try:
            # Read, optimize, and encode the image
            optimized_image = self._optimize_image(image_path)
            encoded_image = base64.b64encode(optimized_image).decode('utf-8')
            
            # Analyze the image for page type indicators
            detection_data = self._analyze_image(encoded_image)
            
            # Process results and create detection result
            result = self._process_detection_results(detection_data)
            
            logger.info(f"Page type detection completed: {result.page_type.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during page type detection: {str(e)}")
            return DetectionResult(
                page_type=PageType.UNKNOWN,
                reasoning=f"Detection failed: {str(e)}",
                visual_indicators={}
            )
    
    def _optimize_image(self, image_path: str, max_size: int = 1024, quality: int = 80) -> bytes:
        """
        Optimize image for faster processing by resizing and compressing.
        
        Args:
            image_path: Path to the original image
            max_size: Maximum width or height in pixels
            quality: JPEG compression quality (1-100)
            
        Returns:
            Optimized image as bytes
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (for JPEG compatibility)
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Calculate new size maintaining aspect ratio
                width, height = img.size
                if width > max_size or height > max_size:
                    if width > height:
                        new_width = max_size
                        new_height = int((height * max_size) / width)
                    else:
                        new_height = max_size
                        new_width = int((width * max_size) / height)
                    
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
                
                # Save as optimized JPEG
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                optimized_bytes = buffer.getvalue()
                
                logger.debug(f"Image optimization complete. Size reduction: {len(open(image_path, 'rb').read())} -> {len(optimized_bytes)} bytes")
                return optimized_bytes
                
        except Exception as e:
            logger.warning(f"Image optimization failed, using original: {str(e)}")
            # Fallback to original image
            with open(image_path, "rb") as f:
                return f.read()
    
    def _analyze_image(self, encoded_image: str) -> Dict:
        """
        Use GPT-4o vision to analyze the image and identify page type indicators.
        
        Args:
            encoded_image: Base64 encoded image data
            
        Returns:
            Dictionary containing detection analysis results
        """
        system_prompt = """
        Analyze this journal page to determine its type: Daily, Weekly, or Monthly.
        
        KEY IDENTIFIERS:
        
        DAILY: Look for hour-by-hour schedule sections, single-day focus, "I AM GRATEFUL FOR" section, "THEME" section, reflection sections
        
        WEEKLY: Look for "HABIT TRACKER" with day letters (M T W T F S S), weekly reflection sections
        
        MONTHLY: Look for calendar grid layout, "MONTHLY CHECK-IN" wellness ratings, monthly goals
        
        Focus on layout structure and key section headers to classify the page type.
        """
        
        tools: list[ChatCompletionToolParam] = [{
            "type": "function",
            "function": {
                "name": "analyze_page_type",
                "description": "Analyze journal page layout to determine page type",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "detected_page_type": {
                            "type": "string",
                            "enum": ["Daily", "Weekly", "Monthly", "Unknown"],
                            "description": "The detected page type based on visual analysis"
                        },
                        "visual_indicators": {
                            "type": "object",
                            "properties": {
                                "has_calendar_grid": {
                                    "type": "boolean", 
                                    "description": "Calendar grid present (Monthly indicator)"
                                },
                                "has_habit_tracker": {
                                    "type": "boolean",
                                    "description": "Habit tracker with day letters (Weekly indicator)"
                                },
                                "has_hourly_schedule": {
                                    "type": "boolean",
                                    "description": "Hour-by-hour schedule (Daily indicator)"
                                }
                            },
                            "description": "Key visual indicators found"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of classification"
                        }
                    },
                    "required": ["detected_page_type", "visual_indicators", "reasoning"]
                }
            }
        }]
        
        try:
            logger.debug("Sending image to GPT-4o-mini for page type analysis")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this journal page image and determine its type. Look carefully at the section headers and layout structure, ignoring specific dates or date formats."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "analyze_page_type"}},
                max_tokens=200
            )
            
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                return json.loads(tool_call.function.arguments)
            else:
                raise ValueError("No tool call response received from GPT-4o")
                
        except Exception as e:
            logger.error(f"Error in GPT-4o image analysis: {str(e)}")
            raise
    
    def _process_detection_results(self, detection_data: Dict) -> DetectionResult:
        """
        Process the raw detection data into a structured result.
        
        Args:
            detection_data: Raw detection results from GPT-4o
            
        Returns:
            Processed DetectionResult object
        """
        try:
            # Map string to enum
            page_type_str = detection_data.get("detected_page_type", "Unknown")
            page_type = PageType(page_type_str)
        except ValueError:
            logger.warning(f"Unknown page type detected: {page_type_str}")
            page_type = PageType.UNKNOWN
        
        visual_indicators = detection_data.get("visual_indicators", {})
        reasoning = detection_data.get("reasoning", "No reasoning provided")
        
        return DetectionResult(
            page_type=page_type,
            reasoning=reasoning,
            visual_indicators=visual_indicators
        )
    

# Agent-compatible tool function
def detect_journal_page_type(image_path: str) -> Dict:
    """
    Agent-compatible function for page type detection.
    
    Args:
        image_path: Path to the journal page image
        
    Returns:
        Dictionary with detection results suitable for agent processing
    """
    detector = PageTypeDetector()
    result = detector.detect_page_type(image_path)
    
    return {
        "page_type": result.page_type.value,
        "reasoning": result.reasoning,
        "visual_indicators": result.visual_indicators
    }

# Example usage for testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    detector = PageTypeDetector()
    
    # Test with an image (replace with actual path)
    # result = detector.detect_page_type("path/to/journal/page.png")
    # print(f"Detected: {result.page_type.value} (confidence: {result.confidence:.2f})")
    # print(f"Reasoning: {result.reasoning}")