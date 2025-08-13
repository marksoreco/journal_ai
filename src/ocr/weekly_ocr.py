import logging
from .base import BaseOCR
import base64
import os
import json
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam

# Configure logger for this module
logger = logging.getLogger(__name__)

class WeeklyOCRAdapter(BaseOCR):
    def __init__(self):
        # Initialize OpenAI client with API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)

    def extract_text(self, image_path: str, category: str = "Week") -> str:
        logger.info(f"Starting weekly page OCR extraction for image: {image_path}")
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Create the multimodal prompt for weekly pages
        system_prompt = """
        You are an intelligent assistant that extracts structured entries from scanned weekly journal pages.

These pages come from the Monk Manual weekly pages, which contain specific sections. Each page may include handwritten or printed text under the following headings:

For weekly pages:
- "WEEK OF": The date range of the week.
- "WEEKLY PRIORITIES": The top priorities for the week.
- "WEEKLY GOALS": Goals to accomplish during the week.
- "HABITS TO FOCUS ON": Habits to work on this week.
- "WEEKLY REFLECTION": Reflection on the previous week.
- "DAILY BREAKDOWN": Day-by-day tasks and priorities.

Your job is to:
- Identify each section based on layout or header.
- Extract the text content written under each section.
- Return a JSON object with a key for each section and its content.
- If a section is not present or has no content, leave it blank or null.

IMPORTANT: For each extracted item, provide a realistic confidence score (0.0 to 1.0) based on:
- Text clarity and readability (clear text = higher confidence)
- Handwriting quality (neat handwriting = higher confidence)
- Completeness of text (complete words = higher confidence)
- Certainty of interpretation (unambiguous = higher confidence)
- Image quality (sharp, well-lit = higher confidence)

Confidence guidelines:
- 0.90-1.0: Perfect clarity, printed text, or very neat handwriting
- 0.80-0.89: Clear handwriting, complete words, high certainty
- 0.70-0.79: Readable but some uncertainty, minor smudges, special characters, etc.
- 0.60-0.69: Somewhat unclear, partial words, moderate uncertainty
- 0.50-0.59: Unclear text, significant uncertainty, possible errors
- Below 0.50: Very unclear, likely incorrect interpretation

Here is an example of the expected output format for a Monk Manual weekly page:

{
  "week_of": {
    "value": "March 15-21, 2024",
    "confidence": 0.98
  },
  "weekly_priorities": [
    {
      "priority": "Complete project proposal",
      "confidence": 0.94
    },
    {
      "priority": "Plan team workshop",
      "confidence": 0.87
    },
    {
      "priority": "Review quarterly goals",
      "confidence": 0.91
    }
  ],
  "weekly_goals": [
    {
      "goal": "Launch marketing campaign",
      "confidence": 0.89
    },
    {
      "goal": "Improve team communication",
      "confidence": 0.83
    }
  ],
  "habits_to_focus_on": [
    {
      "habit": "Morning exercise routine",
      "confidence": 0.92
    },
    {
      "habit": "Daily reading",
      "confidence": 0.95
    }
  ],
  "weekly_reflection": {
    "value": "Last week was productive but lacked focus on personal development",
    "confidence": 0.78
  },
  "daily_breakdown": [
    {
      "day": "Monday",
      "tasks": [
        {
          "task": "Team standup meeting",
          "confidence": 0.93
        }
      ]
    }
  ]
}
"""

        tools: list[ChatCompletionToolParam] = [{
            "type": "function",
            "function": {
                "name": "extract_weekly_page",
                "description": "Extract structured weekly page entries from a scanned Monk Manual page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "week_of": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "The date range of the week"
                        },
                        "weekly_priorities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "priority": { "type": "string" },
                                    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                },
                                "required": ["priority", "confidence"]
                            },
                            "description": "Main priorities for the week"
                        },
                        "weekly_goals": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "goal": { "type": "string" },
                                    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                },
                                "required": ["goal", "confidence"]
                            },
                            "description": "Goals to accomplish during the week"
                        },
                        "habits_to_focus_on": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "habit": { "type": "string" },
                                    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                },
                                "required": ["habit", "confidence"]
                            },
                            "description": "Habits to work on this week"
                        },
                        "weekly_reflection": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "Reflection on the previous week"
                        },
                        "daily_breakdown": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "day": {
                                        "type": "string",
                                        "description": "Day of the week (Monday, Tuesday, etc.)"
                                    },
                                    "tasks": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "task": { "type": "string" },
                                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                            },
                                            "required": ["task", "confidence"]
                                        },
                                        "description": "Tasks for this day"
                                    }
                                },
                                "required": ["day", "tasks"]
                            },
                            "description": "Day-by-day breakdown of tasks and priorities"
                        }
                    },
                    "required": ["week_of", "weekly_priorities"]
                }
            }
        }]

        try:
            logger.debug("Sending request to GPT-4o for weekly page OCR processing")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
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
                tool_choice={"type": "function", "function": {"name": "extract_weekly_page"}},
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            # Check if tool was called
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                logger.debug("GPT-4o returned function call response")
                try:
                    result = json.dumps(json.loads(tool_call.function.arguments), indent=2)
                    logger.info("Weekly page OCR extraction completed successfully")
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response from GPT-4o: {tool_call.function.arguments}")
                    return f"Error: Invalid JSON response from GPT-4o: {tool_call.function.arguments}"
            else:
                logger.warning("GPT-4o did not return function call, using content instead")
            return content.strip() if content else ""
            
        except Exception as e:
            logger.error(f"Error processing weekly page image: {str(e)}")
            return f"Error processing weekly page image: {str(e)}"