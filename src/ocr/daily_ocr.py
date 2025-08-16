import logging
from .base import BaseOCR
import base64
import os
import json
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam

# Configure logger for this module
logger = logging.getLogger(__name__)

class DailyOCRAdapter(BaseOCR):
    def __init__(self):
        # Initialize OpenAI client with API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)

    def extract_text(self, image_path: str, category: str = "Daily") -> str:
        logger.info(f"Starting daily page OCR extraction for image: {image_path}")
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Create the multimodal prompt for daily pages
        system_prompt = """
        You are an intelligent assistant that extracts structured entries from scanned daily journal pages.

These pages come from the Monk Manual daily pages, which contain specific sections. Each page may include handwritten or printed text under the following headings:

For daily pages:
- "DATE": The date of the page.
- "HABIT": The habit that the user wants to work on this day.
- "THEME": The theme or focus for the day.
- "PREPARE - PRIORITY": The top 3 priorities for the day.
- "TO-DO": Other smaller tasks to complete.
- "I AM GRATEFUL FOR": Things the user is grateful for.
- "I'M LOOKING FORWARD TO": Things the user is looking forward to.
- "DAILY": Hour-by-hour breakdown of the day.
- "WAYS I CAN GIVE": Ways the user can give or serve others.
- "REFLECT": General reflection section header.  Includes sub-sections "HIGHLIGHTS", "I WAS AT MY BEST WHEN", "I FELT UNREST WHEN", and "ONE WAY I CAN IMPROVE TOMORROW".
- "HIGHLIGHTS": Highlights of the day.  Part of the "REFLECT" section.
- "I WAS AT MY BEST WHEN": When the user felt they performed at their best.  Part of the "REFLECT" section.
- "I FELT UNREST WHEN": When the user felt unsettled or uncomfortable.  Part of the "REFLECT" section.
- "ONE WAY I CAN IMPROVE TOMORROW": One specific improvement for the next day.  Part of the "REFLECT" section.

Your job is to:
- Identify each section based on layout or header.
- Extract the text content written under each section.
- Return a JSON object with a key for each section and its content.
- If a section is not present or has no content, leave it blank or null.

For the "DAILY" section, be sure to allow for the possibility that there is no text for a given hour.  If there is no text for a given hour do not associate that hour with the text for the following hour.

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

Here is an example of the expected output format for a Monk Manual daily page:

{
  "date": {
    "value": "Monday, Nov 12, 2018",
    "confidence": 0.98
  },
  "habit": {
    "value": "Leave work at work",
    "confidence": 0.82
  },
  "theme": {
    "value": "Focus",
    "confidence": 0.90
  },
  "prepare_priority": [
    {
      "task": "Plan upcoming week",
      "confidence": 0.94
    },
    {
      "task": "Finish monthly report",
      "confidence": 0.87
    },
    {
      "task": "Do laundry/put away",
      "confidence": 0.68
    }
  ],
  "to_do": [
    {
      "task": "Check the AM news",
      "confidence": 0.89
    },
    {
      "task": "Check AM emails",
      "confidence": 0.92
    }
  ],
  "i_am_grateful_for": [
    {
      "item": "Warm weather",
      "confidence": 0.95
    },
    {
      "item": "Coffee",
      "confidence": 0.98
    }
  ],
  "i_am_looking_forward_to": [
    {
      "item": "Catching up with Matt",
      "confidence": 0.93
    }
  ],
  "daily": [
    {
      "hour": 6,
      "activities": [
        {
          "activity": "Check the news",
          "confidence": 0.91
        }
      ]
    },
    {
      "hour": 7,
      "activities": [
        {
          "activity": "Get ready for work",
          "confidence": 0.94
        }
      ]
    }
  ],
  "ways_i_can_give": [
    {
      "item": "Patience and focus in staff meeting",
      "confidence": 0.85
    }
  ],
  "reflect": {
    "highlights": [
      {
        "value": "Finally finishing the monthly report",
        "confidence": 0.88
      },
      {
        "value": "Helping a coworker with a difficult task",
        "confidence": 0.91
      }
    ],
    "i_was_at_my_best_when": {
      "value": "Leading the staff meeting",
      "confidence": 0.92
    },
    "i_felt_unrest_when": {
      "value": "Planning for the busy week ahead",
      "confidence": 0.87
    },
    "one_way_i_can_improve_tomorrow": {
      "value": "Finish work early and fit in an evening workout",
      "confidence": 0.91
    }
  }
}
"""

        tools: list[ChatCompletionToolParam] = [{
            "type": "function",
            "function": {
                "name": "extract_daily_page",
                "description": "Extract structured daily page entries from a scanned Monk Manual page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "The date of the page"
                        },
                        "habit": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "Habit to focus on today"
                        },
                        "theme": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "Theme or focus for the day"
                        },
                        "prepare_priority": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "task": { "type": "string" },
                                    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                },
                                "required": ["task", "confidence"]
                            },
                            "description": "The three biggest priorities for the day"
                        },
                        "to_do": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "task": { "type": "string" },
                                    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                },
                                "required": ["task", "confidence"]
                            },
                            "description": "Additional lower-priority tasks for the day"
                        },
                        "i_am_grateful_for": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "item": { "type": "string" },
                                    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                },
                                "required": ["item", "confidence"]
                            },
                            "description": "Things the user is grateful for today"
                        },
                        "i_am_looking_forward_to": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "item": { "type": "string" },
                                    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                },
                                "required": ["item", "confidence"]
                            },
                            "description": "Things to look forward to today"
                        },
                        "ways_i_can_give": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "item": { "type": "string" },
                                    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                },
                                "required": ["item", "confidence"]
                            },
                            "description": "Ways the user can give or serve others"
                        },
                        "reflect": {
                            "type": "object",
                            "properties": {
                                "highlights": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "value": { "type": "string" },
                                            "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                        },
                                        "required": ["value", "confidence"]
                                    },
                                    "description": "Highlights of the day"
                                },
                                "i_was_at_my_best_when": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"],
                                    "description": "When the user felt they performed at their best"
                                },
                                "i_felt_unrest_when": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"],
                                    "description": "When the user felt unsettled or uncomfortable"
                                },
                                "one_way_i_can_improve_tomorrow": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"],
                                    "description": "One specific improvement for the next day"
                                }
                            },
                            "description": "Reflection section with sub-sections for highlights, best moments, unrest, and improvement"
                        },
                        "daily": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "hour": {
                                        "type": "number",
                                        "description": "Hour of the day (6-22, representing 6 AM to 10 PM)"
                                    },
                                    "activities": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "activity": { "type": "string" },
                                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                            },
                                            "required": ["activity", "confidence"]
                                        },
                                        "description": "Activities for this hour"
                                    }
                                },
                                "required": ["hour", "activities"]
                            },
                            "description": "Hourly breakdown of the day as tuples of [hour, activities]"
                        },
                    },
                    "required": ["date", "prepare_priority"]
                }
            }
        }]

        try:
            logger.debug("Sending request to GPT-4o for daily page OCR processing")
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
                tool_choice={"type": "function", "function": {"name": "extract_daily_page"}},
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            # Check if tool was called
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                logger.debug("GPT-4o returned function call response")
                try:
                    result = json.dumps(json.loads(tool_call.function.arguments), indent=2)
                    logger.info("Daily page OCR extraction completed successfully")
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response from GPT-4o: {tool_call.function.arguments}")
                    return f"Error: Invalid JSON response from GPT-4o: {tool_call.function.arguments}"
            else:
                logger.warning("GPT-4o did not return function call, using content instead")
            return content.strip() if content else ""
            
        except Exception as e:
            logger.error(f"Error processing daily page image: {str(e)}")
            return f"Error processing daily page image: {str(e)}"