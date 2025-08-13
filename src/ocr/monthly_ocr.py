import logging
from .base import BaseOCR
import base64
import os
import json
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam

# Configure logger for this module
logger = logging.getLogger(__name__)

class MonthlyOCRAdapter(BaseOCR):
    def __init__(self):
        # Initialize OpenAI client with API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)

    def extract_text(self, image_path: str, category: str = "Month") -> str:
        logger.info(f"Starting monthly page OCR extraction for image: {image_path}")
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Create the multimodal prompt for monthly pages
        system_prompt = """
        You are an intelligent assistant that extracts structured entries from scanned monthly journal pages.

These pages come from the Monk Manual monthly pages, which contain specific sections. Each page may include handwritten or printed text under the following headings:

For monthly pages:
- "MONTH": The month and year being planned.
- "HABIT": Habit to establish or maintain this month.
- "THEME": Main theme or focus area for the month.
- "CALENDAR": Section in calendar format with days of the month and the corresponding events.
- "PREPARE - PRIORITY": Important items to work on this month.
- "MONTHLY CHECK-IN": Wellness ratings for different categories, including relationships, physical, spiritual, work/vocation, personal growth, play and peace.
- "ONE CHANGE I CAN MAKE THIS MONTH THAT WILL HAVE THE BIGGEST IMPACT": A single change that can be made this month that will have the biggest impact.
- "ONE QUESTION I'D LIKE TO ANSWER THIS MONTH": A single question that I'd like to answer this month.
- "REFLECT": Page section with reflections on the month.  Includes subsections for "BIGGEST ACCOMPLISHMENTS", "RELATIONSHIPS I'M GRATEFUL FOR", and "GREATEST INSIGHT GAINED".

Your job is to:
- Identify each section based on layout or header.
- Extract the text content written under each section.
- Return a JSON object with a key for each section and its content.
- If a section is not present or has no content, leave it blank or null.

NOTE for the MONTHLY CHECK-IN:
- Under the "MONTHLY CHECK-IN" heading, there is a row for each category.
- For each category, there is a scale in the form of a horizontal line with a short vertical line in the middle.
- The user has placed an 'X' somewhere along the scale to indicate their rating for the category.
- The rating is a number between -5 and +5, where -5 is the left end of the line, 0 is the location of the vertical bar in the middle of the line, and +5 is the right end of the line.
- You are to estimate the rating based on the location of the 'X' that the user has placed on the line.

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

Here is an example of the expected output format for a Monk Manual monthly page:

{
    "month": {
        "value": "March 2024",
        "confidence": 0.98
    },
    "habit": {
        "value": "Complete certification program",
        "confidence": 0.91
    },
    "theme": {
        "value": "Professional growth and development",
        "confidence": 0.94
    },
    "calendar": [
        {
            "day": "5",
            "value": "Doctor appointment",
            "confidence": 0.82
        },
        {
            "day": "18",
            "value": "Pick up Jeff at airport",
            "confidence": 0.8
        }
    ],
    "prepare_priority": {
        "1": {"value": "Establish morning routine", "confidence": 0.83},
        "2": {"value": "Complete certification program", "confidence": 0.88},
        "3": {"value": "Launch new product line", "confidence": 0.92},
        "4": {"value": "Team training initiative", "confidence": 0.85},
        "5": {"value": "Website redesign project", "confidence": 0.90}
    },
    "monthly_check_in": {
        "relationships": 1,
        "physical": -2,
        "spiritual": 3,
        "work/vocation": 4,
        "personal growth": -1,
        "play": 5,
        "peace": 2
    },
    "one_change_i_can_make_this_month_that_will_have_the_biggest_impact": {
        "value": "Get more sleep",
        "confidence": 0.83
    },
    "one_question_i_d_like_to_answer_this_month": {
        "value": "What is the best way to improve my relationships?",
        "confidence": 0.88
    },
    "reflect": {
        "biggest_accomplishments": [
            {"value": "Completed certification program", "confidence": 0.92},
            {"value": "Launched new product line", "confidence": 0.80},
            {"value": "Completed certification program", "confidence": 0.72}
        ],
        "relationships_i_m_grateful_for": [
            {"value": "My wife", "confidence": 0.92},
            {"value": "Ellie", "confidence": 0.80},
            {"value": "All the other dogs", "confidence": 0.72}
        ],
        "greatest_insight_gained": {
            "value": "The importance of self-care",
            "confidence": 0.85
        }
    }
}
"""

        tools: list[ChatCompletionToolParam] = [{
            "type": "function",
            "function": {
                "name": "extract_monthly_page",
                "description": "Extract structured monthly page entries from a scanned Monk Manual page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "month": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "The month and year being planned"
                        },
                        "habit": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "Habit to establish or maintain this month"
                        },
                        "theme": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "Main theme or focus area for the month"
                        },
                        "calendar": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "day": { "type": "string" },
                                    "value": { "type": "string" },
                                    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                },
                                "required": ["day", "value", "confidence"]
                            },
                            "description": "Calendar events with days and corresponding events"
                        },
                        "prepare_priority": {
                            "type": "object",
                            "properties": {
                                "1": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"]
                                },
                                "2": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"]
                                },
                                "3": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"]
                                },
                                "4": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"]
                                },
                                "5": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"]
                                }
                            },
                            "description": "Top 5 priorities for the month"
                        },
                        "monthly_check_in": {
                            "type": "object",
                            "properties": {
                                "relationships": { "type": "integer", "minimum": -5, "maximum": 5 },
                                "physical": { "type": "integer", "minimum": -5, "maximum": 5 },
                                "spiritual": { "type": "integer", "minimum": -5, "maximum": 5 },
                                "work/vocation": { "type": "integer", "minimum": -5, "maximum": 5 },
                                "personal growth": { "type": "integer", "minimum": -5, "maximum": 5 },
                                "play": { "type": "integer", "minimum": -5, "maximum": 5 },
                                "peace": { "type": "integer", "minimum": -5, "maximum": 5 }
                            },
                            "description": "Wellness ratings from -5 to +5 for different life categories"
                        },
                        "one_change_i_can_make_this_month_that_will_have_the_biggest_impact": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "One change that can have the biggest impact this month"
                        },
                        "one_question_i_d_like_to_answer_this_month": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "One question to answer this month"
                        },
                        "reflect": {
                            "type": "object",
                            "properties": {
                                "biggest_accomplishments": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "value": { "type": "string" },
                                            "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                        },
                                        "required": ["value", "confidence"]
                                    },
                                    "description": "Biggest accomplishments"
                                },
                                "relationships_i_m_grateful_for": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "value": { "type": "string" },
                                            "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                        },
                                        "required": ["value", "confidence"]
                                    },
                                    "description": "Relationships I'm grateful for"
                                },
                                "greatest_insight_gained": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"],
                                    "description": "Greatest insight gained"
                                }
                            },
                            "required": ["biggest_accomplishments", "relationships_i_m_grateful_for", "greatest_insight_gained"],
                            "description": "Monthly reflection sections"
                        }
                    },
                    "required": ["month", "prepare_priority"]
                }
            }
        }]

        try:
            logger.debug("Sending request to GPT-4o for monthly page OCR processing")
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
                tool_choice={"type": "function", "function": {"name": "extract_monthly_page"}},
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            # Check if tool was called
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                logger.debug("GPT-4o returned function call response")
                try:
                    result = json.dumps(json.loads(tool_call.function.arguments), indent=2)
                    logger.info("Monthly page OCR extraction completed successfully")
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response from GPT-4o: {tool_call.function.arguments}")
                    return f"Error: Invalid JSON response from GPT-4o: {tool_call.function.arguments}"
            else:
                logger.warning("GPT-4o did not return function call, using content instead")
            return content.strip() if content else ""
            
        except Exception as e:
            logger.error(f"Error processing monthly page image: {str(e)}")
            return f"Error processing monthly page image: {str(e)}"