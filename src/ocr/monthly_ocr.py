import logging
from .base import BaseOCR
import base64
import os
import json
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
import dateparser
from datetime import datetime

# Configure logger for this module
logger = logging.getLogger(__name__)

class MonthlyOCRAdapter(BaseOCR):
    def __init__(self):
        # Initialize OpenAI client with API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)
    
    def _parse_month_value(self, month_str: str) -> dict:
        """
        Parse various month/year formats using dateparser.
        
        Handles formats like:
        - "Sept 2023", "March 2024" 
        - "3/2023", "03/2023"
        - "3/23", "03/23" (assumes 20xx for years ≤30, 19xx for years >30)
        - "March" (defaults to current year)
        
        Returns:
            dict with 'month', 'year', and 'confidence' fields
        """
        try:
            if not month_str or not month_str.strip():
                # Default to current month/year
                now = datetime.now()
                return {
                    "month": now.strftime("%B"),
                    "year": now.year,
                    "confidence": 0.5
                }
            
            month_str = month_str.strip()
            
            # Try to parse with dateparser (handles most formats automatically)
            parsed_date = dateparser.parse(month_str, settings={
                'PREFER_DATES_FROM': 'past',  # For ambiguous cases
                'DATE_ORDER': 'MDY'  # Month-Day-Year preference
            })
            
            if parsed_date:
                return {
                    "month": parsed_date.strftime("%B"),  # Full month name
                    "year": parsed_date.year,
                    "confidence": 0.9
                }
            
            # Fallback: try to handle MM/YY format manually (dateparser sometimes struggles with this)
            if '/' in month_str and len(month_str.split('/')) == 2:
                parts = month_str.split('/')
                try:
                    month_num = int(parts[0])
                    year_part = int(parts[1])
                    
                    if 1 <= month_num <= 12:
                        # Handle 2-digit years
                        if year_part < 100:
                            year = 2000 + year_part if year_part <= 30 else 1900 + year_part
                        else:
                            year = year_part
                        
                        month_names = [
                            "January", "February", "March", "April", "May", "June",
                            "July", "August", "September", "October", "November", "December"
                        ]
                        
                        return {
                            "month": month_names[month_num - 1],
                            "year": year,
                            "confidence": 0.8
                        }
                except ValueError:
                    pass
            
            # If nothing else worked, default to current date
            now = datetime.now()
            return {
                "month": now.strftime("%B"),
                "year": now.year,
                "confidence": 0.3  # Low confidence since we couldn't parse
            }
            
        except Exception as e:
            logger.warning(f"Error parsing month value '{month_str}': {str(e)}")
            # Default to current month/year
            now = datetime.now()
            return {
                "month": now.strftime("%B"),
                "year": now.year,
                "confidence": 0.3
            }

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
- "CALENDAR": Section in calendar format with days of the month and the corresponding events.  Note the text will be just below and slightly to the left of the day number.
- "PREPARE - PRIORITY": Important items to work on this month.
- "MONTHLY CHECK-IN": Wellness ratings for different categories, including relationships, physical, spiritual, work/vocation, personal growth, play and peace. These are values between 1 and 10.
- "ONE CHANGE I CAN MAKE THIS MONTH THAT WILL HAVE THE BIGGEST IMPACT": A single change that can be made this month that will have the biggest impact.
- "ONE QUESTION I'D LIKE TO ANSWER THIS MONTH": A single question that I'd like to answer this month.
- "REFLECT": Page section with reflections on the month.  Includes subsections for "BIGGEST ACCOMPLISHMENTS", "RELATIONSHIPS I'M GRATEFUL FOR", and "GREATEST INSIGHT GAINED".

Your job is to:
- Identify each section based on layout or header.
- Extract the text content written under each section.
- Return a JSON object with a key for each section and its content.
- If a section is not present or has no content, include it in the json output with an empty string or empty array.
- If no text content is present for a line under a section, do not include that line in the json output.

IMPORTANT: For the "MONTHLY CHECK-IN" section, go one-by-one through each category and get the numeric value that is located directly to the right of the category name.  IGNORE ANY VALUES LOCATED BELOW OR ABOVE THE CATEGORY NAME'S VERTICAL LOCATION IN THE IMAGE. The values may be located at different distances from the category name but will always be directly to the right of the category name.

For the "CALENDAR" section, double-check and triple-check that the text that was identified is located just below and slightly to the left of the day number that you associated it with.  If it is not then reprocess the image and try again.

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
        "month": "March",
        "year": 2024,
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
        "physical": 9,
        "spiritual": 3,
        "work/vocation": 4,
        "personal growth": 6,
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
                                "month": { "type": "string" },
                                "year": { "type": "integer" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["month", "year", "confidence"],
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
                                "relationships": { "type": "integer", "minimum": 1, "maximum": 10 },
                                "physical": { "type": "integer", "minimum": 1, "maximum": 10 },
                                "spiritual": { "type": "integer", "minimum": 1, "maximum": 10 },
                                "work/vocation": { "type": "integer", "minimum": 1, "maximum": 10 },
                                "personal growth": { "type": "integer", "minimum": 1, "maximum": 10 },
                                "play": { "type": "integer", "minimum": 1, "maximum": 10 },
                                "peace": { "type": "integer", "minimum": 1, "maximum": 10 }
                            },
                            "description": "Wellness ratings from 1 to 10 for different life categories"
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
                    # Parse the OCR result
                    ocr_data = json.loads(tool_call.function.arguments)
                    
                    # Post-process the month field if it exists
                    if 'month' in ocr_data and ocr_data['month']:
                        # Handle both old format (with 'value' field) and new format (with separate fields)
                        if isinstance(ocr_data['month'], dict):
                            if 'value' in ocr_data['month']:
                                # Old format: parse the value field
                                month_str = ocr_data['month']['value']
                                parsed_month = self._parse_month_value(month_str)
                                # Update to new format
                                ocr_data['month'] = {
                                    "month": parsed_month['month'],
                                    "year": parsed_month['year'],
                                    "confidence": min(ocr_data['month'].get('confidence', 0.5), parsed_month['confidence'])
                                }
                            elif 'month' in ocr_data['month'] and 'year' in ocr_data['month']:
                                # New format: validate and potentially re-parse if confidence is low
                                if ocr_data['month'].get('confidence', 0) < 0.7:
                                    month_str = f"{ocr_data['month']['month']} {ocr_data['month']['year']}"
                                    parsed_month = self._parse_month_value(month_str)
                                    ocr_data['month'].update(parsed_month)
                        else:
                            # Handle case where month is just a string
                            parsed_month = self._parse_month_value(str(ocr_data['month']))
                            ocr_data['month'] = parsed_month
                    
                    result = json.dumps(ocr_data, indent=2)
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