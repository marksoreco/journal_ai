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

    def extract_text(self, image_path: str, category: str = "Weekly") -> str:
        logger.info(f"Starting weekly page OCR extraction for image: {image_path}")
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Create the multimodal prompt for weekly pages
        
        # BACKUP - Original detailed prompt (commented out due to hallucination issues):
        # system_prompt = """
        # You are an intelligent assistant that extracts structured entries from scanned weekly journal pages.
        #
        # These pages come from the Monk Manual weekly pages, which contain specific sections. Each page may include handwritten or printed text under the following headings:
        #
        # For weekly pages:
        # - "WEEK": The date range of the week (e.g., "7/29 - 8/3").
        # - "PREPARE - PRIORITY": Numbered priorities (1, 2, 3) for the week.
        # - "TO-DO": Checklist items for weekly tasks. Each item may have an 'X', checkmark (✓), or similar completion mark at the beginning of the line before the text to indicate completion.
        # - "PERSONAL GROWTH": Text about personal development goals.
        # - "RELATIONSHIP(S) GROWTH": Text about relationship improvement goals.
        # - "I'M LOOKING FORWARD TO": Numbered list (1, 2, 3) of anticipated events or activities.
        # - "HABIT TRACKER": Grid showing days of the week (M T W T F S S) for tracking habits. Look for deliberate 'X' marks, checkmarks (✓), or dots that appear immediately BEFORE each day letter. These marks indicate habit completion for that day. Distinguish between intentional completion marks and accidental marks/smudges.
        # - "REFLECT" section contains the sub-sections BIGGEST ACCOMPLISHMENTS, HABITS INSIGHTS, and MEANINGFUL MOMENTS, GOD IS TEACHING ME, and ONE CHANGE I CAN MAKE NEXT WEEK.
        #   - "BIGGEST ACCOMPLISHMENTS": Numbered achievements from the week.  Part of the "REFLECT" section.
        #   - "HABITS INSIGHTS": Reflections on habit formation and progress.  Part of the "REFLECT" section.
        #   - "MEANINGFUL MOMENTS": Special moments or experiences from the week.  Part of the "REFLECT" section.
        #   - "GOD IS TEACHING ME": Spiritual reflections or lessons learned.  Part of the "REFLECT" section.
        #   - "ONE CHANGE I CAN MAKE NEXT WEEK": Single improvement focus for the upcoming week.  Part of the "REFLECT" section.
        # """
        
        # SIMPLIFIED prompt (attempt to fix hallucinations):
        system_prompt = """
        You are an intelligent assistant that extracts structured entries from scanned weekly journal pages.

These pages come from the Monk Manual weekly pages, which contain specific sections. Each page may include handwritten or printed text under the following headings:

For weekly pages:
- "WEEK": The date range of the week (e.g., "7/29 - 8/3").
- "PREPARE - PRIORITY": Top priorities for the week.
- "TO-DO": Weekly tasks to complete.
- "PERSONAL GROWTH": Personal development focus for the week.
- "RELATIONSHIP(S) GROWTH": Relationship improvement goals for the week.
- "I'M LOOKING FORWARD TO": Things anticipated this week.
- "HABIT TRACKER": Grid showing days of the week (M T W T F S S) for tracking habits.
- "REFLECT" section contains the sub-sections BIGGEST ACCOMPLISHMENTS, HABITS INSIGHTS, and MEANINGFUL MOMENTS, GOD IS TEACHING ME, and ONE CHANGE I CAN MAKE NEXT WEEK.
  - "BIGGEST ACCOMPLISHMENTS": Numbered achievements from the week.  Part of the "REFLECT" section.
  - "HABITS INSIGHTS": Reflections on habit formation and progress.  Part of the "REFLECT" section.
  - "MEANINGFUL MOMENTS": Special moments or experiences from the week.  Part of the "REFLECT" section.
  - "GOD IS TEACHING ME": Spiritual reflections or lessons learned.  Part of the "REFLECT" section.
  - "ONE CHANGE I CAN MAKE NEXT WEEK": Single improvement focus for the upcoming week.  Part of the "REFLECT" section.

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

For HABIT TRACKER marks specifically:
- Use high confidence (0.85+) for clear, deliberate marks (X, ✓, etc.) that are clearly positioned before day letters
- Use medium confidence (0.70-0.84) for marks that are likely intentional but positioning is less clear
- Use low confidence (0.50-0.69) for questionable marks or poor image quality
- When marking false (no mark detected), use high confidence if you're certain there's no mark

For HABIT TRACKER, carefully examine each day position for marks that indicate completion.

Confidence guidelines:
- 0.90-1.0: Perfect clarity, printed text, or very neat handwriting
- 0.80-0.89: Clear handwriting, complete words, high certainty
- 0.70-0.79: Readable but some uncertainty, minor smudges, special characters, etc.
- 0.60-0.69: Somewhat unclear, partial words, moderate uncertainty
- 0.50-0.59: Unclear text, significant uncertainty, possible errors
- Below 0.50: Very unclear, likely incorrect interpretation

IMPORTANT: For each extracted item, provide a realistic confidence score (0.0 to 1.0) based on text clarity, handwriting quality, completeness, and image quality.

Here is an example of the expected output format for a Monk Manual weekly page:

{
  "week": {
    "value": "7/29 - 8/3",
    "confidence": 0.98
  },
  "prepare_priority": [
    {"task": "Complete project work", "confidence": 0.94},
    {"task": "Health and exercise", "confidence": 0.87},
    {"task": "Learning activities", "confidence": 0.91}
  ],
  "to_do": [
    {
      "item": "Weekly planning",
      "completed": true,
      "confidence": 0.89
    },
    {
      "item": "Administrative tasks", 
      "completed": false,
      "confidence": 0.85
    },
    {
      "item": "Personal errands",
      "completed": true,
      "confidence": 0.90
    }
  ],
  "personal_growth": {
    "value": "Focus on self-improvement",
    "confidence": 0.85
  },
  "relationships_growth": {
    "value": "Strengthen important relationships",
    "confidence": 0.78
  },
  "looking_forward_to": {
    "1": {"value": "Upcoming projects", "confidence": 0.82},
    "2": {"value": "Personal activities", "confidence": 0.88},
    "3": {"value": "Quality time", "confidence": 0.90}
  },
  "habit_tracker": {
    "monday": {"marked": true, "confidence": 0.88},
    "tuesday": {"marked": false, "confidence": 0.92},
    "wednesday": {"marked": false, "confidence": 0.90},
    "thursday": {"marked": false, "confidence": 0.91},
    "friday": {"marked": false, "confidence": 0.89},
    "saturday": {"marked": false, "confidence": 0.93},
    "sunday": {"marked": false, "confidence": 0.90}
  },
  "reflect": {
    "biggest_accomplishments": [
      {"value": "Achieved important goal", "confidence": 0.89},
      {"value": "Made significant progress", "confidence": 0.84}
    ],
    "habits_insights": {
      "value": "Working on consistency",
      "confidence": 0.75
    },
    "meaningful_moments": {
      "value": "Special moment with family",
      "confidence": 0.73
    },
    "god_is_teaching_me": {
      "value": "Lessons in patience",
      "confidence": 0.80
    },
    "one_change_next_week": {
      "value": "Improve daily routine",
      "confidence": 0.85
    }
  }
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
                        "week": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "The date range of the week (e.g., '7/29 - 8/3')"
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
                            "description": "Top priorities for the week"
                        },
                        "to_do": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "item": { "type": "string" },
                                    "completed": { "type": "boolean" },
                                    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                },
                                "required": ["item", "completed", "confidence"]
                            },
                            "description": "Weekly to-do checklist items"
                        },
                        "personal_growth": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "Personal development focus for the week"
                        },
                        "relationships_growth": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                            },
                            "required": ["value", "confidence"],
                            "description": "Relationship improvement goals for the week"
                        },
                        "looking_forward_to": {
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
                                }
                            },
                            "description": "Top 3 things looking forward to this week"
                        },
                        "habit_tracker": {
                            "type": "object",
                            "properties": {
                                "monday": {
                                    "type": "object",
                                    "properties": {
                                        "marked": { "type": "boolean" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["marked", "confidence"]
                                },
                                "tuesday": {
                                    "type": "object",
                                    "properties": {
                                        "marked": { "type": "boolean" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["marked", "confidence"]
                                },
                                "wednesday": {
                                    "type": "object",
                                    "properties": {
                                        "marked": { "type": "boolean" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["marked", "confidence"]
                                },
                                "thursday": {
                                    "type": "object",
                                    "properties": {
                                        "marked": { "type": "boolean" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["marked", "confidence"]
                                },
                                "friday": {
                                    "type": "object",
                                    "properties": {
                                        "marked": { "type": "boolean" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["marked", "confidence"]
                                },
                                "saturday": {
                                    "type": "object",
                                    "properties": {
                                        "marked": { "type": "boolean" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["marked", "confidence"]
                                },
                                "sunday": {
                                    "type": "object",
                                    "properties": {
                                        "marked": { "type": "boolean" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["marked", "confidence"]
                                }
                            },
                            "required": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                            "description": "Habit tracking for each day of the week (M T W T F S S order).  The 'marked' property indicates whether an X or checkmark appears before each day letter"
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
                                    "description": "Biggest accomplishments from the week"
                                },
                                "habits_insights": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"],
                                    "description": "Insights about habit formation and progress"
                                },
                                "meaningful_moments": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"],
                                    "description": "Meaningful moments from the week"
                                },
                                "god_is_teaching_me": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"],
                                    "description": "Spiritual reflections and lessons learned"
                                },
                                "one_change_next_week": {
                                    "type": "object",
                                    "properties": {
                                        "value": { "type": "string" },
                                        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                                    },
                                    "required": ["value", "confidence"],
                                    "description": "One change to focus on for the upcoming week"
                                }
                            },
                            "required": ["biggest_accomplishments", "habits_insights", "meaningful_moments", "god_is_teaching_me", "one_change_next_week"],
                            "description": "Weekly reflection sections"
                        }
                    },
                    "required": ["week", "prepare_priority", "to_do", "habit_tracker", "reflect", "personal_growth", "relationships_growth", "looking_forward_to"]
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
                max_tokens=2500
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