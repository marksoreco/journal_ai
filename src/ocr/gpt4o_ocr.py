from .base import BaseOCR
import base64
import os
import json
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam

class GPT4oOCRAdapter(BaseOCR):
    def __init__(self):
        # Initialize OpenAI client with API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)

    def extract_text(self, image_path: str) -> str:
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Create the multimodal prompt
        system_prompt = """
        You are an intelligent assistant that extracts structured entries from scanned journal pages.

These pages come from the Monk Manual, which contains specific sections. Each page may include handwritten or printed text under the following headings:

For daily pages:
- "DATE": The date of the page.
- "PREPARE - PRIORITY": The top 3 priorities for the day.
- "TO-DO": Other smaller tasks to complete.
- "I AM GRATEFUL FOR": Things the user is grateful for.
- "I'M LOOKING FORWARD TO": Things the user is looking forward to.
- "HABIT": The habit that the user wants to work on this day.
- "DAILY": Hour-by-hour breakdown of the day.

Your job is to:
- Identify each section based on layout or header.
- Extract the text content written under each section.
- Return a JSON object with a key for each section and its content.
- If a section is not present or has no content, leave it blank or null.

Here is an example of the expected output format for a Monk Manual daily page:

{
  "date": "Monday, Nov 12, 2018",
  "prepare_priority": [
    "Plan upcoming week",
    "Finish monthly report",
    "Catch up with Matt"
  ],
  "to_do": [
    "Check the AM news",
    "Check AM emails",
    "Read afternoon updates",
    "Respond to afternoon emails",
    "Tie up loose ends",
    "Reflect and prepare"
  ],
  "i_am_grateful_for": [
    "Warm weather",
    "Coffee",
    "Madeline"
  ],
  "i_am_looking_forward_to": [
    "Catching up with Matt"
  ],
  "habit": "Leave work at work",
  "daily": [
    {
      "hour": 6,
      "activities": [
        "Check the news"
      ]
    },
    {
      "hour": 7,
      "activities": [
        "Get ready for work"
      ]
    },
    {
      "hour": 8,
      "activities": [
        "Walk to work"
      ]
    },
    {
      "hour": 9,
      "activities": [
        "Check emails"
      ]
    },
    {
      "hour": 10,
      "activities": [
        "Look to week ahead",
        "Finish monthly report"
      ]
    },
    {
      "hour": 11,
      "activities": []
    },
    {
      "hour": 12,
      "activities": [
        "Lunch"
      ]
    },
    {
      "hour": 13,
      "activities": [
        "Meeting with team"
      ]
    },
    {
      "hour": 14,
      "activities": [
        "Read status updates"
      ]
    },
    {
      "hour": 15,
      "activities": [
        "Respond to emails"
      ]
    },
    {
      "hour": 16,
      "activities": [
        "Staff meeting"
      ]
    },
    {
      "hour": 17,
      "activities": [
        "Tie up day's loose ends"
      ]
    },
    {
      "hour": 18,
      "activities": [
        "Walk home"
      ]
    },
    {
      "hour": 19,
      "activities": [
        "Dinner/call Matt"
      ]
    },
    {
      "hour": 20,
      "activities": [
        "Reflect/Prepare"
      ]
    },
    {
      "hour": 21,
      "activities": [
        "Get ready for bed"
      ]
    },
    {
      "hour": 22,
      "activities": [
        "Bed"
      ]
    }
  ]
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
                            "type": "string",
                            "items": { "type": "string" },
                            "description": "The date of the page"
                        },
                        "prepare_priority": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "The three biggest priorities for the day"
                        },
                        "to_do": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Additional lower-prioritytasks for the day"
                        },
                        "i_am_grateful_for": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Things the user is grateful for today"
                        },
                        "i_am_looking_forward_to": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Things to look forward to today"
                        },
                        "habit": {
                            "type": "string",
                            "description": "Habit to focus on today"
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
                                        "items": { "type": "string" },
                                        "description": "Activities for this hour"
                                    }
                                },
                                "required": ["hour", "activities"]
                            },
                            "description": "Hourly breakdown of the day as tuples of [hour, activities]"
                        },
                    },
                    "required": ["date"]
                }
            }
        }]

        try:
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
                return json.dumps(json.loads(tool_call.function.arguments), indent=2)
            else:
                return content.strip() if content else ""
            
        except Exception as e:
            return f"Error processing image: {str(e)}" 