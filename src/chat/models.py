"""
Chat models for session and message management
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel
from datetime import datetime
import uuid

class ChatMessage(BaseModel):
    id: str = None
    type: Literal["user", "assistant", "system", "function_call", "function_result"] = "user"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __init__(self, **data):
        if not data.get('id'):
            data['id'] = str(uuid.uuid4())
        if not data.get('timestamp'):
            data['timestamp'] = datetime.now()
        super().__init__(**data)

class ChatSession(BaseModel):
    session_id: str = None
    conversation_history: List[ChatMessage] = []
    uploaded_files: Dict[str, Dict[str, Any]] = {}
    processing_states: Dict[str, str] = {}
    created_at: datetime = None
    last_activity: datetime = None
    
    def __init__(self, **data):
        if not data.get('session_id'):
            data['session_id'] = str(uuid.uuid4())
        if not data.get('created_at'):
            data['created_at'] = datetime.now()
        if not data.get('last_activity'):
            data['last_activity'] = datetime.now()
        super().__init__(**data)
    
    def add_message(self, message: ChatMessage):
        """Add a message to the conversation history"""
        self.conversation_history.append(message)
        self.last_activity = datetime.now()
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent messages for context"""
        return self.conversation_history[-limit:]
    
    def set_low_confidence_review_state(self, low_confidence_items: List[Dict[str, Any]], ocr_data: Dict[str, Any]):
        """Set up state for reviewing low-confidence items"""
        self.processing_states['low_confidence_review'] = {
            'items': low_confidence_items,
            'current_index': 0,
            'reviewed_items': [],
            'ocr_data': ocr_data,
            'active': True
        }
    
    def get_low_confidence_review_state(self) -> Optional[Dict[str, Any]]:
        """Get the current low-confidence review state"""
        return self.processing_states.get('low_confidence_review')
    
    def update_low_confidence_review(self, action: str, edited_text: Optional[str] = None):
        """Update the current low-confidence item review"""
        state = self.get_low_confidence_review_state()
        if not state or not state.get('active'):
            return False
        
        current_item = state['items'][state['current_index']]
        
        if action == 'keep':
            # Keep the original item
            state['reviewed_items'].append(current_item)
        elif action == 'edit' and edited_text:
            # Keep the item but with edited text
            edited_item = current_item.copy()
            edited_item['text'] = edited_text
            edited_item['confidence'] = 1.0  # Mark as high confidence since user reviewed
            state['reviewed_items'].append(edited_item)
        elif action == 'skip':
            # Skip this item (don't add to reviewed_items)
            pass
        
        # Move to next item
        state['current_index'] += 1
        
        # Check if we're done
        if state['current_index'] >= len(state['items']):
            state['active'] = False
            return True  # Review is complete
        
        return False  # Review continues
    
    def clear_low_confidence_review_state(self):
        """Clear the low-confidence review state"""
        if 'low_confidence_review' in self.processing_states:
            del self.processing_states['low_confidence_review']

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    
class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str
    metadata: Optional[Dict[str, Any]] = None