"""
Simple in-memory session manager for chat sessions
In production, this should use Redis or a database
"""
from typing import Dict, Optional
from .models import ChatSession, ChatMessage
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID"""
        return self._sessions.get(session_id)
    
    def create_session(self) -> ChatSession:
        """Create a new chat session"""
        session = ChatSession()
        self._sessions[session.session_id] = session
        logger.info(f"Created new chat session: {session.session_id}")
        return session
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """Get existing session or create new one"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        # Create new session
        return self.create_session()
    
    def add_message_to_session(self, session_id: str, message: ChatMessage) -> bool:
        """Add a message to a session"""
        session = self.get_session(session_id)
        if session:
            session.add_message(message)
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted chat session: {session_id}")
            return True
        return False
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old inactive sessions"""
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        old_sessions = [
            session_id for session_id, session in self._sessions.items()
            if session.last_activity < cutoff
        ]
        
        for session_id in old_sessions:
            self.delete_session(session_id)
        
        if old_sessions:
            logger.info(f"Cleaned up {len(old_sessions)} old chat sessions")

# Global session manager instance
session_manager = SessionManager()