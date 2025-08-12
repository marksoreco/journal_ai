"""Gmail client for email operations."""

import base64
import email
import html
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from pydantic import BaseModel
from .auth import get_gmail_service

logger = logging.getLogger(__name__)


class EmailMessage(BaseModel):
    """Email message data structure."""
    id: str
    thread_id: str
    subject: str
    sender: str
    recipient: str
    date: datetime
    body: str
    snippet: str
    labels: List[str]


class GmailClient:
    """Client for Gmail API operations."""
    
    def __init__(self):
        """Initialize Gmail client."""
        self.service = None
    
    def _get_service(self):
        """Get Gmail service instance."""
        if not self.service:
            self.service = get_gmail_service()
        return self.service
    
    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities in text."""
        if not text:
            return text
        return html.unescape(text)
    
    def _clean_invisible_characters(self, text: str) -> str:
        """Remove invisible Unicode characters and excessive whitespace."""
        if not text:
            return text
        
        # Remove various invisible/zero-width characters
        invisible_chars = [
            '\u200B',  # Zero Width Space
            '\u200C',  # Zero Width Non-Joiner
            '\u200D',  # Zero Width Joiner
            '\u2060',  # Word Joiner
            '\uFEFF',  # Zero Width No-Break Space (BOM)
            '\u00AD',  # Soft Hyphen
            '\u034F',  # Combining Grapheme Joiner
            '\u061C',  # Arabic Letter Mark
            '\u180E',  # Mongolian Vowel Separator
            '\u17B4',  # Khmer Vowel Inherent Aq
            '\u17B5',  # Khmer Vowel Inherent Aa
        ]
        
        # Remove invisible characters
        for char in invisible_chars:
            text = text.replace(char, '')
        
        # Remove other problematic Unicode characters (including the ones in your example)
        # This covers a broader range of invisible/formatting characters
        text = re.sub(r'[\u00A0\u1680\u2000-\u200F\u2028-\u202F\u205F-\u206F\u3000\uFEFF]', ' ', text)
        
        # Clean up excessive whitespace
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n +', '\n', text)  # Remove spaces after newlines
        text = re.sub(r' +\n', '\n', text)  # Remove spaces before newlines
        
        return text.strip()
    
    def _clean_urls_from_text(self, text: str) -> str:
        """Clean or shorten URLs and remove HTML documents from email text to reduce clutter."""
        if not text:
            return text
        
        original_text = text
        html_removed = False
        
        # Check if HTML document exists and remove it
        if re.search(r'<!DOCTYPE[^>]*>.*?</html>', text, flags=re.DOTALL | re.IGNORECASE):
            text = re.sub(r'<!DOCTYPE[^>]*>.*?</html>', '', text, flags=re.DOTALL | re.IGNORECASE)
            html_removed = True
        
        # Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Pattern to match URLs (http/https, ftp, etc.)
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|ftp://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
        
        def replace_url(match):
            url = match.group(0)
            
            # If URL is very long (>40 chars), replace with placeholder
            if len(url) > 40:
                # Extract domain for context
                domain_match = re.search(r'https?://([^/]+)', url)
                if domain_match:
                    domain = domain_match.group(1)
                    return f"[URL: {domain}]"
                else:
                    return "[LONG_URL]"
            
            # Keep shorter URLs as they might be more relevant
            return url
        
        # Replace URLs with placeholders or shortened versions
        cleaned_text = re.sub(url_pattern, replace_url, text, flags=re.IGNORECASE)
        
        # Clean up excessive whitespace that might result from URL removal
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)  # Multiple blank lines
        cleaned_text = re.sub(r' {3,}', ' ', cleaned_text)  # Multiple spaces
        cleaned_text = cleaned_text.strip()
        
        # If HTML was removed and result is empty or mostly whitespace, use placeholder
        if html_removed and (not cleaned_text or len(cleaned_text.strip()) < 10):
            return "<!DOCTYPE html>...</html>"
        
        return cleaned_text
    
    def get_emails_since_date(self, since_date: datetime, max_results: int = 100, 
                             query: str = "") -> List[EmailMessage]:
        """
        Get emails since a specified date.
        
        Args:
            since_date: Get emails from this date onwards
            max_results: Maximum number of emails to retrieve
            query: Additional Gmail search query parameters
            
        Returns:
            List of EmailMessage objects
        """
        try:
            service = self._get_service()
            
            # Format date for Gmail API query
            date_str = since_date.strftime("%Y/%m/%d")
            search_query = f"after:{date_str}"
            
            if query:
                search_query += f" {query}"
            
            logger.info(f"Searching for emails with query: {search_query}")
            
            # Get list of message IDs
            result = service.users().messages().list(
                userId='me',
                q=search_query,
                maxResults=max_results
            ).execute()
            
            messages = result.get('messages', [])
            logger.info(f"Found {len(messages)} emails")
            
            # Fetch detailed message data
            email_messages = []
            for msg in messages:
                try:
                    email_msg = self._get_message_details(service, msg['id'])
                    if email_msg:
                        email_messages.append(email_msg)
                except Exception as e:
                    logger.warning(f"Failed to fetch message {msg['id']}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(email_messages)} emails")
            return email_messages
            
        except Exception as e:
            logger.error(f"Error fetching emails: {str(e)}")
            raise
    
    def _get_message_details(self, service, message_id: str) -> Optional[EmailMessage]:
        """Get detailed information for a specific message."""
        try:
            message = service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()
            
            # Extract headers
            headers = {h['name']: h['value'] for h in message['payload'].get('headers', [])}
            
            # Extract body
            body = self._extract_body(message['payload'])
            
            # Parse date
            date_str = headers.get('Date', '')
            try:
                # Parse email date format
                parsed_date = email.utils.parsedate_to_datetime(date_str)
            except Exception:
                parsed_date = datetime.now()
            
            return EmailMessage(
                id=message_id,
                thread_id=message.get('threadId', ''),
                subject=self._decode_html_entities(headers.get('Subject', 'No Subject')),
                sender=self._decode_html_entities(headers.get('From', 'Unknown Sender')),
                recipient=self._decode_html_entities(headers.get('To', 'Unknown Recipient')),
                date=parsed_date,
                body=self._clean_urls_from_text(self._decode_html_entities(body)),
                snippet=self._clean_invisible_characters(self._decode_html_entities(message.get('snippet', ''))),
                labels=message.get('labelIds', [])
            )
            
        except Exception as e:
            logger.error(f"Error getting message details for {message_id}: {str(e)}")
            return None
    
    def _extract_body(self, payload: Dict[str, Any]) -> str:
        """Extract body text from message payload."""
        body = ""
        
        if 'parts' in payload:
            # Multi-part message
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                elif part['mimeType'] == 'text/html' and not body:
                    # Use HTML if no plain text found
                    if 'data' in part['body']:
                        body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
        else:
            # Single part message
            if payload['mimeType'] in ['text/plain', 'text/html']:
                if 'data' in payload.get('body', {}):
                    body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        return body.strip()
    
    def search_emails(self, query: str, max_results: int = 50) -> List[EmailMessage]:
        """
        Search emails with a custom query.
        
        Args:
            query: Gmail search query
            max_results: Maximum number of results
            
        Returns:
            List of EmailMessage objects
        """
        try:
            service = self._get_service()
            
            result = service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = result.get('messages', [])
            
            email_messages = []
            for msg in messages:
                try:
                    email_msg = self._get_message_details(service, msg['id'])
                    if email_msg:
                        email_messages.append(email_msg)
                except Exception as e:
                    logger.warning(f"Failed to fetch message {msg['id']}: {str(e)}")
                    continue
            
            return email_messages
            
        except Exception as e:
            logger.error(f"Error searching emails: {str(e)}")
            raise