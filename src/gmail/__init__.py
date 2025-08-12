"""Gmail API package for authentication and email access."""

from .auth import get_gmail_service, OAUTH_SCOPES
from .client import GmailClient

__all__ = ['get_gmail_service', 'GmailClient', 'OAUTH_SCOPES']