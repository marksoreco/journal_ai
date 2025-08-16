import os
import logging
from todoist_api_python.api import TodoistAPI
from typing import List, Dict, Any
import dateparser

try:
    from .sbert_client import SBERTClient
    from .config import SBERT_ENABLED, SBERT_MODEL, SBERT_SIMILARITY_THRESHOLD, SBERT_CACHE_FILE
except ImportError:
    # Fallback for when running as script
    from sbert_client import SBERTClient
    from config import SBERT_ENABLED, SBERT_MODEL, SBERT_SIMILARITY_THRESHOLD, SBERT_CACHE_FILE

# Configure logger for this module
logger = logging.getLogger(__name__)

class TodoistClient:
    def __init__(self, use_sbert: bool | None = None):
        """
        Initialize Todoist client with API token from environment
        
        Args:
            use_sbert: Whether to use SBERT for intelligent duplicate detection
                      If None, uses SBERT_ENABLED from config
        """
        self.api_token = os.getenv('TODOIST_API_TOKEN')
        if not self.api_token:
            raise ValueError("TODOIST_API_TOKEN environment variable is required")
        
        self.api = TodoistAPI(self.api_token)
        
        # Initialize SBERT client for intelligent duplicate detection
        self.use_sbert = use_sbert if use_sbert is not None else SBERT_ENABLED
        self.sbert_client = None  # Initialize to None first
        
        if self.use_sbert:
            try:
                self.sbert_client = SBERTClient(
                    model_name=SBERT_MODEL,
                    cache_file=SBERT_CACHE_FILE,
                    similarity_threshold=SBERT_SIMILARITY_THRESHOLD
                )
            except Exception as e:
                # Fallback to simple text comparison if SBERT fails
                self.use_sbert = False
                self.sbert_client = None

    def get_existing_tasks(self, due_date: str = "today") -> List[str]:
        """
        Get all existing tasks from Todoist
        
        Returns:
            List of existing task content strings
        """
        try:
            logger.debug("Fetching existing tasks from Todoist")
            # Get tasks and convert to list once
            tasks_iterator = self.api.get_tasks()
            tasks_list = list(tasks_iterator)  # Convert iterator to list once
            
            # Extract task content - the API seems to return a list where the first element contains all tasks
            existing_tasks = []
            
            # Check if the first element contains the actual task list
            if tasks_list and isinstance(tasks_list[0], list):
                actual_tasks = tasks_list[0]
                for task in actual_tasks:
                    content = getattr(task, 'content', '')
                    if content and content.strip():
                        existing_tasks.append(content.strip())
            else:
                # Handle normal case where each element is a task
                for task in tasks_list:
                    content = getattr(task, 'content', '')
                    due = getattr(task, 'due', '')
                    task_due_date = getattr(due, 'date', '')
                    if content and content.strip() and task_due_date == due_date:
                        existing_tasks.append(content.strip())
            
            logger.info(f"Retrieved {len(existing_tasks)} existing tasks from Todoist")
            return existing_tasks
        except Exception as e:
            logger.error(f"Failed to fetch existing tasks from Todoist: {str(e)}")
            # Return empty list if API call fails
            return []
    
    def check_duplicates_intelligently(self, new_tasks: List[str], due_date: str) -> Dict[str, bool]:
        """
        Check for duplicates using SBERT if available, otherwise fallback to simple comparison
        
        Args:
            new_tasks: List of new task strings to check
            
        Returns:
            Dict mapping task to boolean (True if duplicate, False if unique)
        """
        if not self.use_sbert or not self.sbert_client:
            # Fallback to simple text comparison
            existing_tasks = self.get_existing_tasks()
            return self._simple_duplicate_check(new_tasks, existing_tasks)
        
        try:
            # Use SBERT for intelligent duplicate detection
            existing_tasks = self.get_existing_tasks()
            return self.sbert_client.check_duplicate_tasks(new_tasks, existing_tasks, due_date)
        except Exception as e:
            # Fallback to simple text comparison if SBERT fails
            existing_tasks = self.get_existing_tasks()
            return self._simple_duplicate_check(new_tasks, existing_tasks)
    
    def _simple_duplicate_check(self, new_tasks: List[str], existing_tasks: List[str]) -> Dict[str, bool]:
        """
        Simple case-insensitive text comparison for duplicate detection
        
        Args:
            new_tasks: List of new task strings
            existing_tasks: List of existing task strings
            
        Returns:
            Dict mapping task to boolean (True if duplicate, False if unique)
        """
        duplicate_results = {}
        
        for new_task in new_tasks:
            is_duplicate = False
            
            for existing_task in existing_tasks:
                if new_task.lower().strip() == existing_task.lower().strip():
                    is_duplicate = True
                    break
            
            duplicate_results[new_task] = is_duplicate
        
        return duplicate_results

    def create_task(self, content: str, priority: int = 3, due_string: str = "today") -> Dict[str, Any]:
        """
        Create a new task in Todoist
        
        Args:
            content: Task content
            priority: Task priority (1=high, 3=normal, 4=low)
            due_string: Due date string
            
        Returns:
            Dict containing task information
        """
        try:
            new_task = self.api.add_task(
                content=content,
                priority=priority,
                due_string=due_string
            )
            return {
                "content": content,
                "priority": "High" if priority == 1 else "Normal" if priority == 3 else "Low",
                "id": new_task.id
            }
        except Exception as e:
            raise Exception(f"Failed to create task '{content}': {str(e)}")

    def _parse_date_for_todoist(self, date_string: str) -> str:
        """
        Parse date string from OCR data and convert to Todoist-compatible format
        
        Args:
            date_string: Date string from OCR (e.g., "Monday, Nov 12, 2018")
            
        Returns:
            Todoist-compatible date string
        """
        try:
            parsed_date = dateparser.parse(date_string)
            if parsed_date:
                return parsed_date.strftime('%Y-%m-%d')
            else:
                return "today"
                    
        except Exception as e:
            logger.error(f"Error parsing date '{date_string}': {str(e)}, using 'today' instead")
            return "today"

    def upload_tasks_from_ocr(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upload tasks from OCR data to Todoist with intelligent duplicate checking
        
        Args:
            task_data: Dictionary containing OCR extracted task data
            
        Returns:
            Dict containing upload results and statistics
        """
        logger.info("Starting task upload from OCR data")
        tasks_created = []
        
        # Extract date from task_data for due date
        due_date = "today"  # Default fallback
        if 'date' in task_data and task_data['date']:
            due_date = self._parse_date_for_todoist(task_data['date']['value'])
            logger.info(f"Using date from OCR data: {due_date}")
        
        # Extract all new tasks
        new_tasks = []
        task_metadata = {}  # Store task type and priority for later use
        
        # Collect priority tasks
        if 'prepare_priority' in task_data and task_data['prepare_priority']:
            for task in task_data['prepare_priority']:
                task_content = task if isinstance(task, str) else task.get('task', '')
                if task_content.strip():
                    new_tasks.append(task_content)
                    task_metadata[task_content] = {'type': 'priority', 'priority': 1}
        
        # Collect regular to-do tasks
        if 'to_do' in task_data and task_data['to_do']:
            for task in task_data['to_do']:
                # to_do items use 'item' field, not 'task' field
                task_content = task if isinstance(task, str) else (task.get('task', '') or task.get('item', ''))
                if task_content.strip():
                    new_tasks.append(task_content)
                    task_metadata[task_content] = {'type': 'todo', 'priority': 3}
        
        logger.info(f"Extracted {len(new_tasks)} tasks from OCR data")
        
        if not new_tasks:
            logger.info("No tasks found in OCR data")
            return {
                "message": "No tasks to upload",
                "tasks_created": [],
                "total_tasks": 0,
                "created_count": 0,
                "skipped_count": 0
            }
        
        # Check for duplicates using intelligent detection
        logger.info("Checking for duplicate tasks")
        duplicate_results = self.check_duplicates_intelligently(new_tasks, due_date)
        
        # Process each task based on duplicate check results
        for task_content in new_tasks:
            metadata = task_metadata[task_content]
            is_duplicate = duplicate_results.get(task_content, False)
            
            if is_duplicate:
                # Task is a duplicate, skip it
                logger.debug(f"Skipping duplicate task: {task_content}")
                tasks_created.append({
                    "content": task_content,
                    "priority": "High" if metadata['priority'] == 1 else "Normal",
                    "status": "skipped - duplicate detected"
                })
            else:
                # Task is unique, create it
                logger.debug(f"Creating new task: {task_content} with due date: {due_date}")
                try:
                    new_task = self.create_task(task_content, priority=metadata['priority'], due_string=due_date)
                    tasks_created.append(new_task)
                except Exception as e:
                    logger.error(f"Failed to create task '{task_content}': {str(e)}")
                    tasks_created.append({
                        "content": task_content,
                        "priority": "High" if metadata['priority'] == 1 else "Normal",
                        "status": f"failed - {str(e)}"
                    })
        
        # Count created vs skipped tasks
        created_count = len([t for t in tasks_created if 'id' in t])
        skipped_count = len([t for t in tasks_created if 'status' in t and 'skipped' in t['status']])
        failed_count = len([t for t in tasks_created if 'status' in t and 'failed' in t['status']])
        
        message = f"Created {created_count} new tasks, skipped {skipped_count} duplicates"
        if failed_count > 0:
            message += f", {failed_count} failed"
        
        logger.info(f"Task upload completed: {message}")
        
        return {
            "message": message,
            "tasks_created": tasks_created,
            "total_tasks": len(tasks_created),
            "created_count": created_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count
        } 