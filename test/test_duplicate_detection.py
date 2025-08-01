import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Type ignore for linter - these imports work at runtime
from todoist_client import TodoistClient  # type: ignore
from sbert_client import SBERTClient  # type: ignore


class TestDuplicateDetection(unittest.TestCase):
    """Test cases for intelligent duplicate detection functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'TODOIST_API_TOKEN': 'test_token'
        })
        self.env_patcher.start()
        
        # Mock Todoist API
        self.todoist_api_patcher = patch('todoist_client.TodoistAPI')
        self.mock_todoist_api = self.todoist_api_patcher.start()
        
        # Create mock Todoist tasks with semantically similar variations
        self.mock_tasks = [
            Mock(content="Check emails"),
            Mock(content="Review project proposal"),
            Mock(content="Call client"),
            Mock(content="Prepare presentation"),
            Mock(content="Send follow-up email"),
            Mock(content="Schedule team meeting"),
            Mock(content="Update project documentation"),
            Mock(content="Review quarterly budget"),
            Mock(content="Contact customer support"),
            Mock(content="Complete project setup"),
            Mock(content="Review Q1 results"),
            Mock(content="")  # Empty task to test filtering
        ]
        
        # Set up mock API response
        self.mock_todoist_api.return_value.get_tasks.return_value = self.mock_tasks
        
    def tearDown(self):
        """Clean up after tests"""
        self.env_patcher.stop()
        self.todoist_api_patcher.stop()
    
    def test_todoist_client_initialization_with_sbert(self):
        """Test TodoistClient initialization with SBERT enabled"""
        with patch('todoist_client.SBERTClient') as mock_sbert_class:
            mock_sbert_instance = Mock()
            mock_sbert_class.return_value = mock_sbert_instance
            
            client = TodoistClient(use_sbert=True)
            
            self.assertTrue(client.use_sbert)
            self.assertIsNotNone(client.sbert_client)
            mock_sbert_class.assert_called_once()
    
    def test_todoist_client_initialization_without_sbert(self):
        """Test TodoistClient initialization with SBERT disabled"""
        client = TodoistClient(use_sbert=False)
        
        self.assertFalse(client.use_sbert)
        self.assertIsNone(client.sbert_client)
    
    def test_todoist_client_initialization_sbert_failure(self):
        """Test TodoistClient initialization when SBERT fails to initialize"""
        with patch('todoist_client.SBERTClient', side_effect=Exception("SBERT not available")):
            client = TodoistClient(use_sbert=True)
            
            self.assertFalse(client.use_sbert)
            self.assertIsNone(client.sbert_client)
    
    def test_get_existing_tasks(self):
        """Test getting existing tasks from Todoist"""
        client = TodoistClient(use_sbert=False)
        
        existing_tasks = client.get_existing_tasks()
        
        expected_tasks = [
            "Check emails", "Review project proposal", "Call client", "Prepare presentation",
            "Send follow-up email", "Schedule team meeting", "Update project documentation",
            "Review quarterly budget", "Contact customer support", "Complete project setup",
            "Review Q1 results"
        ]
        self.assertEqual(existing_tasks, expected_tasks)
    
    def test_get_existing_tasks_api_error(self):
        """Test getting existing tasks when API fails"""
        # Make the API call fail
        self.mock_todoist_api.return_value.get_tasks.side_effect = Exception("API Error")
        
        client = TodoistClient(use_sbert=False)
        existing_tasks = client.get_existing_tasks()
        
        self.assertEqual(existing_tasks, [])
    
    def test_simple_duplicate_check(self):
        """Test simple case-insensitive duplicate checking"""
        client = TodoistClient(use_sbert=False)
        
        new_tasks = ["Check emails", "New task", "CALL CLIENT", "Another task", "Send email"]
        existing_tasks = ["Check emails", "Call client", "Prepare presentation"]
        
        results = client._simple_duplicate_check(new_tasks, existing_tasks)
        
        expected = {
            "Check emails": True,      # Exact match (case insensitive)
            "New task": False,         # No match
            "CALL CLIENT": True,       # Exact match (case insensitive)
            "Another task": False,     # No match
            "Send email": False        # No match
        }
        self.assertEqual(results, expected)
    
    def test_check_duplicates_intelligently_with_sbert(self):
        """Test intelligent duplicate detection using SBERT with semantically similar tasks"""
        with patch('todoist_client.SBERTClient') as mock_sbert_class:
            mock_sbert_instance = Mock()
            mock_sbert_class.return_value = mock_sbert_instance
            
            # Mock SBERT response - should detect semantic similarities
            mock_sbert_instance.check_duplicate_tasks.return_value = {
                "Check emails": True,           # Exact match
                "Read emails": True,            # Semantically similar
                "Send email": True,             # Semantically similar to "Send follow-up email"
                "Email client": True,           # Semantically similar
                "Schedule meeting": True,       # Semantically similar to "Schedule team meeting"
                "Plan team meeting": True,      # Semantically similar
                "Update docs": True,            # Semantically similar to "Update project documentation"
                "Call support": True,           # Semantically similar to "Contact customer support"
                "New unique task": False,       # Truly unique
                "Review budget": True,          # Semantically similar to "Review quarterly budget"
                "Prepare slides": True,         # Semantically similar to "Prepare presentation"
                "Check emails!": True,          # Punctuation difference
                "Check emails...": True,        # Punctuation difference
                "Check emails?": True,          # Punctuation difference
                "Complete proj setup": True,    # Abbreviation difference
                "Review Q1 results": True,      # Abbreviation match
                "Review Q1 res": True,          # Abbreviation difference
                "Setup project": True,          # Word order difference
                "Project setup": True,          # Word order difference
                "Setup proj": True              # Abbreviation + word order
            }
            
            client = TodoistClient(use_sbert=True)
            client.sbert_client = mock_sbert_instance
            
            new_tasks = [
                "Check emails", "Read emails", "Send email", "Email client",
                "Schedule meeting", "Plan team meeting", "Update docs",
                "Call support", "New unique task", "Review budget", "Prepare slides",
                "Check emails!", "Check emails...", "Check emails?",
                "Complete proj setup", "Review Q1 results", "Review Q1 res",
                "Setup project", "Project setup", "Setup proj"
            ]
            results = client.check_duplicates_intelligently(new_tasks)
            
            expected = {
                "Check emails": True, "Read emails": True, "Send email": True,
                "Email client": True, "Schedule meeting": True, "Plan team meeting": True,
                "Update docs": True, "Call support": True, "New unique task": False,
                "Review budget": True, "Prepare slides": True, "Check emails!": True,
                "Check emails...": True, "Check emails?": True, "Complete proj setup": True,
                "Review Q1 results": True, "Review Q1 res": True, "Setup project": True,
                "Project setup": True, "Setup proj": True
            }
            self.assertEqual(results, expected)
            mock_sbert_instance.check_duplicate_tasks.assert_called_once()
    
    def test_check_duplicates_intelligently_sbert_failure(self):
        """Test intelligent duplicate detection when SBERT fails"""
        with patch('todoist_client.SBERTClient') as mock_sbert_class:
            mock_sbert_instance = Mock()
            mock_sbert_class.return_value = mock_sbert_instance
            
            # Make SBERT call fail
            mock_sbert_instance.check_duplicate_tasks.side_effect = Exception("SBERT error")
            
            client = TodoistClient(use_sbert=True)
            client.sbert_client = mock_sbert_instance
            
            new_tasks = ["Check emails", "New task"]
            results = client.check_duplicates_intelligently(new_tasks)
            
            # Should fall back to simple comparison
            self.assertIn("Check emails", results)
            self.assertIn("New task", results)
    
    def test_check_duplicates_intelligently_no_sbert(self):
        """Test intelligent duplicate detection without SBERT"""
        client = TodoistClient(use_sbert=False)
        
        new_tasks = ["Check emails", "New task"]
        results = client.check_duplicates_intelligently(new_tasks)
        
        # Should use simple comparison
        self.assertIn("Check emails", results)
        self.assertIn("New task", results)
    
    def test_upload_tasks_from_ocr_with_duplicates(self):
        """Test uploading tasks with intelligent duplicate detection"""
        with patch('todoist_client.SBERTClient') as mock_sbert_class:
            mock_sbert_instance = Mock()
            mock_sbert_class.return_value = mock_sbert_instance
            
            # Mock SBERT to identify semantic duplicates
            mock_sbert_instance.check_duplicate_tasks.return_value = {
                "Check emails": True,           # Exact duplicate
                "Read emails": True,            # Semantic duplicate
                "Send follow-up": True,         # Semantic duplicate of "Send follow-up email"
                "Schedule meeting": True,       # Semantic duplicate of "Schedule team meeting"
                "New unique task": False,       # Unique task
                "Another unique task": False    # Unique task
            }
            
            # Mock task creation
            mock_task = Mock()
            mock_task.id = "12345"
            self.mock_todoist_api.return_value.add_task.return_value = mock_task
            
            client = TodoistClient(use_sbert=True)
            client.sbert_client = mock_sbert_instance
            
            task_data = {
                "prepare_priority": ["Check emails", "New unique task"],
                "to_do": ["Read emails", "Send follow-up", "Schedule meeting", "Another unique task"]
            }
            
            result = client.upload_tasks_from_ocr(task_data)
            
            # Verify results
            self.assertEqual(result["created_count"], 2)  # Only unique tasks created
            self.assertEqual(result["skipped_count"], 4)  # 4 duplicates skipped
            
            # Verify only unique tasks were created
            created_tasks = [task for task in result["tasks_created"] if "id" in task]
            self.assertEqual(len(created_tasks), 2)
            
            # Verify duplicates were marked as skipped
            skipped_tasks = [task for task in result["tasks_created"] if "skipped" in task.get("status", "")]
            self.assertEqual(len(skipped_tasks), 4)
    
    def test_upload_tasks_from_ocr_no_tasks(self):
        """Test uploading when no tasks are provided"""
        client = TodoistClient(use_sbert=False)
        
        task_data = {}
        result = client.upload_tasks_from_ocr(task_data)
        
        self.assertEqual(result["message"], "No tasks to upload")
        self.assertEqual(result["total_tasks"], 0)
        self.assertEqual(result["created_count"], 0)
    
    def test_upload_tasks_from_ocr_task_creation_failure(self):
        """Test uploading when task creation fails"""
        with patch('todoist_client.SBERTClient') as mock_sbert_class:
            mock_sbert_instance = Mock()
            mock_sbert_class.return_value = mock_sbert_instance
            
            # Mock SBERT to say no duplicates
            mock_sbert_instance.check_duplicate_tasks.return_value = {
                "New task": False,
                "Another task": False
            }
            
            # Make task creation fail
            self.mock_todoist_api.return_value.add_task.side_effect = Exception("API Error")
            
            client = TodoistClient(use_sbert=True)
            client.sbert_client = mock_sbert_instance
            
            task_data = {
                "to_do": ["New task", "Another task"]
            }
            
            result = client.upload_tasks_from_ocr(task_data)
            
            # Verify results
            self.assertEqual(result["created_count"], 0)
            self.assertEqual(result["failed_count"], 2)
            
            # Verify failed tasks were marked
            failed_tasks = [task for task in result["tasks_created"] if "failed" in task.get("status", "")]
            self.assertEqual(len(failed_tasks), 2)


class TestSBERTClient(unittest.TestCase):
    """Test cases for SBERTClient functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the SentenceTransformer to avoid loading actual model
        self.sbert_patcher = patch('sbert_client.SentenceTransformer')
        self.mock_sbert = self.sbert_patcher.start()
        self.mock_model = Mock()
        self.mock_sbert.return_value = self.mock_model
        
        # Mock numpy and sklearn
        self.numpy_patcher = patch('sbert_client.np')
        self.mock_np = self.numpy_patcher.start()
        
        self.sklearn_patcher = patch('sbert_client.cosine_similarity')
        self.mock_cosine_similarity = self.sklearn_patcher.start()
        
        # Create SBERT client instance
        self.sbert_client = SBERTClient(model_name="test-model", cache_file="test_cache.pkl")
    
    def tearDown(self):
        """Clean up after tests"""
        self.sbert_patcher.stop()
        self.numpy_patcher.stop()
        self.sklearn_patcher.stop()
        
        # Clean up test cache file
        if os.path.exists("test_cache.pkl"):
            os.remove("test_cache.pkl")
    
    def test_sbert_client_initialization(self):
        """Test SBERTClient initialization"""
        client = SBERTClient()
        self.assertIsNotNone(client.model)
        self.assertEqual(client.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(client.similarity_threshold, 0.85)
    
    def test_check_duplicate_tasks_no_existing_tasks(self):
        """Test duplicate checking when no existing tasks"""
        new_tasks = ["Task 1", "Task 2", "Task 3"]
        existing_tasks = []
        
        results = self.sbert_client.check_duplicate_tasks(new_tasks, existing_tasks)
        
        expected = {"Task 1": False, "Task 2": False, "Task 3": False}
        self.assertEqual(results, expected)
    
    def test_check_duplicate_tasks_no_new_tasks(self):
        """Test duplicate checking when no new tasks"""
        new_tasks = []
        existing_tasks = ["Existing task"]
        
        results = self.sbert_client.check_duplicate_tasks(new_tasks, existing_tasks)
        
        self.assertEqual(results, {})
    
    def test_check_duplicate_tasks_with_mock_response(self):
        """Test duplicate checking with mock SBERT response"""
        # Mock embeddings for all tasks (existing + new)
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]
        self.mock_model.encode.return_value = mock_embeddings
        
        # Mock similarity matrix (3 new tasks x 2 existing tasks)
        self.mock_cosine_similarity.return_value = [[0.9, 0.3], [0.2, 0.8], [0.1, 0.4]]
        
        # Mock numpy operations
        self.mock_np.array.side_effect = [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # existing_embeddings_array
            [[0.7, 0.8, 0.9], [0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]  # new_embeddings_array
        ]
        self.mock_np.max.side_effect = [0.9, 0.87, 0.4]  # Max similarity for each new task (0.87 >= 0.85)
        self.mock_np.argmax.side_effect = [0, 1, 1]     # Index of most similar task
        
        new_tasks = ["Task 1", "Task 2", "Task 3"]
        existing_tasks = ["Existing 1", "Existing 2"]
        
        results = self.sbert_client.check_duplicate_tasks(new_tasks, existing_tasks)
        
        # Task 1 and Task 2 should be duplicates (similarity >= 0.85)
        # Task 3 should not be duplicate (similarity < 0.85)
        expected = {"Task 1": True, "Task 2": True, "Task 3": False}
        self.assertEqual(results, expected)
    
    def test_check_duplicate_tasks_api_failure(self):
        """Test duplicate checking when SBERT fails"""
        # Make SBERT fail
        self.mock_model.encode.side_effect = Exception("SBERT error")
        
        new_tasks = ["Task 1", "Task 2"]
        existing_tasks = ["Existing task"]
        
        results = self.sbert_client.check_duplicate_tasks(new_tasks, existing_tasks)
        
        # Should fall back to simple comparison
        self.assertIn("Task 1", results)
        self.assertIn("Task 2", results)
    
    def test_embedding_caching(self):
        """Test that embeddings are cached and reused"""
        # Mock embeddings
        mock_embedding = [0.1, 0.2, 0.3]
        self.mock_model.encode.return_value = mock_embedding
        
        text = "Test task"
        
        # First call should compute embedding
        embedding1 = self.sbert_client.get_embedding(text)
        self.assertEqual(self.mock_model.encode.call_count, 1)
        
        # Second call should use cache
        embedding2 = self.sbert_client.get_embedding(text)
        self.assertEqual(self.mock_model.encode.call_count, 1)  # Should not increase
        
        # Should return same embedding
        self.assertEqual(embedding1, embedding2)
    
    def test_batch_embedding_caching(self):
        """Test batch embedding with caching"""
        # Mock embeddings
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        self.mock_model.encode.return_value = mock_embeddings
        
        texts = ["Task 1", "Task 2"]
        
        # First batch call
        embeddings1 = self.sbert_client.get_embeddings_batch(texts)
        self.assertEqual(self.mock_model.encode.call_count, 1)
        
        # Second batch call with same texts
        embeddings2 = self.sbert_client.get_embeddings_batch(texts)
        self.assertEqual(self.mock_model.encode.call_count, 1)  # Should not increase
        
        # Should return same embeddings
        self.assertEqual(embeddings1, embeddings2)
    
    def test_semantic_similarity_examples(self):
        """Test examples of semantically similar tasks that should be detected as duplicates"""
        semantic_duplicates = [
            ("Check emails", "Read emails"),
            ("Send email", "Send follow-up email"),
            ("Email client", "Contact client via email"),
            ("Schedule meeting", "Schedule team meeting"),
            ("Plan meeting", "Organize team meeting"),
            ("Set up meeting", "Arrange meeting"),
            ("Update docs", "Update project documentation"),
            ("Update documentation", "Refresh project docs"),
            ("Maintain docs", "Keep documentation current"),
            ("Call support", "Contact customer support"),
            ("Contact support", "Reach out to support team"),
            ("Get help", "Contact help desk"),
            ("Review budget", "Review quarterly budget"),
            ("Check budget", "Examine budget"),
            ("Budget review", "Financial review"),
            ("Prepare slides", "Prepare presentation"),
            ("Create presentation", "Make slides"),
            ("Build slides", "Develop presentation"),
            ("Check emails!", "Check emails"),
            ("Check emails...", "Check emails"),
            ("Check emails?", "Check emails"),
            ("Check emails.", "Check emails"),
            ("Check emails - urgent", "Check emails"),
            ("Check emails (urgent)", "Check emails"),
            ("Check emails [urgent]", "Check emails"),
            ("Complete proj setup", "Complete project setup"),
            ("Complete project setup", "Complete proj setup"),
            ("Review Q1 res", "Review Q1 results"),
            ("Review Q1 res", "Review Q1 results"),
            ("Setup proj", "Setup project"),
            ("Setup proj", "Project setup"),
            ("Proj setup", "Project setup"),
            ("Proj setup", "Complete project setup"),
            ("Setup project", "Project setup"),
            ("Project setup", "Setup project"),
            ("Setup proj", "Project setup"),
            ("Setup proj", "Complete project setup"),
            ("Complete proj setup!", "Complete project setup"),
            ("Setup proj?", "Project setup"),
            ("Review Q1 res...", "Review Q1 results")
        ]

        for new_task, existing_task in semantic_duplicates:
            results = self.sbert_client._fallback_duplicate_check([new_task], [existing_task])
            self.assertFalse(results[new_task],
                           f"Simple comparison incorrectly detected '{new_task}' as duplicate of '{existing_task}'")

        exact_matches = [
            ("Check emails", "Check emails"),
            ("Schedule meeting", "Schedule meeting"),
            ("Update docs", "Update docs"),
            ("Complete project setup", "Complete project setup"),
            ("Review Q1 results", "Review Q1 results")
        ]

        for new_task, existing_task in exact_matches:
            results = self.sbert_client._fallback_duplicate_check([new_task], [existing_task])
            self.assertTrue(results[new_task],
                          f"Simple comparison failed to detect exact match: '{new_task}'")


if __name__ == '__main__':
    unittest.main() 