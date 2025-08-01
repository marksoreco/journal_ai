import unittest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestBasicSetup(unittest.TestCase):
    """Basic tests to verify test setup works"""
    
    def test_imports_work(self):
        """Test that we can import the main modules"""
        try:
            # Test importing config
            from config import OCR_ENGINE, TASK_CONFIDENCE_THRESHOLD  # type: ignore
            self.assertIsInstance(OCR_ENGINE, str)
            self.assertIsInstance(TASK_CONFIDENCE_THRESHOLD, float)
        except ImportError as e:
            self.fail(f"Failed to import config: {e}")
    
    def test_config_values(self):
        """Test that config values are reasonable"""
        from config import OCR_ENGINE, TASK_CONFIDENCE_THRESHOLD  # type: ignore
        
        # Test OCR_ENGINE is a valid string
        self.assertIsInstance(OCR_ENGINE, str)
        self.assertGreater(len(OCR_ENGINE), 0)
        
        # Test TASK_CONFIDENCE_THRESHOLD is in valid range
        self.assertIsInstance(TASK_CONFIDENCE_THRESHOLD, float)
        self.assertGreaterEqual(TASK_CONFIDENCE_THRESHOLD, 0.0)
        self.assertLessEqual(TASK_CONFIDENCE_THRESHOLD, 1.0)
    
    def test_path_setup(self):
        """Test that the path setup works correctly"""
        # Verify we can access the src directory
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        self.assertTrue(os.path.exists(src_path))
        
        # Verify we can access main.py
        main_path = os.path.join(src_path, 'main.py')
        self.assertTrue(os.path.exists(main_path))


if __name__ == '__main__':
    unittest.main() 