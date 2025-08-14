import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestOCRIntegration(unittest.TestCase):
    """Test OCR integration with category parameter"""
    
    def setUp(self):
        """Set up test environment"""
        from ocr.ocr_factory import OCRFactory  # type: ignore
        self.ocr_factory = OCRFactory()
    
    def test_ocr_dispatcher_routing(self):
        """Test that OCR dispatcher routes correctly based on category"""
        from ocr.gpt4o_ocr import GPT4oOCRAdapter  # type: ignore
        
        # Mock OpenAI API key requirement
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Create dispatcher
            dispatcher = GPT4oOCRAdapter()
        
            # Mock the individual OCR adapters
            with patch.object(dispatcher, 'daily_ocr') as mock_daily, \
                 patch.object(dispatcher, 'weekly_ocr') as mock_weekly, \
                 patch.object(dispatcher, 'monthly_ocr') as mock_monthly:
                
                mock_daily.extract_text.return_value = "daily_result"
                mock_weekly.extract_text.return_value = "weekly_result"
                mock_monthly.extract_text.return_value = "monthly_result"
                
                # Test daily routing
                result = dispatcher.extract_text("test_path", "Day")
                mock_daily.extract_text.assert_called_once_with("test_path")
                self.assertEqual(result, "daily_result")
                
                # Reset mocks
                mock_daily.reset_mock()
                mock_weekly.reset_mock()
                mock_monthly.reset_mock()
                
                # Test weekly routing
                result = dispatcher.extract_text("test_path", "Week")
                mock_weekly.extract_text.assert_called_once_with("test_path")
                self.assertEqual(result, "weekly_result")
                
                # Reset mocks
                mock_daily.reset_mock()
                mock_weekly.reset_mock()
                mock_monthly.reset_mock()
                
                # Test monthly routing
                result = dispatcher.extract_text("test_path", "Month")
                mock_monthly.extract_text.assert_called_once_with("test_path")
                self.assertEqual(result, "monthly_result")
    
    def test_ocr_dispatcher_default_routing(self):
        """Test that OCR dispatcher defaults to daily for unknown categories"""
        from ocr.gpt4o_ocr import GPT4oOCRAdapter  # type: ignore
        
        # Mock OpenAI API key requirement
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Create dispatcher
            dispatcher = GPT4oOCRAdapter()
        
            # Mock the individual OCR adapters
            with patch.object(dispatcher, 'daily_ocr') as mock_daily:
                mock_daily.extract_text.return_value = "daily_default_result"
                
                # Test with unknown category
                result = dispatcher.extract_text("test_path", "UnknownCategory")
                mock_daily.extract_text.assert_called_once_with("test_path")
                self.assertEqual(result, "daily_default_result")
    
    def test_ocr_adapter_method_signatures(self):
        """Test that all OCR adapters have correct method signatures"""
        from ocr.daily_ocr import DailyOCRAdapter  # type: ignore
        from ocr.weekly_ocr import WeeklyOCRAdapter  # type: ignore
        from ocr.monthly_ocr import MonthlyOCRAdapter  # type: ignore
        
        # Check that all adapters have extract_text method with correct signature
        import inspect
        
        # Test DailyOCRAdapter
        daily_sig = inspect.signature(DailyOCRAdapter.extract_text)
        self.assertIn('image_path', daily_sig.parameters)
        self.assertIn('category', daily_sig.parameters)
        self.assertEqual(daily_sig.parameters['category'].default, 'Day')
        
        # Test WeeklyOCRAdapter
        weekly_sig = inspect.signature(WeeklyOCRAdapter.extract_text)
        self.assertIn('image_path', weekly_sig.parameters)
        self.assertIn('category', weekly_sig.parameters)
        self.assertEqual(weekly_sig.parameters['category'].default, 'Week')
        
        # Test MonthlyOCRAdapter
        monthly_sig = inspect.signature(MonthlyOCRAdapter.extract_text)
        self.assertIn('image_path', monthly_sig.parameters)
        self.assertIn('category', monthly_sig.parameters)
        self.assertEqual(monthly_sig.parameters['category'].default, 'Month')
    
    def test_ocr_factory_creates_dispatcher(self):
        """Test that OCR factory creates the correct dispatcher"""
        # Mock OpenAI API key requirement
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            ocr_engine = self.ocr_factory.get_current_engine()
            from ocr.gpt4o_ocr import GPT4oOCRAdapter  # type: ignore
            
            # Should create GPT4oOCRAdapter (dispatcher)
            self.assertIsInstance(ocr_engine, GPT4oOCRAdapter)
            
            # Should have the category-specific adapters
            self.assertTrue(hasattr(ocr_engine, 'daily_ocr'))
            self.assertTrue(hasattr(ocr_engine, 'weekly_ocr'))
            self.assertTrue(hasattr(ocr_engine, 'monthly_ocr'))


if __name__ == '__main__':
    unittest.main()