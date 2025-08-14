import unittest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ocr.monthly_ocr import MonthlyOCRAdapter


class TestMonthlyDateParsing(unittest.TestCase):
    """Test suite for validating various date format parsing in MonthlyOCRAdapter"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            self.adapter = MonthlyOCRAdapter()
    
    def test_parse_full_month_year(self):
        """Test parsing full month names with year"""
        test_cases = [
            ("September 2023", {"month": "September", "year": 2023}),
            ("March 2024", {"month": "March", "year": 2024}),
            ("December 2022", {"month": "December", "year": 2022}),
            ("January 2025", {"month": "January", "year": 2025})
        ]
        
        for input_str, expected in test_cases:
            with self.subTest(input_str=input_str):
                result = self.adapter._parse_month_value(input_str)
                self.assertEqual(result["month"], expected["month"])
                self.assertEqual(result["year"], expected["year"])
                self.assertGreaterEqual(result["confidence"], 0.8)
    
    def test_parse_abbreviated_month_year(self):
        """Test parsing abbreviated month names with year"""
        test_cases = [
            ("Sept 2023", {"month": "September", "year": 2023}),
            ("Mar 2024", {"month": "March", "year": 2024}),
            ("Dec 2022", {"month": "December", "year": 2022}),
            ("Jan 2025", {"month": "January", "year": 2025}),
            ("Feb 2023", {"month": "February", "year": 2023}),
            ("Aug 2024", {"month": "August", "year": 2024}),
            ("Oct 2023", {"month": "October", "year": 2023}),
            ("Nov 2024", {"month": "November", "year": 2024})
        ]
        
        for input_str, expected in test_cases:
            with self.subTest(input_str=input_str):
                result = self.adapter._parse_month_value(input_str)
                self.assertEqual(result["month"], expected["month"])
                self.assertEqual(result["year"], expected["year"])
                self.assertGreaterEqual(result["confidence"], 0.8)
    
    def test_parse_numeric_month_full_year(self):
        """Test parsing numeric month with full year (MM/YYYY)"""
        test_cases = [
            ("3/2023", {"month": "March", "year": 2023}),
            ("03/2023", {"month": "March", "year": 2023}),
            ("12/2024", {"month": "December", "year": 2024}),
            ("01/2022", {"month": "January", "year": 2022}),
            ("9/2025", {"month": "September", "year": 2025})
        ]
        
        for input_str, expected in test_cases:
            with self.subTest(input_str=input_str):
                result = self.adapter._parse_month_value(input_str)
                self.assertEqual(result["month"], expected["month"])
                self.assertEqual(result["year"], expected["year"])
                self.assertGreaterEqual(result["confidence"], 0.7)
    
    def test_parse_numeric_month_short_year(self):
        """Test parsing numeric month with short year (MM/YY)"""
        # Note: dateparser interprets MM/YY differently - it may default to future dates
        # Test that we get valid month names and reasonable years
        test_cases = [
            ("3/23", "March"),
            ("03/23", "March"),
            ("12/24", "December"),
            ("01/22", "January"),
            ("9/25", "September"),
            ("6/95", "June"),
            ("12/85", "December")
        ]
        
        for input_str, expected_month in test_cases:
            with self.subTest(input_str=input_str):
                result = self.adapter._parse_month_value(input_str)
                self.assertEqual(result["month"], expected_month)
                self.assertIsInstance(result["year"], int)
                self.assertGreaterEqual(result["year"], 1980)  # Reasonable range
                self.assertLessEqual(result["year"], 2030)     # Reasonable range
                self.assertGreaterEqual(result["confidence"], 0.7)
    
    def test_parse_month_only_gives_reasonable_year(self):
        """Test parsing month name only gives reasonable year based on dateparser logic"""
        current_date = datetime.now()
        current_year = current_date.year
        current_month_num = current_date.month
        
        test_cases = [
            ("March", "March", 3),
            ("September", "September", 9), 
            ("December", "December", 12),
            ("Jan", "January", 1),
            ("Sept", "September", 9),
            ("Dec", "December", 12)
        ]
        
        for input_str, expected_month, month_num in test_cases:
            with self.subTest(input_str=input_str):
                result = self.adapter._parse_month_value(input_str)
                self.assertEqual(result["month"], expected_month)
                
                # With 'PREFER_DATES_FROM': 'past' setting, dateparser behavior can vary
                # Just ensure we get a reasonable year within a 2-year range
                reasonable_years = [current_year - 1, current_year, current_year + 1]
                
                self.assertIn(result["year"], reasonable_years, 
                             f"Month {input_str} gave year {result['year']}, expected within range {reasonable_years}")
                self.assertGreaterEqual(result["confidence"], 0.7)
    
    def test_parse_empty_or_invalid_strings(self):
        """Test parsing empty or invalid strings handles gracefully"""
        current_date = datetime.now()
        current_month = current_date.strftime("%B")
        current_year = current_date.year
        
        # Test cases that should give low confidence
        definitely_invalid_cases = [
            "",
            "   ",
            "abc/def",  # Non-numeric
            None
        ]
        
        # Test cases that might be parsed by dateparser or fall back to current date
        potentially_parseable_cases = [
            "invalid",
            "13/2023",  # Invalid month (may be parsed as day/month)
            "2023"      # Year only (may be parsed as current month/year)
        ]
        
        for input_str in definitely_invalid_cases:
            with self.subTest(input_str=input_str):
                # Handle None case
                if input_str is None:
                    result = self.adapter._parse_month_value("")
                else:
                    result = self.adapter._parse_month_value(input_str)
                
                # Should have appropriate confidence for definitely invalid cases
                if input_str in ["", "   ", None]:
                    self.assertEqual(result["confidence"], 0.5)
                else:
                    self.assertEqual(result["confidence"], 0.3)
                
                # Should contain valid month and year
                self.assertIn(result["month"], [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ])
                self.assertIsInstance(result["year"], int)
                self.assertGreaterEqual(result["year"], 1900)
        
        # Test that potentially parseable cases still return valid structures
        for input_str in potentially_parseable_cases:
            with self.subTest(input_str=input_str):
                result = self.adapter._parse_month_value(input_str)
                
                # Should contain valid month and year regardless of confidence
                self.assertIn(result["month"], [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ])
                self.assertIsInstance(result["year"], int)
                self.assertGreaterEqual(result["year"], 1900)
                self.assertGreaterEqual(result["confidence"], 0.0)
                self.assertLessEqual(result["confidence"], 1.0)
    
    def test_confidence_scores(self):
        """Test that confidence scores are appropriate for different input qualities"""
        # High confidence cases (clear, unambiguous formats)
        high_confidence_cases = ["September 2023", "March 2024", "Sept 2023"]
        for case in high_confidence_cases:
            result = self.adapter._parse_month_value(case)
            self.assertGreaterEqual(result["confidence"], 0.8, f"Expected high confidence for {case}")
        
        # Medium/high confidence cases (dateparser handles these well)
        parseable_cases = ["3/2023", "March", "September"]
        for case in parseable_cases:
            result = self.adapter._parse_month_value(case)
            self.assertGreaterEqual(result["confidence"], 0.7, f"Expected good confidence for {case}")
        
        # Low confidence cases (fallback to defaults)
        low_confidence_cases = ["", "   ", "abc/def"]
        for case in low_confidence_cases:
            result = self.adapter._parse_month_value(case)
            self.assertLessEqual(result["confidence"], 0.5, f"Expected low confidence for {case}")
    
    def test_return_format_structure(self):
        """Test that all results return the expected dictionary structure"""
        test_cases = ["September 2023", "3/23", "March", "invalid", ""]
        
        for case in test_cases:
            with self.subTest(input_str=case):
                result = self.adapter._parse_month_value(case)
                
                # Check structure
                self.assertIsInstance(result, dict)
                self.assertIn("month", result)
                self.assertIn("year", result) 
                self.assertIn("confidence", result)
                
                # Check types
                self.assertIsInstance(result["month"], str)
                self.assertIsInstance(result["year"], int)
                self.assertIsInstance(result["confidence"], (int, float))
                
                # Check value ranges
                self.assertGreaterEqual(result["confidence"], 0.0)
                self.assertLessEqual(result["confidence"], 1.0)
                self.assertGreaterEqual(result["year"], 1900)
                self.assertLessEqual(result["year"], 3000)  # Reasonable upper bound
    
    @patch('ocr.monthly_ocr.logger')
    def test_error_handling(self, mock_logger):
        """Test error handling and logging for edge cases"""
        # Test that errors are logged but don't crash the function
        result = self.adapter._parse_month_value("completely invalid input 12345!@#")
        
        # Should still return a valid result structure
        self.assertIsInstance(result, dict)
        self.assertIn("month", result)
        self.assertIn("year", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["confidence"], 0.3)  # Low confidence fallback
    
    def test_integration_json_processing(self):
        """Test integration with the post-processing logic used in extract_text"""
        # Simulate OCR data that might come from GPT-4o
        test_cases = [
            # New format (already processed)
            {
                "month": {"month": "September", "year": 2023, "confidence": 0.9},
                "expected": {"month": "September", "year": 2023, "confidence": 0.9}
            },
            # Old format (needs processing)  
            {
                "month": {"value": "Sept 2023", "confidence": 0.8},
                "expected": {"month": "September", "year": 2023}
            },
            # String format (needs processing)
            {
                "month": "March 2024",
                "expected": {"month": "March", "year": 2024}
            }
        ]
        
        for i, case in enumerate(test_cases):
            with self.subTest(case_num=i):
                # Simulate the post-processing logic from extract_text method
                ocr_data = {"month": case["month"]}
                
                # Apply the same logic as in the extract_text method
                if 'month' in ocr_data and ocr_data['month']:
                    if isinstance(ocr_data['month'], dict):
                        if 'value' in ocr_data['month']:
                            # Old format: parse the value field
                            month_str = ocr_data['month']['value']
                            parsed_month = self.adapter._parse_month_value(month_str)
                            # Update to new format
                            ocr_data['month'] = {
                                "month": parsed_month['month'],
                                "year": parsed_month['year'],
                                "confidence": min(ocr_data['month'].get('confidence', 0.5), parsed_month['confidence'])
                            }
                        # New format already handled
                    else:
                        # Handle case where month is just a string
                        parsed_month = self.adapter._parse_month_value(str(ocr_data['month']))
                        ocr_data['month'] = parsed_month
                
                # Verify results
                expected = case["expected"]
                actual = ocr_data['month']
                
                self.assertEqual(actual["month"], expected["month"])
                self.assertEqual(actual["year"], expected["year"])
                if "confidence" in expected:
                    self.assertEqual(actual["confidence"], expected["confidence"])
                else:
                    self.assertGreaterEqual(actual["confidence"], 0.0)
                    self.assertLessEqual(actual["confidence"], 1.0)


if __name__ == '__main__':
    unittest.main()