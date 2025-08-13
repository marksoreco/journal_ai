import logging
from .base import BaseOCR
from .daily_ocr import DailyOCRAdapter
from .weekly_ocr import WeeklyOCRAdapter
from .monthly_ocr import MonthlyOCRAdapter

# Configure logger for this module
logger = logging.getLogger(__name__)

class GPT4oOCRAdapter(BaseOCR):
    def __init__(self):
        # Initialize the category-specific OCR adapters
        self.daily_ocr = DailyOCRAdapter()
        self.weekly_ocr = WeeklyOCRAdapter()
        self.monthly_ocr = MonthlyOCRAdapter()

    def extract_text(self, image_path: str, category: str = "Day") -> str:
        """
        Extract text from image using the appropriate page-specific OCR adapter.
        
        Args:
            image_path: Path to the image file
            category: Page category - "Day", "Week", or "Month"
            
        Returns:
            Extracted text as JSON string
        """
        logger.info(f"Starting {category} page OCR extraction for image: {image_path}")
        
        try:
            # Route to the appropriate OCR adapter based on category
            if category.lower() in ["day", "daily"]:
                return self.daily_ocr.extract_text(image_path)
            elif category.lower() in ["week", "weekly"]:
                return self.weekly_ocr.extract_text(image_path)
            elif category.lower() in ["month", "monthly"]:
                return self.monthly_ocr.extract_text(image_path)
            else:
                logger.warning(f"Unknown category '{category}', defaulting to daily page processing")
                return self.daily_ocr.extract_text(image_path)
                
        except Exception as e:
            logger.error(f"Error in GPT4oOCRAdapter dispatcher: {str(e)}")
            return f"Error in OCR processing: {str(e)}" 