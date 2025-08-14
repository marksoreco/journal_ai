import logging
from typing import Dict, Type
from typing import Optional
from .base import BaseOCR
from .config import OCR_ENGINE

# Configure logger for this module
logger = logging.getLogger(__name__)

class OCRFactory:
    """
    Factory class for creating OCR engine instances based on configuration.
    """
    
    def __init__(self):
        self._engines: Dict[str, Type[BaseOCR]] = {}
        self._current_engine = None
        self._register_engines()
    
    def _register_engines(self):
        """Register all available OCR engines"""
        try:
            from .gpt4o_ocr import GPT4oOCRAdapter
            self._engines["GPT4oOCRAdapter"] = GPT4oOCRAdapter
        except ImportError:
            pass  # Engine not available
        
        # Register page-specific OCR adapters
        try:
            from .daily_ocr import DailyOCRAdapter
            self._engines["DailyOCRAdapter"] = DailyOCRAdapter
        except ImportError:
            pass  # Engine not available
            
        try:
            from .weekly_ocr import WeeklyOCRAdapter
            self._engines["WeeklyOCRAdapter"] = WeeklyOCRAdapter
        except ImportError:
            pass  # Engine not available
            
        try:
            from .monthly_ocr import MonthlyOCRAdapter
            self._engines["MonthlyOCRAdapter"] = MonthlyOCRAdapter
        except ImportError:
            pass  # Engine not available
        
        # Add more engines here as they become available
        # try:
        #     from .tesseract_ocr import TesseractOCRAdapter
        #     self._engines["TesseractOCRAdapter"] = TesseractOCRAdapter
        # except ImportError:
        #     pass
    
    def get_available_engines(self) -> list[str]:
        """Get list of available OCR engine names"""
        return list(self._engines.keys())
    
    def create_engine(self, engine_name: Optional[str] = None) -> BaseOCR:
        """
        Create an OCR engine instance
        
        Args:
            engine_name: Name of the engine to create. If None, uses config default.
            
        Returns:
            BaseOCR: OCR engine instance
            
        Raises:
            ValueError: If engine name is not supported or engine is not available
        """
        if engine_name is None:
            engine_name = OCR_ENGINE
        
        if engine_name not in self._engines:
            available = ", ".join(self.get_available_engines())
            raise ValueError(
                f"Unknown OCR_ENGINE: {engine_name}. "
                f"Available engines: {available}"
            )
        
        engine_class = self._engines[engine_name]
        return engine_class()
    
    def get_current_engine(self) -> BaseOCR:
        """
        Get the current OCR engine instance (creates if not exists)
        
        Returns:
            BaseOCR: Current OCR engine instance
        """
        if self._current_engine is None:
            self._current_engine = self.create_engine()
        return self._current_engine
    
    def set_engine(self, engine_name: str) -> BaseOCR:
        """
        Set and create a new OCR engine instance
        
        Args:
            engine_name: Name of the engine to create
            
        Returns:
            BaseOCR: New OCR engine instance
        """
        self._current_engine = self.create_engine(engine_name)
        return self._current_engine 