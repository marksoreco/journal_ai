"""
OCR Results Formatter - Server-side formatting using classic UI logic
"""
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class OCRFormatter:
    """Format OCR results using the same logic as the classic UI"""
    
    def __init__(self, confidence_threshold: float = 0.9):
        self.confidence_threshold = confidence_threshold
    
    def _format_single_value_section(self, lines: list, header: str, data: dict, key: str, field_name: str = 'value') -> None:
        """Helper method to format single-value sections with consistent 'No content' handling"""
        lines.append(header)
        if data.get(key):
            text = self.format_with_confidence(data[key], field_name)
            if text and text.strip():
                lines.append(f"• {text}")
            else:
                lines.append("*No content*")
        else:
            lines.append("*No content*")
        lines.append("")
    
    def _format_array_section(self, lines: list, header: str, data: dict, key: str, field_name: str = 'value') -> None:
        """Helper method to format array sections with consistent 'No content' handling"""
        lines.append(header)
        if data.get(key) and len(data[key]) > 0:
            for item in data[key]:
                text = self.format_with_confidence(item, field_name)
                if text and text.strip():
                    lines.append(f"• {text}")
        else:
            lines.append("*No content*")
        lines.append("")
    
    def _clean_numbered_text(self, text: str) -> str:
        """Remove leading numbers and periods from text"""
        if not text:
            return text
        
        import re
        # Remove patterns like "1. ", "2. ", etc. at the beginning of lines
        cleaned = re.sub(r'^(\d+\.\s*)', '', text.strip())
        # Also handle multiple numbered items separated by newlines
        cleaned = re.sub(r'\n(\d+\.\s*)', '\n', cleaned)
        
        return cleaned.strip()
    
    def format_with_confidence(self, item: Any, field_name: str = 'task') -> str:
        """Format text with confidence checking - same logic as classic UI"""
        if isinstance(item, str):
            return item
        
        if isinstance(item, dict):
            text = item.get(field_name) or item.get('item') or item.get('value')
            confidence = item.get('confidence')
            
            # If no text found in expected fields, return empty string instead of dict representation
            if not text:
                return ""
            
            if confidence and confidence < self.confidence_threshold:
                return f"*{text}*"
            return text
        
        return str(item)
    
    def detect_low_confidence_items(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect all low-confidence items across all sections for task-related content"""
        low_confidence_items = []
        logger.info(f"Detecting low-confidence items. Available sections: {list(data.keys())}")
        
        # Check all sections that contain lists of items
        for section_name, section_items in data.items():
            if isinstance(section_items, list) and section_items:
                logger.info(f"Checking {len(section_items)} {section_name} items")
                for index, item in enumerate(section_items):
                    if isinstance(item, dict):
                        # Try different field names for text content (in priority order)
                        text = item.get('task') or item.get('item') or item.get('value') or item.get('text')
                        confidence = item.get('confidence')
                        
                        # Determine the actual field name used
                        field_name = 'task' if item.get('task') else \
                                    'item' if item.get('item') else \
                                    'value' if item.get('value') else \
                                    'text' if item.get('text') else None
                        
                        if text and confidence and confidence < self.confidence_threshold and field_name:
                            low_confidence_items.append({
                                'text': text,
                                'confidence': confidence,
                                'section': section_name,
                                'field_name': field_name,
                                'item_index': index,
                                'original_item': item
                            })
                            logger.info(f"Found low-confidence item in {section_name}: '{text}' (confidence: {confidence})")
        
        logger.info(f"Found {len(low_confidence_items)} low-confidence items total")
        
        return low_confidence_items
    
    def format_daily_page(self, data: Dict[str, Any]) -> str:
        """Format Daily page data - matches classic UI formatDailyPageMarkdown"""
        try:
            lines = []
            logger.info(f"Formatting daily page with data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            # Header with date
            if data.get('date'):
                date_text = self.format_with_confidence(data['date'], 'value')
                lines.append(f"📅 {date_text}")
                lines.append("")
            
            # Habit Focus
            if data.get('habit'):
                habit_text = self.format_with_confidence(data['habit'], 'value')
                lines.append("💪 **Habit**")
                lines.append(f"• {habit_text}")
                lines.append("")
            
            # Theme
            if data.get('theme'):
                theme_text = self.format_with_confidence(data['theme'], 'value')
                lines.append("🎨 **Theme**")
                lines.append(f"• {theme_text}")
                lines.append("")
            
            # Priority tasks
            if data.get('prepare_priority') and len(data['prepare_priority']) > 0:
                lines.append("📋 **Priority Tasks**")
                for task in data['prepare_priority']:
                    formatted_task = self.format_with_confidence(task)
                    formatted_task = self._clean_numbered_text(formatted_task)
                    lines.append(f"• {formatted_task}")
                lines.append("")
            
            # To-Do items
            if data.get('to_do') and len(data['to_do']) > 0:
                lines.append("✓ **To-Do Items**")
                for task in data['to_do']:
                    formatted_task = self.format_with_confidence(task)
                    formatted_task = self._clean_numbered_text(formatted_task)
                    lines.append(f"• {formatted_task}")
                lines.append("")
            
            # Gratitude
            if data.get('i_am_grateful_for') and len(data['i_am_grateful_for']) > 0:
                lines.append("🙏 **Grateful For**")
                for item in data['i_am_grateful_for']:
                    formatted_item = self.format_with_confidence(item, 'item')
                    formatted_item = self._clean_numbered_text(formatted_item)
                    lines.append(f"• {formatted_item}")
                lines.append("")
            
            # Looking forward to
            if data.get('i_am_looking_forward_to') and len(data['i_am_looking_forward_to']) > 0:
                lines.append("🎯 **Looking Forward To**")
                for item in data['i_am_looking_forward_to']:
                    formatted_item = self.format_with_confidence(item, 'item')
                    formatted_item = self._clean_numbered_text(formatted_item)
                    lines.append(f"• {formatted_item}")
                lines.append("")
            
            # Daily schedule
            if data.get('daily') and len(data['daily']) > 0:
                lines.append("⏰ **Daily Schedule**")
                for entry in data['daily']:
                    hour = entry.get('hour', 0)
                    time_str = f"{hour % 12 if hour % 12 != 0 else 12}:00 {'PM' if hour >= 12 else 'AM'}"
                    if entry.get('activities') and len(entry['activities']) > 0:
                        activities = []
                        for activity in entry['activities']:
                            formatted_activity = self.format_with_confidence(activity, 'activity')
                            activities.append(formatted_activity)
                        lines.append(f"{time_str}: {', '.join(activities)}")
                lines.append("")
            
            # Ways to give
            if data.get('ways_i_can_give') and len(data['ways_i_can_give']) > 0:
                lines.append("🤝 **Ways I Can Give**")
                for item in data['ways_i_can_give']:
                    formatted_item = self.format_with_confidence(item, 'item')
                    formatted_item = self._clean_numbered_text(formatted_item)
                    lines.append(f"• {formatted_item}")
                lines.append("")
            
            # Reflection sections (new nested structure)
            if data.get('reflect'):
                lines.append("**Reflect:**\n")
                
                # Highlights
                if data['reflect'].get('highlights') and len(data['reflect']['highlights']) > 0:
                    lines.append("💭 **Highlights:**")
                    for highlight in data['reflect']['highlights']:
                        highlight_text = self.format_with_confidence(highlight, 'value')
                        highlight_text = self._clean_numbered_text(highlight_text)
                        lines.append(f"• {highlight_text}")
                    lines.append("")
                
                # I was at my best when
                if data['reflect'].get('i_was_at_my_best_when'):
                    best_text = self.format_with_confidence(data['reflect']['i_was_at_my_best_when'], 'value')
                    best_text = self._clean_numbered_text(best_text)
                    lines.append("⭐ **At My Best When**")
                    lines.append(f"• {best_text}")
                    lines.append("")
                
                # I felt unrest when
                if data['reflect'].get('i_felt_unrest_when'):
                    unrest_text = self.format_with_confidence(data['reflect']['i_felt_unrest_when'], 'value')
                    unrest_text = self._clean_numbered_text(unrest_text)
                    lines.append("😰 **Felt Unrest When**")
                    lines.append(f"• {unrest_text}")
                    lines.append("")
                
                # One way I can improve tomorrow
                if data['reflect'].get('one_way_i_can_improve_tomorrow'):
                    improve_text = self.format_with_confidence(data['reflect']['one_way_i_can_improve_tomorrow'], 'value')
                    improve_text = self._clean_numbered_text(improve_text)
                    lines.append("🚀 **Tomorrow's Improvement**")
                    lines.append(f"• {improve_text}")
                    lines.append("")
            
            result = '\n'.join(lines).strip()
            logger.info(f"Daily page formatting completed successfully, {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Critical error in format_daily_page: {e}")
            logger.error(f"Data that caused error: {data}")
            return f"**Daily Page**\n\nError formatting daily page: {str(e)}"
    
    def format_weekly_page(self, data: Dict[str, Any]) -> str:
        """Format Weekly page data - matches classic UI formatWeeklyPageMarkdown"""
        try:
            lines = []
            
            logger.info(f"Formatting weekly page with data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            # Week date range
            if data.get('week'):
                week_text = self.format_with_confidence(data['week'], 'value')
                lines.append(f"📅 {week_text}")
                lines.append("")
            
            # Priority tasks
            if data.get('prepare_priority') and len(data['prepare_priority']) > 0:
                logger.info(f"Processing {len(data['prepare_priority'])} priority tasks")
                lines.append("📋 **Weekly Priorities**")
                for task in data['prepare_priority']:
                    try:
                        formatted_task = self.format_with_confidence(task)
                        lines.append(f"• {formatted_task}")
                    except Exception as e:
                        logger.error(f"Error formatting priority task: {e}")
                        lines.append("• [Error formatting task]")
                lines.append("")
            
            # Habit tracker
            if data.get('habit_tracker'):
                logger.info("Processing habit tracker object")
                try:
                    lines.append("📈 **Habit Tracker**")
                    days = [
                        ('M', 'monday'),
                        ('T', 'tuesday'), 
                        ('W', 'wednesday'),
                        ('T', 'thursday'),
                        ('F', 'friday'),
                        ('S', 'saturday'),
                        ('S', 'sunday')
                    ]
                    day_line = []
                    status_line = []
                    
                    habit_data = data['habit_tracker']
                    logger.info(f"Habit tracker data: {habit_data}")
                    for day_letter, day_key in days:
                        try:
                            day_data = habit_data.get(day_key)
                            logger.info(f"Processing {day_key}: {day_data}")
                            if day_data:
                                marked = '✔' if day_data.get('marked') else '○'
                            else:
                                marked = '○'
                                logger.warning(f"No data found for {day_key}, using default ○")
                            day_line.append(day_letter)
                            status_line.append(marked)
                        except Exception as e:
                            logger.error(f"Error processing habit tracker day {day_key}: {e}")
                            day_line.append(day_letter)
                            status_line.append('?')
                    
                    logger.info(f"Final day_line: {day_line} (length: {len(day_line)})")
                    logger.info(f"Final status_line: {status_line} (length: {len(status_line)})")
                    lines.append(' '.join(day_line))
                    lines.append(' '.join(status_line))
                    lines.append("")
                except Exception as e:
                    logger.error(f"Error processing habit tracker: {e}")
                    lines.append("📈 **Habit Tracker**: Error processing data")
                    lines.append("")
            
            # To-Do items with completion status
            if data.get('to_do') and len(data['to_do']) > 0:
                logger.info(f"Processing {len(data['to_do'])} to-do items")
                lines.append("✓ **To-Do Items**")
                for task in data['to_do']:
                    try:
                        completed = '✔' if task.get('completed') else '⬜'
                        formatted_task = self.format_with_confidence(task)
                        lines.append(f"• {completed} {formatted_task}")
                    except Exception as e:
                        logger.error(f"Error formatting to-do task: {e}")
                        lines.append("• ⬜ [Error formatting task]")
                lines.append("")
            
            # Personal Growth
            if data.get('personal_growth'):
                growth_text = self.format_with_confidence(data['personal_growth'], 'value')
                lines.append("🌱 **Personal Growth**")
                lines.append(f"• {growth_text}")
                lines.append("")
            
            # Relationship Growth
            if data.get('relationships_growth'):
                rel_text = self.format_with_confidence(data['relationships_growth'], 'value')
                lines.append("💕 **Relationship Growth**")
                lines.append(f"• {rel_text}")
                lines.append("")
            
            # Looking Forward To
            if data.get('looking_forward_to'):
                lines.append("🎯 **Looking Forward To**")
                for key in ['1', '2', '3']:
                    if data['looking_forward_to'].get(key):
                        item_text = self.format_with_confidence(data['looking_forward_to'][key], 'value')
                        lines.append(f"• {item_text}")
                lines.append("")
            
            # Reflect sections (updated structure to match daily/monthly formatter pattern)
            if data.get('reflect'):
                lines.append("**Reflect**\n")
                lines.append("")
                
                reflect_data = data.get('reflect', {})
                
                # Biggest accomplishments
                if reflect_data.get('biggest_accomplishments') and len(reflect_data['biggest_accomplishments']) > 0:
                    lines.append("🏆 **Biggest Accomplishments:**")
                    for accomplishment in reflect_data['biggest_accomplishments']:
                        acc_text = self.format_with_confidence(accomplishment, 'value')
                        if acc_text and acc_text.strip():
                            lines.append(f"• {acc_text}")
                else:
                    lines.append("🏆 **Biggest Accomplishments:**")
                    lines.append("*No content*")
                lines.append("")
                
                # Habits insights
                if reflect_data.get('habits_insights'):
                    habits_text = self.format_with_confidence(reflect_data['habits_insights'], 'value')
                    lines.append("💪 **Habits Insights:**")
                    if habits_text and habits_text.strip():
                        lines.append(f"• {habits_text}")
                    else:
                        lines.append("*No content*")
                else:
                    lines.append("💪 **Habits Insights:**")
                    lines.append("*No content*")
                lines.append("")
                
                # Meaningful moments
                if reflect_data.get('meaningful_moments'):
                    moments_text = self.format_with_confidence(reflect_data['meaningful_moments'], 'value')
                    lines.append("✨ **Meaningful Moments:**")
                    if moments_text and moments_text.strip():
                        lines.append(f"• {moments_text}")
                    else:
                        lines.append("*No content*")
                else:
                    lines.append("✨ **Meaningful Moments:**")
                    lines.append("*No content*")
                lines.append("")
                
                # God is teaching me
                if reflect_data.get('god_is_teaching_me'):
                    god_text = self.format_with_confidence(reflect_data['god_is_teaching_me'], 'value')
                    lines.append("🙏 **God is Teaching Me:**")
                    if god_text and god_text.strip():
                        lines.append(f"• {god_text}")
                    else:
                        lines.append("*No content*")
                else:
                    lines.append("🙏 **God is Teaching Me:**")
                    lines.append("*No content*")
                lines.append("")
                
                # One change next week
                if reflect_data.get('one_change_next_week'):
                    change_text = self.format_with_confidence(reflect_data['one_change_next_week'], 'value')
                    lines.append("🚀 **One Change Next Week:**")
                    if change_text and change_text.strip():
                        lines.append(f"• {change_text}")
                    else:
                        lines.append("*No content*")
                else:
                    lines.append("🚀 **One Change Next Week:**")
                    lines.append("*No content*")
                lines.append("")
            else:
                lines.append("**Reflect**\n")
                lines.append("")
                lines.append("🏆 **Biggest Accomplishments:**")
                lines.append("*No content*")
                lines.append("")
                lines.append("💪 **Habits Insights:**")
                lines.append("*No content*")
                lines.append("")
                lines.append("✨ **Meaningful Moments:**")
                lines.append("*No content*")
                lines.append("")
                lines.append("🙏 **God is Teaching Me:**")
                lines.append("*No content*")
                lines.append("")
                lines.append("🚀 **One Change Next Week:**")
                lines.append("*No content*")
                lines.append("")
            
            result = '\n'.join(lines).strip()
            logger.info(f"Weekly page formatting completed successfully, {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Critical error in format_weekly_page: {e}")
            logger.error(f"Data that caused error: {data}")
            return f"**Weekly Page**\n\nError formatting weekly page: {str(e)}"
    
    def format_monthly_page(self, data: Dict[str, Any]) -> str:
        """Format Monthly page data - matches classic UI formatMonthlyPageMarkdown"""
        try:
            lines = []
            logger.info(f"Formatting monthly page with data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            # Header with month/year
            if data.get('month'):
                month_text = data['month'].get('month', 'Unknown Month')
                year_text = data['month'].get('year', 'Unknown Year')
                lines.append(f"📅 {month_text} {year_text}")
                lines.append("")
            
            # Monthly Habit and Theme (reordered to match prompt)
            self._format_single_value_section(lines, "💪 **Habit**", data, 'habit')
            self._format_single_value_section(lines, "🎨 **Theme**", data, 'theme')
            
            # Calendar Events
            lines.append("📅 **Calendar Events**")
            if data.get('calendar') and len(data['calendar']) > 0:
                for event in data['calendar']:
                    day = event.get('day', 'Unknown')
                    event_text = self.format_with_confidence(event, 'value')
                    lines.append(f"• {day}: {event_text}")
            else:
                lines.append("*No content*")
            lines.append("")
            
            # Monthly priorities/goals
            self._format_array_section(lines, "🎯 **Prepare - Priority**", data, 'prepare_priority', 'task')
            
            # Monthly check-in wellness ratings
            lines.append("🌟 **Monthly Check-In**")
            if data.get('monthly_check_in') and len(data['monthly_check_in']) > 0:
                for category, rating in data['monthly_check_in'].items():
                    if rating and isinstance(rating, (int, float)):
                        rating = int(rating)
                        # Use + and - characters for better visibility and alignment
                        progress = '+' * rating + '-' * (10 - rating)
                        # Use 18 characters padding to accommodate "personal growth" (15 chars) + buffer
                        lines.append(f"{category:<18}: {progress} ({rating}/10)")
            else:
                lines.append("*No content*")
            lines.append("")
            
            # One Change and One Question sections
            self._format_single_value_section(lines, "🚀 **One Change I Can Make**", data, 'one_change_i_can_make_this_month_that_will_have_the_biggest_impact')
            self._format_single_value_section(lines, "❓ **One Question To Answer**", data, 'one_question_i_d_like_to_answer_this_month')
            
            # Monthly Reflect sections (updated structure to match daily formatter)
            if data.get('reflect'):
                lines.append("**Reflect**\n")
                lines.append("")
                
                reflect_data = data.get('reflect', {})
                
                # Biggest accomplishments
                if reflect_data.get('biggest_accomplishments') and len(reflect_data['biggest_accomplishments']) > 0:
                    lines.append("🏆 **Biggest Accomplishments:**")
                    for accomplishment in reflect_data['biggest_accomplishments']:
                        acc_text = self.format_with_confidence(accomplishment, 'value')
                        if acc_text and acc_text.strip():
                            lines.append(f"• {acc_text}")
                else:
                    lines.append("🏆 **Biggest Accomplishments:**")
                    lines.append("*No content*")
                lines.append("")
                
                # Relationships I'm grateful for
                if reflect_data.get('relationships_i_m_grateful_for') and len(reflect_data['relationships_i_m_grateful_for']) > 0:
                    lines.append("💕 **Relationships I'm Grateful For:**")
                    for relationship in reflect_data['relationships_i_m_grateful_for']:
                        rel_text = self.format_with_confidence(relationship, 'value')
                        if rel_text and rel_text.strip():
                            lines.append(f"• {rel_text}")
                else:
                    lines.append("💕 **Relationships I'm Grateful For:**")
                    lines.append("*No content*")
                lines.append("")
                
                # Greatest insight gained
                if reflect_data.get('greatest_insight_gained'):
                    insight_text = self.format_with_confidence(reflect_data['greatest_insight_gained'], 'value')
                    lines.append("💡 **Greatest Insight Gained:**")
                    if insight_text and insight_text.strip():
                        lines.append(f"• {insight_text}")
                    else:
                        lines.append("*No content*")
                else:
                    lines.append("💡 **Greatest Insight Gained:**")
                    lines.append("*No content*")
                lines.append("")
            else:
                lines.append("**Reflect**\n")
                lines.append("🏆 **Biggest Accomplishments:**")
                lines.append("*No content*")
                lines.append("")
                lines.append("💕 **Relationships I'm Grateful For:**")
                lines.append("*No content*")
                lines.append("")
                lines.append("💡 **Greatest Insight Gained:**")
                lines.append("*No content*")
                lines.append("")
        
            result = '\n'.join(lines).strip()
            logger.info(f"Monthly page formatting completed successfully, {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Critical error in format_monthly_page: {e}")
            logger.error(f"Data that caused error: {data}")
            return f"**Monthly Page**\n\nError formatting monthly page: {str(e)}"
    
    def format_ocr_results(self, page_type: str, ocr_data: Dict[str, Any]) -> str:
        """Format OCR results based on page type"""
        try:
            logger.info(f"Formatting {page_type} page with data type: {type(ocr_data)}")
            
            # Validate that ocr_data is a dictionary
            if not isinstance(ocr_data, dict):
                raise ValueError(f"OCR data must be a dictionary, got {type(ocr_data)}: {ocr_data}")
            
            if page_type.lower() == 'daily':
                return self.format_daily_page(ocr_data)
            elif page_type.lower() == 'weekly':
                return self.format_weekly_page(ocr_data)
            elif page_type.lower() == 'monthly':
                return self.format_monthly_page(ocr_data)
            else:
                # Fallback for unknown page types
                logger.warning(f"Unknown page type: {page_type}")
                return f"**{page_type} Page**\n\n" + str(ocr_data)
        except Exception as e:
            logger.error(f"Error formatting {page_type} page: {str(e)}")
            logger.error(f"OCR data that caused error: {ocr_data}")
            return f"**{page_type} Page**\n\nError formatting results: {str(e)}"