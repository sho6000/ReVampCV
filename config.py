import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for Resume Analyzer Pro"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Application Settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", 8501))
    STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    
    # File Upload Settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES = ['pdf', 'docx', 'txt']
    
    # Analysis Settings
    SKILLS_MATCH_WEIGHT = 0.6
    EXPERIENCE_MATCH_WEIGHT = 0.3
    EDUCATION_MATCH_WEIGHT = 0.1
    
    # UI Settings
    THEME_COLORS = {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#4CAF50',
        'warning': '#ff9800',
        'error': '#f44336',
        'info': '#2196F3'
    }
    
    @classmethod
    def is_openai_available(cls):
        """Check if OpenAI API is available"""
        return bool(cls.OPENAI_API_KEY)
    
    @classmethod
    def get_theme_colors(cls):
        """Get theme colors"""
        return cls.THEME_COLORS 