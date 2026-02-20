
class AgenticConfig:
    """
    Configuration for Agentic behavior and models.
    """
    # Models
    FLASH_MODEL = "gemini-2.5-flash"
    PRO_MODEL = "gemini-2.5-pro"
    
    # Analysis settings
    MAX_VENDORS = 5
    TIMEOUT_SECONDS = 60

    @classmethod
    def validate(cls):
        """Validate configuration."""
        pass
