
import os
import logging
from typing import List, Optional
from itertools import cycle

logger = logging.getLogger(__name__)

class APIKeyManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.google_keys = []
        self._load_google_keys()
        self.google_key_cycle = cycle(self.google_keys) if self.google_keys else None
        
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if self.openai_key:
            logger.info("[APIKeyManager] OpenAI key available for fallback")
            
        self._initialized = True

    def _load_google_keys(self):
        # Primary key
        key1 = os.getenv("GOOGLE_API_KEY")
        if key1:
            self.google_keys.append(key1)
            
        # Specific rotation keys
        key2 = os.getenv("GOOGLE_API_KEY_2")
        if key2:
            self.google_keys.append(key2)
            
        # Scan for others if needed
        i = 3
        while True:
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if not key:
                break
            self.google_keys.append(key)
            i += 1
            
        count = len(self.google_keys)
        logger.info(f"[APIKeyManager] Initialized with {count} Google key(s)")

    def get_google_key_count(self) -> int:
        return len(self.google_keys)

    def get_current_google_key(self) -> Optional[str]:
        if not self.google_keys:
            return None
        # Use first key as default or current cycle
        # For simplicity, returning the first one available or maintain state
        if not hasattr(self, '_current_key_idx'):
            self._current_key_idx = 0
        if self._current_key_idx < len(self.google_keys):
            return self.google_keys[self._current_key_idx]
        return self.google_keys[0]

    def get_google_key_by_index(self, index: int) -> Optional[str]:
        if 0 <= index < len(self.google_keys):
            return self.google_keys[index]
        return None

    def rotate_google_key(self) -> bool:
        """Rotate to next key. Returns True if rotated, False if cycled back or empty."""
        if not self.google_keys:
            return False
            
        if not hasattr(self, '_current_key_idx'):
            self._current_key_idx = 0
            
        self._current_key_idx = (self._current_key_idx + 1) % len(self.google_keys)
        logger.info(f"[APIKeyManager] Rotated to Google key index {self._current_key_idx}")
        return True

    def get_openai_key(self) -> Optional[str]:
        return self.openai_key

api_key_manager = APIKeyManager()
