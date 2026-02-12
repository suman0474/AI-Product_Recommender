"""
MongoDB Connection Manager
===========================
Singleton pattern for MongoDB Atlas connection management.

This replaces AzureBlobCollection for metadata queries while keeping
Azure Blob Storage for file storage.

Usage:
    from core.mongodb_manager import mongodb_manager, get_mongodb_connection

    # Direct collection access
    db = mongodb_manager.database
    specs = db['specs']
    result = specs.find_one({'product_type': 'Pressure Transmitter'})

    # Or use the compatibility function
    conn = get_mongodb_connection()
    specs = conn['collections']['specs']
"""
import os
import threading
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Try to import pymongo
try:
    from pymongo import MongoClient
    from pymongo.database import Database
    from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    MongoClient = None
    Database = None
    print("[WARNING] pymongo not installed. Run: pip install pymongo>=4.6.0")


class MongoDBManager:
    """
    Singleton MongoDB connection manager.

    Features:
    - Lazy initialization (connects on first use)
    - Thread-safe singleton pattern
    - Automatic reconnection on failure
    - Health check capabilities
    - Graceful fallback when MongoDB unavailable
    """

    _instance: Optional['MongoDBManager'] = None
    _lock = threading.Lock()
    _client: Optional[Any] = None  # MongoClient
    _database: Optional[Any] = None  # Database

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._connected = False
            self._connection_error = None
            self._mongodb_uri = os.getenv('MONGODB_URI')
            self._db_name = os.getenv('MONGODB_DATABASE')

    def _setup_connection(self) -> bool:
        """
        Initialize MongoDB connection (lazy).

        Returns:
            True if connected successfully, False otherwise
        """
        if not PYMONGO_AVAILABLE:
            self._connection_error = "pymongo not installed"
            print("[ERROR] pymongo not installed. Run: pip install pymongo>=4.6.0")
            return False

        if not self._mongodb_uri:
            self._connection_error = "MONGODB_URI not set"
            print("[WARNING] MONGODB_URI not set in environment")
            print("   Add to .env: MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/")
            return False

        try:
            print(f"[MongoDB] Connecting to database: {self._db_name}")

            self._client = MongoClient(
                self._mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=30000
            )

            # Test connection
            self._client.admin.command('ping')
            self._database = self._client[self._db_name]
            self._connected = True
            self._connection_error = None

            print(f"[MongoDB] Connected successfully")
            print(f"[MongoDB] Database: {self._db_name}")
            print(f"[MongoDB] Collections: {self._database.list_collection_names()}")

            return True

        except ServerSelectionTimeoutError as e:
            self._connection_error = f"Server selection timeout: {str(e)}"
            print(f"[MongoDB] Connection timeout: {str(e)}")
            print("[MongoDB] Check your MONGODB_URI and network connection")
            return False

        except ConnectionFailure as e:
            self._connection_error = f"Connection failure: {str(e)}"
            print(f"[MongoDB] Connection failed: {str(e)}")
            return False

        except Exception as e:
            self._connection_error = str(e)
            print(f"[MongoDB] Connection error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    @property
    def database(self) -> Optional[Any]:
        """
        Get database instance (lazy connection).

        Returns:
            MongoDB Database instance or None if connection failed
        """
        if self._database is None:
            self._setup_connection()
        return self._database

    @property
    def client(self) -> Optional[Any]:
        """
        Get MongoDB client instance.

        Returns:
            MongoClient instance or None if connection failed
        """
        if self._client is None:
            self._setup_connection()
        return self._client

    def get_collection(self, name: str):
        """
        Get a collection by name.

        Args:
            name: Collection name

        Returns:
            MongoDB Collection or None if not connected
        """
        db = self.database
        if db is not None:
            return db[name]
        return None

    def is_connected(self) -> bool:
        """
        Check if MongoDB is currently connected.

        Returns:
            True if connected and healthy, False otherwise
        """
        if not self._connected or self._client is None:
            return False

        try:
            self._client.admin.command('ping')
            return True
        except Exception:
            self._connected = False
            return False

    def get_connection_error(self) -> Optional[str]:
        """Get the last connection error message"""
        return self._connection_error

    def reconnect(self) -> bool:
        """
        Force reconnection to MongoDB.

        Returns:
            True if reconnected successfully
        """
        self.close()
        return self._setup_connection()

    def close(self):
        """Close MongoDB connection"""
        if self._client:
            try:
                self._client.close()
                print("[MongoDB] Connection closed")
            except Exception as e:
                print(f"Error closing MongoDB connection: {e}")
            finally:
                self._client = None
                self._database = None
                self._connected = False

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on MongoDB connection.

        Returns:
            Dict with health status information
        """
        result = {
            'service': 'MongoDB',
            'status': 'unknown',
            'connected': False,
            'database': self._db_name,
            'error': None
        }

        if not PYMONGO_AVAILABLE:
            result['status'] = 'error'
            result['error'] = 'pymongo not installed'
            return result

        if not self._mongodb_uri:
            result['status'] = 'error'
            result['error'] = 'MONGODB_URI not configured'
            return result

        try:
            if self.is_connected():
                result['status'] = 'healthy'
                result['connected'] = True
                result['collections'] = self._database.list_collection_names()
            else:
                result['status'] = 'disconnected'
                result['error'] = self._connection_error
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        return result


# Singleton instance
mongodb_manager = MongoDBManager()


# ============================================================
# Compatibility Layer
# ============================================================

def get_mongodb_connection() -> Dict[str, Any]:
    """
    Get MongoDB connection with collections dict.

    This provides compatibility with the existing code pattern:
        conn = get_azure_blob_connection()
        specs = conn['collections']['specs']

    Can be replaced with:
        conn = get_mongodb_connection()
        specs = conn['collections']['specs']

    Returns:
        Dict with 'db' and 'collections' keys
    """
    db = mongodb_manager.database

    if db is None:
        # Return empty structure if not connected
        return {
            'db': None,
            'collections': {},
            'connected': False,
            'error': mongodb_manager.get_connection_error()
        }

    return {
        'db': db,
        'collections': {
            'specs': db['specs'],
            'vendors': db['vendors'],
            'images': db['images'],
            'generic_images': db['generic_images'],
            'vendor_logos': db['vendor_logos'],
            'advanced_parameters': db['advanced_parameters'],
            'user_projects': db['user_projects'],
            'stratergy': db['stratergy'],  # Note: Collection name in database
            'standards': db['standards'],
            'documents': db['documents']
        },
        'connected': True,
        'error': None
    }


def is_mongodb_available() -> bool:
    """
    Check if MongoDB is available and connected.

    Use this to determine whether to use MongoDB or fall back to Azure Blob.

    Returns:
        True if MongoDB is available and connected
    """
    return mongodb_manager.is_connected()


# ============================================================
# Fallback Manager (for gradual migration)
# ============================================================

class HybridConnectionManager:
    """
    Manages connections to both MongoDB and Azure Blob.

    During migration, this allows gradual transition:
    - Read from MongoDB first, fall back to Azure Blob
    - Write to both systems (dual-write)
    - Eventually switch fully to MongoDB
    """

    def __init__(self):
        self._use_mongodb = os.getenv('USE_MONGODB', 'true').lower() == 'true'
        self._dual_write = os.getenv('DUAL_WRITE', 'false').lower() == 'true'

    def get_connection(self, prefer_mongodb: bool = True) -> Dict[str, Any]:
        """
        Get database connection (MongoDB or Azure Blob).

        Args:
            prefer_mongodb: If True, try MongoDB first

        Returns:
            Connection dict with 'db', 'collections', 'source'
        """
        # Try MongoDB first
        if prefer_mongodb and self._use_mongodb:
            if is_mongodb_available():
                conn = get_mongodb_connection()
                conn['source'] = 'mongodb'
                return conn

        # Fall back to Azure Blob
        try:
            from azure_blob_config import get_azure_blob_connection
            conn = get_azure_blob_connection()
            conn['source'] = 'azure_blob'
            return conn
        except ImportError:
            return {
                'db': None,
                'collections': {},
                'source': 'none',
                'error': 'No database available'
            }

    def is_dual_write_enabled(self) -> bool:
        """Check if dual-write mode is enabled"""
        return self._dual_write


# Singleton for hybrid manager
hybrid_manager = HybridConnectionManager()


# ============================================================
# Module-level initialization
# ============================================================

def init_mongodb():
    """
    Initialize MongoDB connection.

    Call this at application startup to ensure connection is ready.
    """
    if mongodb_manager.database is not None:
        print("[MongoDB] Initialized successfully")
        return True
    else:
        print("[WARNING] MongoDB not available - will use fallback")
        return False


# Auto-initialize on module import (optional)
# Uncomment the following line to connect immediately on import:
# init_mongodb()
