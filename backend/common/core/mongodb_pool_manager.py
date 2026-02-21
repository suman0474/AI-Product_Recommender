"""
MongoDB Connection Pool Manager
================================
Enhanced MongoDB manager with connection pooling for high concurrency.

PROBLEM: Current MongoDBManager (mongodb_manager.py) uses a single connection
         which is not optimized for high concurrency scenarios.

SOLUTION: Implement connection pooling with:
1. Multiple connections in pool
2. Connection reuse and recycling
3. Health checks and auto-recovery
4. Per-request connection management
5. Metrics and monitoring

Usage:
    from common.core.mongodb_pool_manager import mongodb_pool
    
    # Context manager (recommended)
    with mongodb_pool.get_connection() as db:
        collection = db['strategy']
        docs = collection.find({'vendor': 'Agilent'})
    
    # Or get collection directly
    collection = mongodb_pool.get_collection('strategy')
    docs = collection.find({'vendor': 'Agilent'})
"""

import os
import threading
import time
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime, timedelta
# Import dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are already set

logger = logging.getLogger(__name__)

# Try to import pymongo
try:
    from pymongo import MongoClient
    from pymongo.database import Database
    from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure, NetworkTimeout
    from pymongo.pool import PoolOptions
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    MongoClient = None
    Database = None
    logger.warning("[MongoDBPool] pymongo not installed. Run: pip install pymongo>=4.6.0")


class ConnectionStats:
    """Track connection pool statistics"""
    
    def __init__(self):
        self.total_requests = 0
        self.active_connections = 0
        self.failed_connections = 0
        self.total_queries = 0
        self.avg_query_time_ms = 0.0
        self.last_health_check = None
        self._lock = threading.Lock()
    
    def record_request(self):
        with self._lock:
            self.total_requests += 1
    
    def record_query(self, duration_ms: float):
        with self._lock:
            self.total_queries += 1
            # Running average
            self.avg_query_time_ms = (
                (self.avg_query_time_ms * (self.total_queries - 1) + duration_ms) 
                / self.total_queries
            )
    
    def record_failure(self):
        with self._lock:
            self.failed_connections += 1
    
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'total_requests': self.total_requests,
                'active_connections': self.active_connections,
                'failed_connections': self.failed_connections,
                'total_queries': self.total_queries,
                'avg_query_time_ms': round(self.avg_query_time_ms, 2),
                'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None
            }


class MongoDBPoolManager:
    """
    MongoDB Connection Pool Manager with high-concurrency support.
    
    Features:
    - Connection pooling (configurable size)
    - Automatic connection recycling
    - Health checks and monitoring
    - Thread-safe operations
    - Graceful degradation
    - Connection timeout handling
    - Metrics collection
    """
    
    _instance: Optional['MongoDBPoolManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            
            # Configuration
            self._mongodb_uri = os.getenv('MONGODB_URI')
            self._db_name = os.getenv('MONGODB_DATABASE')
            
            # Pool configuration
            self._max_pool_size = int(os.getenv('MONGODB_MAX_POOL_SIZE', '100'))
            self._min_pool_size = int(os.getenv('MONGODB_MIN_POOL_SIZE', '10'))
            self._max_idle_time_ms = int(os.getenv('MONGODB_MAX_IDLE_TIME_MS', '60000'))  # 60 seconds
            self._wait_queue_timeout_ms = int(os.getenv('MONGODB_WAIT_QUEUE_TIMEOUT_MS', '10000'))  # 10 seconds
            
            # Connection objects
            self._client: Optional[Any] = None
            self._database: Optional[Any] = None
            
            # State
            self._connected = False
            self._connection_error = None
            
            # Statistics
            self.stats = ConnectionStats()
            
            # Health check
            self._last_health_check = None
            self._health_check_interval = timedelta(seconds=30)
    
    def _setup_connection(self) -> bool:
        """
        Initialize MongoDB connection pool.
        
        Returns:
            True if connected successfully, False otherwise
        """
        if not PYMONGO_AVAILABLE:
            self._connection_error = "pymongo not installed"
            logger.error("[MongoDBPool] pymongo not installed. Run: pip install pymongo>=4.6.0")
            return False
        
        if not self._mongodb_uri:
            self._connection_error = "MONGODB_URI not set"
            logger.warning("[MongoDBPool] MONGODB_URI not set in environment")
            return False
        
        try:
            logger.info(
                f"[MongoDBPool] Initializing connection pool: "
                f"max_pool_size={self._max_pool_size}, "
                f"min_pool_size={self._min_pool_size}"
            )
            
            # Create client with connection pool settings
            self._client = MongoClient(
                self._mongodb_uri,
                # Connection pool settings
                maxPoolSize=self._max_pool_size,
                minPoolSize=self._min_pool_size,
                maxIdleTimeMS=self._max_idle_time_ms,
                waitQueueTimeoutMS=self._wait_queue_timeout_ms,
                
                # Timeout settings
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=30000,
                
                # Connection management
                retryWrites=True,
                retryReads=True,
                
                # Application name for monitoring
                appName="agentic-rag-backend"
            )
            
            # Test connection
            self._client.admin.command('ping')
            
            # Get database
            self._database = self._client[self._db_name]
            
            self._connected = True
            self._connection_error = None
            
            logger.info(
                f"[MongoDBPool] Connected successfully to database: {self._db_name}"
            )
            
            return True
            
        except Exception as e:
            self._connection_error = str(e)
            logger.error(f"[MongoDBPool] Connection failed: {e}")
            self.stats.record_failure()
            return False
    
    @property
    def database(self) -> Optional[Any]:
        """
        Get database instance (lazy initialization).
        
        Returns:
            Database instance or None if connection failed
        """
        if self._database is None and not self._connected:
            self._setup_connection()
        return self._database
    
    @property
    def client(self) -> Optional[Any]:
        """Get MongoDB client instance"""
        if self._client is None:
            self._setup_connection()
        return self._client
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a database connection.
        
        Usage:
            with mongodb_pool.get_connection() as db:
                collection = db['my_collection']
                docs = collection.find({})
        
        Yields:
            Database instance
        """
        self.stats.record_request()
        
        db = self.database
        if db is None:
            raise ConnectionError("MongoDB connection not available")
        
        start_time = time.time()
        
        try:
            yield db
        finally:
            # Record query time
            duration_ms = (time.time() - start_time) * 1000
            self.stats.record_query(duration_ms)
    
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
        
        # Check if health check is needed
        now = datetime.now()
        if (self._last_health_check is None or 
            now - self._last_health_check > self._health_check_interval):
            
            try:
                self._client.admin.command('ping')
                self._last_health_check = now
                self.stats.last_health_check = now
                return True
            except Exception:
                self._connected = False
                return False
        
        return True
    
    def get_connection_error(self) -> Optional[str]:
        """Get the last connection error message"""
        return self._connection_error
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Dict with pool metrics
        """
        if not self._client:
            return {'error': 'Not connected'}
        
        try:
            # Get pool stats from client
            pool_stats = {
                'max_pool_size': self._max_pool_size,
                'min_pool_size': self._min_pool_size,
                'active_connections': len(self._client.nodes),
                'connection_stats': self.stats.to_dict()
            }
            
            return pool_stats
            
        except Exception as e:
            logger.error(f"[MongoDBPool] Failed to get pool stats: {e}")
            return {'error': str(e)}
    
    def reconnect(self) -> bool:
        """
        Force reconnection to MongoDB.
        
        Returns:
            True if reconnected successfully
        """
        logger.info("[MongoDBPool] Attempting reconnection...")
        self.close()
        return self._setup_connection()
    
    def close(self):
        """Close all MongoDB connections in pool"""
        if self._client:
            try:
                self._client.close()
                logger.info("[MongoDBPool] Connection pool closed")
            except Exception as e:
                logger.error(f"[MongoDBPool] Error closing pool: {e}")
            finally:
                self._client = None
                self._database = None
                self._connected = False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Dict with health status information
        """
        result = {
            'service': 'MongoDBPool',
            'status': 'unknown',
            'connected': False,
            'database': self._db_name,
            'pool_stats': None,
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
                result['pool_stats'] = self.get_pool_stats()
            else:
                result['status'] = 'disconnected'
                result['error'] = self._connection_error
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result


# Global singleton instance
mongodb_pool = MongoDBPoolManager()


# Backward compatibility functions
def get_mongodb_pool() -> MongoDBPoolManager:
    """Get the global MongoDB pool manager instance"""
    return mongodb_pool


def init_mongodb_pool() -> bool:
    """
    Initialize MongoDB connection pool.
    
    Returns:
        True if successful, False otherwise
    """
    if mongodb_pool.database is not None:
        logger.info("[MongoDBPool] Initialized successfully")
        return True
    else:
        logger.warning("[MongoDBPool] Initialization failed - will use fallback")
        return False


def is_mongodb_pool_available() -> bool:
    """Check if MongoDB pool is available and connected"""
    return mongodb_pool.is_connected()


def get_pool_stats() -> Dict[str, Any]:
    """Get connection pool statistics"""
    return mongodb_pool.get_pool_stats()
