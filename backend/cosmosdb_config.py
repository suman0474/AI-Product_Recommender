import os
import logging
from typing import Optional
from enum import Enum
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError

logger = logging.getLogger(__name__)

class CosmosContainers(str, Enum):
    WORKFLOW_SESSIONS = "workflow_sessions"
    USER_PROJECTS = "user_projects"
    STRATEGY_DOCUMENTS = "strategy_documents"
    STANDARDS_DOCUMENTS = "standards_documents"

class CosmosDBConnection:
    _instance: Optional['CosmosDBConnection'] = None
    _client: Optional[CosmosClient] = None
    DATABASE_NAME = "engenie"

    def __init__(self):
        if CosmosDBConnection._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            CosmosDBConnection._instance = self
            self._initialize_client()

    @staticmethod
    def get_instance() -> 'CosmosDBConnection':
        if CosmosDBConnection._instance is None:
            CosmosDBConnection()
        return CosmosDBConnection._instance

    def _initialize_client(self):
        """Initialize Azure Cosmos DB Client"""
        endpoint = os.getenv("COSMOS_ENDPOINT")
        key = os.getenv("COSMOS_KEY")
        connection_string = os.getenv("COSMOS_CONNECTION_STRING")

        if not (endpoint and key) and not connection_string:
            logger.warning("[COSMOS] No credentials found (COSMOS_ENDPOINT/KEY or COSMOS_CONNECTION_STRING). Cosmos DB features will be disabled.")
            return

        try:
            if connection_string:
                self._client = CosmosClient.from_connection_string(connection_string)
            else:
                self._client = CosmosClient(endpoint, credential=key)
            
            # Verify/Create Database
            try:
                self._client.create_database_if_not_exists(id=self.DATABASE_NAME)
            except Exception as e:
                logger.warning(f"[COSMOS] Could not verify/create database '{self.DATABASE_NAME}': {e}")
            
            logger.info("[COSMOS] Client initialized successfully")
        except Exception as e:
            logger.error(f"[COSMOS] Failed to initialize client: {e}")
            self._client = None

    @property
    def client(self) -> Optional[CosmosClient]:
        return self._client

    def get_or_create_container(self, container_name: str, partition_key: str = "/user_id"):
        """
        Get or create a Cosmos DB container.

        Args:
            container_name: Name of the container
            partition_key: Partition key path (default: /user_id)

        Returns:
            Container client or None if failed
        """
        if not self._client:
            logger.warning(f"[COSMOS] Cannot get container '{container_name}' - client not initialized")
            return None

        try:
            database = self._client.get_database_client(self.DATABASE_NAME)

            # Try to create container if not exists
            try:
                container = database.create_container_if_not_exists(
                    id=container_name,
                    partition_key={"paths": [partition_key], "kind": "Hash"}
                )
                logger.info(f"[COSMOS] Container ready: {container_name}")
                return container
            except CosmosHttpResponseError as e:
                logger.error(f"[COSMOS] Failed to create/get container '{container_name}': {e}")
                return None

        except Exception as e:
            logger.error(f"[COSMOS] Error accessing container '{container_name}': {e}")
            return None

    def get_container(self, container_name: str):
        """
        Get a Cosmos DB container (backward compatibility).

        Args:
            container_name: Name of the container (can be CosmosContainers enum value)

        Returns:
            Container client or None if failed
        """
        return self.get_or_create_container(container_name)
