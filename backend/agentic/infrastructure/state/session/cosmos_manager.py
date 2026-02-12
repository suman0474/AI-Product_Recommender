
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceExistsError

logger = logging.getLogger(__name__)

class CosmosWorkflowSessionManager:
    """
    Azure Cosmos DB-based session manager with automatic TTL expiration.
    Replaces file-based storage and MongoDB storage.
    """
    
    def __init__(
        self,
        cosmos_client: CosmosClient,
        database_name: str = "engenie",
        container_name: str = "workflow_sessions",
        ttl_seconds: int = 7200  # 2 hours default
    ):
        self.client = cosmos_client
        self.database_name = database_name
        self.container_name = container_name
        self.ttl_seconds = ttl_seconds
        
        self.database = None
        self.container = None
        
        self._initialize_container()

    def _initialize_container(self):
        """Initialize database and container if they don't exist"""
        try:
            # Create Database if not exists
            self.database = self.client.create_database_if_not_exists(id=self.database_name)
            
            # Create Container if not exists
            # Partition Key: /thread_id
            self.container = self.database.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path="/thread_id"),
                default_ttl=self.ttl_seconds
            )
            
            logger.info(f"[COSMOS_SESSION] Initialized container '{self.container_name}' with TTL={self.ttl_seconds}s")
            
        except CosmosHttpResponseError as e:
            logger.error(f"[COSMOS_SESSION] Initialization failed: {e}")
            raise

    def create_session(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new workflow session."""
        import uuid
        
        thread_id = thread_id or f"thread_{uuid.uuid4().hex[:12]}"
        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow().isoformat()
        
        session_doc = {
            "id": thread_id,  # Cosmos DB required 'id' field
            "thread_id": thread_id,  # Partition Key
            "session_id": session_id,
            "phase": "initial",
            "user_input": user_input,
            "product_type": None,
            "schema": None,
            "provided_requirements": {},
            "missing_fields": [],
            "discovered_specs": [],
            "selected_specs": [],
            "vendor_analysis": None,
            "ranking_result": None,
            "steps_completed": [],
            "awaiting_user_input": False,
            "last_message": "",
            "error": None,
            "created_at": now,
            "updated_at": now,
            "ttl": self.ttl_seconds # Per-document TTL (optional override)
        }
        
        try:
            self.container.create_item(body=session_doc)
            logger.info(f"[COSMOS_SESSION] Created session: thread={thread_id}")
            return session_doc
        except CosmosHttpResponseError as e:
            logger.error(f"[COSMOS_SESSION] Failed to create session: {e}")
            raise

    def get_state(
        self,
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get workflow state by thread_id or session_id."""
        try:
            # Efficient point read requires valid partition key (thread_id)
            if thread_id:
                try:
                    state = self.container.read_item(item=thread_id, partition_key=thread_id)
                    return state
                except CosmosHttpResponseError as e:
                    if e.status_code == 404:
                        pass # Not found
                    else:
                        logger.error(f"[COSMOS_SESSION] Read error: {e}")
            
            # Query by session_id (Cross-partition query, less efficient but necessary fallback)
            if session_id:
                query = "SELECT * FROM c WHERE c.session_id = @session_id"
                parameters = [{"name": "@session_id", "value": session_id}]
                
                items = list(self.container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True
                ))
                
                if items:
                    return items[0]

            logger.debug(f"[COSMOS_SESSION] No state found for thread={thread_id}, session={session_id}")
            return None

        except CosmosHttpResponseError as e:
            logger.error(f"[COSMOS_SESSION] Get state failed: {e}")
            return None

    def update_state(self, state: Dict[str, Any]) -> None:
        """Update workflow state."""
        try:
            thread_id = state.get("thread_id")
            if not thread_id:
                logger.error("[COSMOS_SESSION] Cannot update state without thread_id")
                return

            # Ensure id matches thread_id for Cosmos
            if "id" not in state:
                state["id"] = thread_id
                
            state["updated_at"] = datetime.utcnow().isoformat()
            
            # Upsert (Insert or Replace)
            self.container.upsert_item(body=state)
            logger.debug(f"[COSMOS_SESSION] Updated state: thread={thread_id}")

        except CosmosHttpResponseError as e:
            logger.error(f"[COSMOS_SESSION] Failed to update state: {e}")

    def delete_state(self, thread_id: str) -> bool:
        """Explicitly delete a session."""
        try:
            self.container.delete_item(item=thread_id, partition_key=thread_id)
            logger.info(f"[COSMOS_SESSION] Deleted session: {thread_id}")
            return True
        except CosmosHttpResponseError as e:
            if e.status_code == 404:
                return False
            logger.error(f"[COSMOS_SESSION] Failed to delete state: {e}")
            return False

def get_cosmos_workflow_session_manager() -> Optional[CosmosWorkflowSessionManager]:
    """Factory to get Cosmos session manager instance"""
    from cosmosdb_config import CosmosDBConnection
    
    conn = CosmosDBConnection.get_instance()
    client = conn.client
    
    if not client:
        return None
        
    return CosmosWorkflowSessionManager(
        cosmos_client=client,
        database_name="engenie",
        container_name="workflow_sessions"
    )
