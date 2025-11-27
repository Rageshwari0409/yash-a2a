import os
import json
import boto3
from botocore.exceptions import ClientError
import base64
import psycopg2
import psycopg2.pool
import psycopg2.extras
import logging
from typing import Dict, Any, Optional, List
from litellm import Router
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(self):
        self.db_pool = None
        self.router = None
        # self._team_routers: Dict[str, Router] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def initialize_db_pool(self):
        """Initialize database connection pool"""
        try:
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            
            if not all([db_host, db_port, db_name, db_user, db_password]):
                raise ValueError("All database environment variables (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD) are required")
            
            # Run the synchronous pool creation in thread pool
            loop = asyncio.get_event_loop()
            self.db_pool = await loop.run_in_executor(
                self.executor,
                lambda: psycopg2.pool.ThreadedConnectionPool(
                    1, 2,
                    host=db_host,
                    port=db_port,
                    database=db_name,
                    user=db_user,
                    password=db_password
                )
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {str(e)}")
            raise
    
    async def close_db_pool(self):
        """Close database connection pool"""
        if self.db_pool:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.db_pool.closeall
            )
            self.executor.shutdown(wait=True)
            logger.info("Database connection pool closed")
    
    
    def _execute_query(self, query: str, params: tuple):
        """Execute query synchronously in thread pool"""
        conn = None
        try:
            conn = self.db_pool.getconn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchone()
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    async def get_team_model_config(self, team_id: str) -> Dict[str, Any]:
        """Fetch team's selected model configuration from database"""
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                query = """
                    SELECT
                    l.model_id,
                    l.config AS selected_model_config,
                    m.model_code AS selected_model,
                    m.provider_id,
                    p.name AS provider
                    FROM
                    teams_llm_config AS l
                    JOIN
                    llm_models AS m ON l.model_id = m.id
                    JOIN
                    llm_providers AS p ON m.provider_id = p.id
                    WHERE
                    l.is_active = TRUE and l.team_id = %s
                """
                
                loop = asyncio.get_event_loop()
                row = await loop.run_in_executor(
                    self.executor,
                    self._execute_query,
                    query,
                    (team_id,)
                )
                
                if not row:
                    raise ValueError(f"No model configuration found for team_id: {team_id}")
                
                config = row["selected_model_config"]
                if isinstance(config, str):
                    config = await self.decrypt(config)
                    config = json.loads(config)
                
                return {
                    "provider": row["provider"],
                    "selected_model": row["selected_model"],
                    "config": config
                }
            except psycopg2.OperationalError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to fetch model config after {max_retries} retries: {str(e)}")
                    raise
                logger.warning(f"Connection error, retrying ({retry_count}/{max_retries}): {str(e)}")
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
            except Exception as e:
                logger.error(f"Failed to fetch model config for team {team_id}: {str(e)}")
                raise
    
    def create_router_for_team(self, provider: str, selected_model: str, model_config: Dict[str, Any]) -> Router:
        """Create a LiteLLM router for a specific team's model configuration"""
        try:
            # Construct provider_model string for LiteLLM
            provider_model = f"{provider}/{selected_model}"
            
            # Create model list for router
            model_list = [{
                "model_name": selected_model,
                "litellm_params": {
                    "model": provider_model,
                    **model_config 
                }
            }]

            print("Creating router with model list:", model_list)
            
            # Create and return router
            router = Router(model_list=model_list)
            logger.info(f"Created router for model: {provider_model}")
            return router
            
        except Exception as e:
            logger.error(f"Failed to create router for {provider}/{selected_model}: {str(e)}")
            raise
    
    async def get_router_for_team(self, team_id: str) -> tuple[Router, str]:
        """Get or create a router for a specific team"""
        try:
            # # Check if we already have a cached router for this team
            # if team_id in self._team_routers:
            #     # Get the team config to return the model name
            #     team_config = await self.get_team_model_config(team_id)
            #     return self._team_routers[team_id], team_config["selected_model"]
            
            # Fetch team's model configuration
            team_config = await self.get_team_model_config(team_id)
            provider = team_config["provider"]
            selected_model = team_config["selected_model"]
            model_config = team_config["config"]
            
            # Create router for this team
            router = self.create_router_for_team(provider, selected_model, model_config)
            
            # # Cache the router
            # self._team_routers[team_id] = router
            
            return router, selected_model
            
        except Exception as e:
            logger.error(f"Failed to get router for team {team_id}: {str(e)}")
            raise

    async def create_kms_client(self):
        """Create and return a KMS client"""
        kms_client = boto3.client(
            'kms',
            region_name=os.getenv("AWS_REGION"),
        )

        return kms_client
    
    async def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt data using AWS KMS
        
        Args:
            ciphertext: Base64 encoded encrypted data
            key_id: Optional KMS Key ID (AWS can auto-detect from ciphertext)
        
        Returns:
            Decrypted plaintext as string
        """
        try:
            kms_client = await self.create_kms_client()
            key_id = os.getenv("AWS_KMS_KEY_ID_ARN")
            # Decode base64 ciphertext
            ciphertext_blob = base64.b64decode(ciphertext)
            
            # Decrypt parameters
            decrypt_params = {'CiphertextBlob': ciphertext_blob}
            if key_id:
                decrypt_params['KeyId'] = key_id
            
            # Decrypt the data
            response = kms_client.decrypt(**decrypt_params)
            
            # Return decrypted plaintext
            return response['Plaintext'].decode('utf-8')
            
        except ClientError as e:
            print(f"AWS KMS Decryption Error: {e}")
            raise
    

# Global instance
model_config = ModelConfig()

@asynccontextmanager
async def get_model_config():
    """Context manager for model configuration"""
    try:
        if not model_config.db_pool:
            await model_config.initialize_db_pool()
        yield model_config
    except Exception as e:
        logger.error(f"Error in model config context: {str(e)}")
        raise
    finally:
        pass  # Keep connection pool alive for reuse

async def initialize_config():
    """Initialize the global model configuration"""
    await model_config.initialize_db_pool()

async def cleanup_config():
    """Cleanup the global model configuration"""
    await model_config.close_db_pool()