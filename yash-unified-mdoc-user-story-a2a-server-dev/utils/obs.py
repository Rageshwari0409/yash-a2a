# utils/obs.py

import os
import logging
import inspect
import jwt
from typing import Dict, Any, Optional, List
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Import the singleton logger instance from your kafka module
from .kafka import kafka_logger

logger = logging.getLogger(__name__)

CUSTOM_TOKEN_SEPARATOR = "$YashUnified2025$"

class TokenTracker(BaseCallbackHandler):
    """
    A self-contained callback handler that inspects its creation context to
    capture request metadata. It constructs a minimal payload tailored for
    the consumer's database logic and sends it to Kafka.
    """
    def __init__(self, model: str):
        """
        Initializes the tracker. It inspects the calling frame to extract the
        'encrypted_payload' and 'user_email' from the 'query' object's auth token.
        """
        # --- Context will be stored here after inspection ---
        self._context: Dict[str, Any] = {}
        self._context_extracted = False

        # --- Environment-based constants for the payload ---
        self.model_name = model
        self.agent_name = os.getenv("AGENT_NAME", "Unknown Agent")
        self.server_name = os.getenv("SERVER_NAME", "Unknown A2A Server")

        try:
            # Look at the frame of the function that called this __init__
            caller_frame = inspect.stack()[1].frame
            caller_locals = caller_frame.f_locals

            query = caller_locals.get('query')
            if not isinstance(query, dict):
                logger.warning("Could not find a 'query' dictionary in the caller's scope.")
                return

            # --- Step 1: Extract ONLY the necessary context ---
            user_email, encrypted_payload = "N/A", "N/A"
            auth_token = query.get('auth_token')

            if auth_token:
                jwt_part = auth_token
                if CUSTOM_TOKEN_SEPARATOR in auth_token:
                    jwt_part, encrypted_payload = auth_token.split(CUSTOM_TOKEN_SEPARATOR, 1)

                try:
                    # We only need the user_email from the JWT for the created_by/updated_by fields
                    decoded_token = jwt.decode(jwt_part, options={"verify_signature": False})
                    custom_data = decoded_token.get("custom-data", {})
                    user_email = custom_data.get("user_email") or decoded_token.get("email") or "N/A"
                except jwt.PyJWTError as e:
                    logger.warning(f"Could not decode JWT from caller context: {e}")

            # --- Step 2: Store the minimal context ---
            self._context = {
                "user_email": user_email,
                "encrypted_payload": encrypted_payload,
            }
            self._context_extracted = True
            logger.debug("Successfully extracted minimal context via inspection.")

        except Exception as e:
            logger.error(f"Failed to inspect caller context for logging: {e}", exc_info=True)
        finally:
            if 'caller_frame' in locals():
                del caller_frame

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Triggered when an LLM call ends. Assembles the minimal, targeted
        log payload and sends it to Kafka.
        """
        if not self._context_extracted:
            logger.warning("Context was not extracted during init. Skipping Kafka log.")
            return

        try:
            # if not response.generations:
            #     return

            # for generation_list in response.generations:
            #     for generation in generation_list:
            #         if hasattr(generation, 'message') and hasattr(generation.message, 'usage_metadata'):
            #             usage_meta = generation.message.usage_metadata
            #             if not usage_meta:
            #                 continue
            #             # --- Assemble the final, minimal Kafka Log Payload ---
            #             final_log = {
            #                 # --- Fields required by the consumer ---

            #                 # The encrypted block containing all IDs
            #                 "encrypted_payload": self._context.get("encrypted_payload"),

            #                 # For the 'created_by' and 'updated_by' columns
            #                 "user_email": self._context.get("user_email"),

            #                 # For token_usage_record columns
            #                 "prompt_tokens": usage_meta.get('input_tokens', 0),
            #                 "completion_tokens": usage_meta.get('output_tokens', 0),
            #                 "total_tokens": usage_meta.get('total_tokens', 0),
            #                 "thoughts_token_count": usage_meta.get('output_token_details', {}).get('reasoning', 0),

            #                 # For lookup and direct insertion
            #                 "model_name": self.model_name,
            #                 "agent_name_constant": self.agent_name,
            #                 "server_name": self.server_name,
            #             }

            #             kafka_logger.log(final_log)
            #             return # We've logged this call, so we can exit

            if not response.llm_output:
                return
        
            usage_meta = response.llm_output.get('token_usage', {})
            completion_tokens_details = usage_meta.get('completion_tokens_details')

            final_log = {
                # --- Fields required by the consumer ---

                # The encrypted block containing all IDs
                "encrypted_payload": self._context.get("encrypted_payload"),

                # For the 'created_by' and 'updated_by' columns
                "user_email": self._context.get("user_email"),

                # For token_usage_record columns
                "prompt_tokens": usage_meta.get('prompt_tokens', 0),
                "completion_tokens": usage_meta.get('completion_tokens', 0),
                "total_tokens": usage_meta.get('total_tokens', 0),
                "thoughts_token_count": completion_tokens_details.reasoning_tokens if completion_tokens_details else 0,

                # For lookup and direct insertion
                "model_name": self.model_name,
                "agent_name_constant": self.agent_name,
                "server_name": self.server_name,
            }

            kafka_logger.log(final_log)
        except Exception as e:
            logger.error(f"Error in TokenTracker.on_llm_end: {e}", exc_info=True)