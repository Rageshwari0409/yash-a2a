# kafka.py
 
import os
import json
import logging
import atexit
import threading  # Import the threading module
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union
from uuid import uuid4

try:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable

    KAFKA_INSTALLED = True
except ImportError:
    KAFKA_INSTALLED = False

    class NoBrokersAvailable(Exception):
        pass


logger = logging.getLogger(__name__)


class KafkaLogger:
    """
    A resilient, production-grade logger that sends payloads asynchronously.
    It initializes its connection to Kafka lazily and is failsafe, meaning
    application startup and requests will not fail if Kafka is unavailable.
    """

    def __init__(self):
        self.producer = None
        self._lock = threading.Lock()
        self.topic = os.getenv("KAFKA_TOPIC_NAME", "llm-token-usage")
        atexit.register(self.close)
 
    def _initialize_producer(self) -> bool:
        """
        Initializes the KafkaProducer. This method is called internally and
        is protected by a lock to ensure it's thread-safe.
        Returns True on success, False on failure.
        """
        if not KAFKA_INSTALLED:
            logger.critical(
                "Dependency 'kafka-python' is not installed. Kafka logging is disabled."
            )
            return False
 
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
 
        if not bootstrap_servers or not self.topic:
            logger.critical(
                "KAFKA_BOOTSTRAP_SERVERS or KAFKA_TOPIC_NAME is not set. Kafka logging disabled."
            )
            return False
       
        try:
            logger.info(
                f"Attempting to initialize KafkaProducer and connect to {bootstrap_servers}..."
            )
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers.split(","),
                security_protocol="SSL",
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                retries=5,
                request_timeout_ms=30000,
                acks="all",
                # A small buffer time to allow the producer to batch messages even under low load
                linger_ms=5,
            )
            logger.info(
                f"KafkaProducer connected successfully. Logging to topic '{self.topic}'."
            )
            return True
        except (NoBrokersAvailable, Exception) as e:
            # Catch ANY exception during initialization.
            logger.critical(
                f"FATAL: Could not initialize KafkaProducer. Logging will be disabled. Error: {e}",
                exc_info=True,
            )
            self.producer = None  # Ensure producer is None on failure
            return False
 
    def _on_send_success(self, record_metadata):
        """Callback for successful message sends."""
        logging.debug(
            f"Message delivered to topic '{record_metadata.topic}' partition {record_metadata.partition}"
        )

    def _on_send_error(self, excp):
        """Callback for failed message sends."""
        logger.error(
            f"Error sending message to Kafka in the background: {excp}", exc_info=excp
        )

    def log(self, data: dict):
        """
        Sends a log asynchronously. If the producer is not initialized, it will
        attempt to do so. This operation will not block the caller or raise an
        exception if Kafka is unavailable.
        """
        if not self.producer:
            with self._lock:
                if not self.producer:
                    if not self._initialize_producer():
                        logger.warning("Kafka producer is not available. Message not sent.")
                        return
       
        if self.producer:
            try:
                print("\n--- [KAFKA PAYLOAD DEBUG] ---")
                print(json.dumps(data, indent=2))
                print("-----------------------------\n")
            except Exception as e:
                print(f"--- [KAFKA PAYLOAD DEBUG] FAILED TO PRINT PAYLOAD: {e} ---")
 
            try:
                future = self.producer.send(self.topic, value=data)
                future.add_callback(self._on_send_success)
                future.add_errback(self._on_send_error)
            except Exception as e:
                logger.error(f"Error while queuing message for Kafka: {e}", exc_info=True)
 
    def close(self):
        """Flushes buffered messages and closes the producer during graceful shutdown."""
        if self.producer:
            logger.info("Flushing remaining messages and closing Kafka producer...")
            try:
                self.producer.flush(timeout=10)
            except Exception as e:
                logger.error(f"Error flushing messages to Kafka: {e}", exc_info=True)
            finally:
                self.producer.close()
                logger.info("Kafka producer closed.")


class KafkaResponseLogger:
    """
    Kafka logger for streaming function responses to the 'agent-response-notification' topic.
    This captures all function responses (success/error) for monitoring and debugging.

    Usage:
        response_logger = KafkaResponseLogger()
        response_logger.log_response(response_data, auth_token)
        response_logger.log_error_response(error_data, auth_token)
    """

    def __init__(self):
        """Initialize the Kafka Response Logger."""
        self.producer = None
        self._lock = threading.Lock()
        self.topic = os.getenv("KAFKA_RESPONSE_TOPIC_NAME", "agent-response-notification")
        self.server_name = os.getenv("SERVER_NAME", "Unknown Server")
        logger.info(f"KafkaResponseLogger initialized for topic {self.topic}")

    def _initialize_producer(self) -> bool:
        """Initialize Kafka producer for response logging."""
        if not KAFKA_INSTALLED:
            logger.warning("kafka-python not installed. Response logging disabled.")
            return False

        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        if not bootstrap_servers:
            logger.warning(
                "KAFKA_BOOTSTRAP_SERVERS not set. Response logging disabled."
            )
            return False

        try:
            logger.info(
                f"Initializing KafkaResponseLogger producer for topic '{self.topic}'..."
            )
            producer_config = {
                "bootstrap_servers": bootstrap_servers.split(","),
                "value_serializer": lambda v: json.dumps(v, default=str).encode(
                    "utf-8"
                ),
                "key_serializer": lambda k: k.encode("utf-8") if k else None,
                "retries": 3,
                "request_timeout_ms": 15000,
                "acks": 1,
                "linger_ms": 10,
                "batch_size": 1024,
            }

            if os.getenv("KAFKA_USE_SSL", "true").lower() == "true":
                producer_config["security_protocol"] = "SSL"

            self.producer = KafkaProducer(**producer_config)
            logger.info(f"KafkaResponseLogger producer connected successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize KafkaResponseLogger producer: {e}")
            self.producer = None
            return False

    def _extract_user_context_from_token(self, auth_token: str = None) -> dict:
        """Extract encrypted_payload from auth token for user context."""
        user_context = {"encrypted_payload": "N/A"}

        if not auth_token:
            return user_context

        try:
            # Split JWT from encrypted payload using custom separator
            CUSTOM_TOKEN_SEPARATOR = "$YashUnified2025$"
            if CUSTOM_TOKEN_SEPARATOR in auth_token:
                _, encrypted_payload = auth_token.split(CUSTOM_TOKEN_SEPARATOR, 1)
                user_context["encrypted_payload"] = encrypted_payload
            else:
                # Fallback: use a mock encrypted payload derived from token
                jwt_part = auth_token
                if jwt_part.lower().startswith("bearer "):
                    jwt_part = jwt_part[7:]
                user_context["encrypted_payload"] = (
                    f"mock-encrypted-payload-{jwt_part[-10:]}"
                )

        except Exception as e:
            logger.debug(f"Error extracting user context from token: {e}")

        return user_context

    def _create_response_event(
        self, response_data: dict, auth_token: str = None
    ) -> dict:
        """Create response event structure with encrypted_payload, timestamp, response."""
        user_context = self._extract_user_context_from_token(auth_token)

        return {
            "encrypted_payload": user_context["encrypted_payload"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response": response_data,
            "kafka_topic_name": self.topic,
            "type" : "agent-response",
            "server_name": self.server_name
        }

    def _send_response(self, response_event: dict) -> bool:
        """Send response event to Kafka topic."""
        if not self.producer:
            with self._lock:
                if not self.producer:
                    if not self._initialize_producer():
                        return False

        if not self.producer:
            logger.debug("Response producer not available. Response not sent.")
            return False

        try:
            # Enhanced terminal logging for developer visibility
            print("--- A2A RESPONSE ---")
            print(f"Topic: '{self.topic}'")
            print(f"Encrypted Payload: {response_event.get('encrypted_payload', 'N/A')}")
            print(f"Timestamp: {response_event.get('timestamp', 'N/A')}")
            print(f"Response Type: {type(response_event.get('response', {}))}")
            print(f"Full Response JSON: {json.dumps(response_event, indent=2)}")
            print("--------------------")

            # Send to Kafka
            message_key = f"response_{response_event['timestamp']}"
            future = self.producer.send(
                self.topic, value=response_event, key=message_key
            )

            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)

            return True

        except Exception as e:
            logger.error(f"Error sending response to Kafka: {e}")
            return False

    def _on_send_success(self, record_metadata):
        """Callback for successful response sends."""
        logger.debug(
            f"Response sent to {record_metadata.topic}:{record_metadata.partition}"
        )

    def _on_send_error(self, exception):
        """Callback for failed response sends."""
        logger.error(f"Failed to send response: {exception}")

    # PUBLIC API METHODS

    def log(self, data: dict, auth_token: str = None) -> bool:
        """
        Log a response.

        Args:
            data (dict): The response data from the function
            auth_token (str): Authentication token for user context

        Returns:
            bool: True if response was sent successfully
        """
        response_event = self._create_response_event(data, auth_token)
        return self._send_response(response_event)

    def close(self):
        """Close the Kafka response producer."""
        if self.producer:
            try:
                self.producer.flush(timeout=5)
                self.producer.close()
                logger.info("KafkaResponseLogger producer closed.")
            except Exception as e:
                logger.error(f"Error closing KafkaResponseLogger producer: {e}")


class KafkaEventLogger:
    """
    Custom Kafka logger for sending real-time events to the 'agent-event-notification' topic.
    This provides users with visibility into what the system is doing, similar to AI "thinking mode".

    Usage:
        event_logger = KafkaEventLogger(session_id="user-session-123")
        event_logger.log_event("Agent is thinking...", "agent")
        event_logger.log_progress("Processing your request...", 50)
        event_logger.log_success("Task completed successfully!")
    """

    def __init__(self, session_id: str = None, user_context: Dict[str, Any] = None):
        """
        Initialize the Kafka Event Logger.

        Args:
            session_id (str): Session identifier to group related events
            user_context (Dict): User context information (email, id, etc.)
        """
        self.session_id = session_id or str(uuid4())
        self.user_context = user_context or {}
        self.producer = None
        self._lock = threading.Lock()
        self.topic = os.getenv("KAFKA_EVENT_TOPIC_NAME", "agent-event-notification")
        self.server_name = os.getenv("SERVER_NAME", "Unknown Server")

        # Event sequence counter for ordering
        self._event_sequence = 0

        logger.info(f"KafkaEventLogger initialized for session {self.session_id}")

    def _initialize_producer(self) -> bool:
        """Initialize Kafka producer for event logging."""
        if not KAFKA_INSTALLED:
            logger.warning("kafka-python not installed. Event logging disabled.")
            return False

        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        if not bootstrap_servers:
            logger.warning("KAFKA_BOOTSTRAP_SERVERS not set. Event logging disabled.")
            return False

        try:
            logger.info(
                f"Initializing KafkaEventLogger producer for topic '{self.topic}'..."
            )
            producer_config = {
                "bootstrap_servers": bootstrap_servers.split(","),
                "value_serializer": lambda v: json.dumps(v, default=str).encode(
                    "utf-8"
                ),
                "key_serializer": lambda k: k.encode("utf-8") if k else None,
                "retries": 3,
                "request_timeout_ms": 15000,
                "acks": 1,  # Faster delivery for real-time events
                "linger_ms": 10,  # Small batching for efficiency
                "batch_size": 1024,  # Small batches for real-time feel
            }

            if os.getenv("KAFKA_USE_SSL", "true").lower() == "true":
                producer_config["security_protocol"] = "SSL"

            self.producer = KafkaProducer(**producer_config)
            logger.info(f"KafkaEventLogger producer connected successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize KafkaEventLogger producer: {e}")
            self.producer = None
            return False

    def _get_next_sequence(self) -> int:
        """Get next event sequence number."""
        self._event_sequence += 1
        return self._event_sequence

    def _create_base_event(self, message: str) -> Dict[str, Any]:
        """Create simplified event structure with only essential data."""
        # Extract encrypted_payload from user_context if available
        encrypted_payload = "N/A"
        if self.user_context and isinstance(self.user_context, dict):
            encrypted_payload = self.user_context.get("encrypted_payload", "N/A")

        return {
            "encrypted_payload": encrypted_payload,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "kafka_topic_name": self.topic,
            "type": "agent-event",
            "server_name": self.server_name
        }

    def _send_event(self, event_data: Dict[str, Any]) -> bool:
        """Send event to Kafka topic."""
        if not self.producer:
            with self._lock:
                if not self.producer:
                    if not self._initialize_producer():
                        return False

        if not self.producer:
            logger.debug("Event producer not available. Event not sent.")
            return False

        try:
            # Enhanced terminal logging for developer visibility
            print("--- A2A EVENT ---")
            print(f"Topic: '{self.topic}'")
            print(f"Full JSON: {json.dumps(event_data, indent=2)}")
            print("-----------------")

            # Debug output for development
            logger.debug(f"[EVENT] {event_data.get('message', 'No message')}")

            # Send to Kafka with simplified message key
            message_key = f"{self.session_id}_{event_data['timestamp']}"
            future = self.producer.send(
                self.topic, value=event_data, key=message_key
            )

            # Non-blocking send for real-time performance
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)

            return True

        except Exception as e:
            logger.error(f"Error sending event: {e}")
            return False

    def _on_send_success(self, record_metadata):
        """Callback for successful event sends."""
        logger.debug(
            f"Event sent to {record_metadata.topic}:{record_metadata.partition}"
        )

    def _on_send_error(self, exception):
        """Callback for failed event sends."""
        logger.error(f"Failed to send event: {exception}")

    # PUBLIC API METHODS

    def log_event(self, message: str) -> bool:
        """
        Log a general event.

        Args:
            message (str): Event message

        Returns:
            bool: True if event was sent successfully
        """
        event_data = self._create_base_event(message)
        return self._send_event(event_data)

    def log_progress(
        self, message: str, progress_percent: Union[int, float] = None
    ) -> bool:
        """
        Log progress events.

        Args:
            message (str): Progress message
            progress_percent (Union[int, float]): Optional progress percentage (0-100)
        """
        # Include progress percentage in message if provided
        if progress_percent is not None:
            full_message = f"{message} ({progress_percent}%)"
        else:
            full_message = message

        event_data = self._create_base_event(full_message)
        return self._send_event(event_data)

    def close(self):
        """Close the Kafka producer."""
        if self.producer:
            try:
                self.producer.flush(timeout=5)
                self.producer.close()
                logger.info("KafkaEventLogger producer closed.")
            except Exception as e:
                logger.error(f"Error closing KafkaEventLogger producer: {e}")


# Factory functions for easy instantiation
def create_event_logger(
    session_id: str = None, user_context: Dict[str, Any] = None, auth_token: str = None
) -> KafkaEventLogger:
    """
    Factory function to create a KafkaEventLogger instance.

    Args:
        session_id (str): Session identifier
        user_context (Dict): User context information
        auth_token (str): Authentication token for user context

    Returns:
        KafkaEventLogger: Configured event logger instance
    """
    if auth_token and not user_context:
        user_context = _extract_user_context_from_token(auth_token)
    return KafkaEventLogger(session_id=session_id, user_context=user_context)


def create_response_logger() -> KafkaResponseLogger:
    """
    Factory function to create a KafkaResponseLogger instance.

    Returns:
        KafkaResponseLogger: Configured response logger instance
    """
    return KafkaResponseLogger()


# Singleton instance for token usage logging
kafka_logger = KafkaLogger()
atexit.register(kafka_logger.close)
