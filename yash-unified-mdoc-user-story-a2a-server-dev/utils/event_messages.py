# utils/event_messages.py
"""
Constants file containing predefined messages for the custom Kafka event logger.
These messages provide user-friendly descriptions of what the system is doing.
"""


class EventMessages:
    """
    Centralized collection of event messages for the Kafka event logger.
    """

    # General Task Flow
    ANALYZING_REQUEST = "Analyzing the request..."
    PROCESSING_REQUEST = "Processing the request..."
    RESPONSE_SENT = "Response sent."
    TASK_COMPLETED = "Task completed successfully."

    # Error Messages
    ERROR_INVALID_INPUT = "Invalid input received."
    ERROR_TASK_FAILED = "Task failed due to: {error}"
    ERROR_STREAMING = "Error: {error}"
