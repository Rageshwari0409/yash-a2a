import asyncio
import logging
import traceback
from typing import List, Union, AsyncIterable

from common.types import (
    SendTaskRequest,
    TaskSendParams,
    Message,
    TaskStatus,
    Artifact,
    TextPart,
    FilePart,
    FileContent,
    AuthPart,
    TaskState,
    SendTaskResponse,
    InternalError,
    JSONRPCResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    Task,
    TaskIdParams,
    PushNotificationConfig,
    InvalidParamsError,
)
from common.server.task_manager import InMemoryTaskManager
from agents.obligation_user_story.agent import MdocUserAgent
from common.utils.push_notification_auth import PushNotificationSenderAuth
import common.server.utils as utils

from utils.kafka import create_event_logger, create_response_logger
from utils.event_messages import EventMessages


logger = logging.getLogger(__name__)


class AgentTaskManager(InMemoryTaskManager):
    def __init__(
        self, agent: MdocUserAgent, notification_sender_auth: PushNotificationSenderAuth
    ):
        super().__init__()
        self.agent = agent
        self.notification_sender_auth = notification_sender_auth

    async def _run_streaming_agent(self, request: SendTaskStreamingRequest):
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        
        session_id = task_send_params.sessionId
        user_context = self._extract_user_context_from_request(request)
        event_logger = create_event_logger(
            session_id=session_id, user_context=user_context
        )
        response_logger = create_response_logger()
        auth_token = query.get("auth_token")

        event_logger.log_event(EventMessages.PROCESSING_REQUEST)

        try:
            async for item in self.agent.stream(query, task_send_params.sessionId):
                response_logger.log(item, auth_token)
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]
                artifact = None
                message = None
                parts = [{"type": "text", "text": item["content"]}]
                end_stream = False

                if not is_task_complete and not require_user_input:
                    task_state = TaskState.WORKING
                    message = Message(role="agent", parts=parts)
                    event_logger.log_event(EventMessages.PROCESSING_REQUEST)
                elif require_user_input:
                    task_state = TaskState.INPUT_REQUIRED
                    message = Message(role="agent", parts=parts)
                    end_stream = True
                else:
                    task_state = TaskState.COMPLETED
                    artifact = Artifact(parts=parts, index=0, append=False)
                    end_stream = True
                    event_logger.log_event(EventMessages.RESPONSE_SENT)
                    event_logger.log_event(EventMessages.TASK_COMPLETED)

                task_status = TaskStatus(state=task_state, message=message)
                latest_task = await self.update_store(
                    task_send_params.id,
                    task_status,
                    None if artifact is None else [artifact],
                )
                await self.send_task_notification(latest_task)

                if artifact:
                    task_artifact_update_event = TaskArtifactUpdateEvent(
                        id=task_send_params.id, artifact=artifact
                    )
                    await self.enqueue_events_for_sse(
                        task_send_params.id, task_artifact_update_event
                    )                    
                    

                task_update_event = TaskStatusUpdateEvent(
                    id=task_send_params.id, status=task_status, final=end_stream
                )
                await self.enqueue_events_for_sse(
                    task_send_params.id, task_update_event
                )

        except Exception as e:
            logger.error(f"An error occurred while streaming the response: {e}")
            error_response = {
                "error": True,
                "error_message": str(e),
                "error_type": type(e).__name__,
                "streaming": True,
            }
            response_logger.log(error_response, auth_token)
            event_logger.log_event(EventMessages.ERROR_STREAMING.format(error=e))
            await self.enqueue_events_for_sse(
                task_send_params.id,
                InternalError(message=f"An error occurred while streaming the response: {e}")                
            )

    def _validate_request(
        self, request: Union[SendTaskRequest, SendTaskStreamingRequest]
    ) -> JSONRPCResponse | None:
        task_send_params: TaskSendParams = request.params
        if not utils.are_modalities_compatible(
            task_send_params.acceptedOutputModes, MdocUserAgent.SUPPORTED_CONTENT_TYPES
        ):
            logger.warning(
                "Unsupported output mode. Received %s, Support %s",
                task_send_params.acceptedOutputModes,
                MdocUserAgent.SUPPORTED_CONTENT_TYPES,
            )
            return utils.new_incompatible_types_error(request.id)

        if (
            task_send_params.pushNotification
            and not task_send_params.pushNotification.url
        ):
            logger.warning("Push notification URL is missing")
            return JSONRPCResponse(
                id=request.id,
                error=InvalidParamsError(message="Push notification URL is missing"),
            )

        return None

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handles the 'send task' request."""
        session_id = request.params.sessionId
        user_context = self._extract_user_context_from_request(request)
        event_logger = create_event_logger(
            session_id=session_id, user_context=user_context
        )
        
        validation_error = self._validate_request(request)
        if validation_error:
            event_logger.log_event(EventMessages.ERROR_INVALID_INPUT)
            return SendTaskResponse(id=request.id, error=validation_error.error)
        
        if request.params.pushNotification:
            if not await self.set_push_notification_info(request.params.id, request.params.pushNotification):
                return SendTaskResponse(id=request.id, error=InvalidParamsError(message="Push notification URL is invalid"))
        
        await self.upsert_task(request.params)
        task = await self.update_store(
            request.params.id, TaskStatus(state=TaskState.WORKING), None
        )
        await self.send_task_notification(task)

        task_send_params: TaskSendParams = request.params

        logging.info("Query Extraction Started!!")
        query = self._get_user_query(task_send_params)
        logging.info(f"Query Extraction Ended!! {query}")
        
        event_logger.log_event(EventMessages.ANALYZING_REQUEST)
        
        try:
            agent_response = await self.agent.invoke(query, task_send_params.sessionId)
            print(agent_response)
            
            response_logger = create_response_logger()
            auth_token = query.get("auth_token")
            response_logger.log(agent_response, auth_token)
            
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            
            response_logger = create_response_logger()
            auth_token = query.get("auth_token")
            error_response = {
                "error": True,
                "error_message": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }
            response_logger.log(error_response, auth_token)
            
            event_logger.log_event(EventMessages.ERROR_TASK_FAILED.format(error=e))
            raise ValueError(f"Error invoking agent: {e}")
        return await self._process_agent_response(
            request, agent_response, event_logger
        )

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        try:
            error = self._validate_request(request)
            if error:
                return error

            await self.upsert_task(request.params)

            if request.params.pushNotification:
                if not await self.set_push_notification_info(
                    request.params.id, request.params.pushNotification
                ):
                    return JSONRPCResponse(
                        id=request.id,
                        error=InvalidParamsError(
                            message="Push notification URL is invalid"
                        ),
                    )

            task_send_params: TaskSendParams = request.params
            sse_event_queue = await self.setup_sse_consumer(task_send_params.id, False)

            asyncio.create_task(self._run_streaming_agent(request))

            return self.dequeue_events_for_sse(
                request.id, task_send_params.id, sse_event_queue
            )
        except Exception as e:
            logger.error(f"Error in SSE stream: {e}")
            print(traceback.format_exc())
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message="An error occurred while streaming the response"
                ),
            )

    async def _process_agent_response(
        self, request: SendTaskRequest, agent_response: dict, event_logger
    ) -> SendTaskResponse:
        """Processes the agent's response and updates the task store."""
        task_send_params: TaskSendParams = request.params
        task_id = task_send_params.id
    
        task_status = None

        parts = [{"type": "text", "text": agent_response["content"]}]

        # Additing File part
        logging.info(f"Agent Response: {agent_response}")
        if 'files' in agent_response and isinstance(agent_response['files'], List):
            for file_path in agent_response['files']:
                try:                    
                    parts.append(
                        FilePart(
                            file = FileContent(
                                uri=file_path,
                            )
                        )
                    )
                except Exception as e:
                    logging.error(f"No file path found {file_path}: {e}")

        artifact = None
        if agent_response["require_user_input"]:
            task_status = TaskStatus(
                state=TaskState.INPUT_REQUIRED,
                message=Message(role="agent", parts=parts),
            )
        else:
            task_status = TaskStatus(state=TaskState.COMPLETED)
            artifact = Artifact(parts=parts, description="Final output of the query")
        task = await self.update_store(
            task_id, task_status, None if artifact is None else [artifact]
        )
        history_length = len(task.history)
        task_result = self.append_task_history(task, history_length)
        await self.send_task_notification(task)

        event_logger.log_event(EventMessages.RESPONSE_SENT)
        if task_status.state == TaskState.COMPLETED:
            event_logger.log_event(EventMessages.TASK_COMPLETED)

        return SendTaskResponse(id=request.id, result=task_result)

    def _get_user_query(self, task_send_params: TaskSendParams) -> dict:
        parts = task_send_params.message.parts

        query = {
            "text": [],
            "files": [],
            "auth_token": None,
            "user_metadata": task_send_params.metadata.get("user_metadata","")
        }

        for part in parts:
            if isinstance(part, TextPart):
                query['text'].append(part.text)
            elif isinstance(part, FilePart):
                query['files'].append({
                    'uri': part.file.uri,
                    'mimeType': part.file.mimeType
                })
            elif isinstance(part, AuthPart):
                query['auth_token'] = part.token
                logger.info(f"Auth token extracted for task: {task_send_params.id}")
            else:
                raise ValueError("This Message type is not supported")
            
        query['text'] = "\n".join(query['text'])
        return query

    def _extract_user_context_from_request(
        self, request: Union[SendTaskRequest, SendTaskStreamingRequest]
    ) -> dict:
        """Extract user context from request for event logging."""
        user_context = {}

        # Try to extract user info from auth token if available
        task_send_params: TaskSendParams = request.params
        parts = task_send_params.message.parts

        for part in parts:
            if isinstance(part, AuthPart):
                # Extract user context from auth token (basic extraction)
                try:
                    import jwt

                    # Extract encrypted payload and JWT part
                    jwt_part = part.token
                    encrypted_payload = "N/A"

                    if "$YashUnified2025$" in part.token:
                        jwt_part, encrypted_payload = part.token.split(
                            "$YashUnified2025$", 1
                        )

                    if jwt_part.lower().startswith("bearer "):
                        jwt_part = jwt_part[7:]

                    # Decode JWT without verification for logging purposes
                    decoded_token = jwt.decode(
                        jwt_part, options={"verify_signature": False}
                    )
                    custom_data = decoded_token.get("custom-data", {})

                    user_context["user_email"] = (
                        custom_data.get("user_email")
                        or decoded_token.get("email")
                        or decoded_token.get("upn")
                        or decoded_token.get("preferred_username")
                        or "N/A"
                    )
                    user_context["user_id"] = decoded_token.get("sub", "N/A")
                    user_context["encrypted_payload"] = encrypted_payload

                except Exception as e:
                    logger.debug(f"Could not extract user context from token: {e}")
                    user_context = {"user_email": "N/A", "user_id": "N/A"}
                break

        return user_context

    async def send_task_notification(self, task: Task):
        if not await self.has_push_notification_info(task.id):
            logger.info(f"No push notification info found for task {task.id}")
            return
        push_info = await self.get_push_notification_info(task.id)

        logger.info(f"Notifying for task {task.id} => {task.status.state}")
        await self.notification_sender_auth.send_push_notification(
            push_info.url, data=task.model_dump(exclude_none=True)
        )

    async def on_resubscribe_to_task(
        self, request
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        task_id_params: TaskIdParams = request.params
        try:
            sse_event_queue = await self.setup_sse_consumer(task_id_params.id, True)
            return self.dequeue_events_for_sse(
                request.id, task_id_params.id, sse_event_queue
            )
        except Exception as e:
            logger.error(f"Error while reconnecting to SSE stream: {e}")
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message=f"An error occurred while reconnecting to stream: {e}"
                ),
            )

    async def set_push_notification_info(
        self, task_id: str, push_notification_config: PushNotificationConfig
    ):
        # Verify the ownership of notification URL by issuing a challenge request.
        is_verified = await self.notification_sender_auth.verify_push_notification_url(
            push_notification_config.url
        )
        if not is_verified:
            return False

        await super().set_push_notification_info(task_id, push_notification_config)
        return True
