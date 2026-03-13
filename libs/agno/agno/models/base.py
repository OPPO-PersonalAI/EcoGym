import asyncio
import collections.abc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import AsyncGeneratorType, GeneratorType
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
)
from uuid import uuid4

from pydantic import BaseModel

from agno.exceptions import AgentRunException
from agno.media import Audio, File, Image, Video
from agno.models.message import Citations, Message
from agno.models.metrics import Metrics
from agno.models.response import ModelResponse, ModelResponseEvent, ToolExecution
from agno.run.agent import CustomEvent, RunContentEvent, RunOutput, RunOutputEvent
from agno.run.team import RunContentEvent as TeamRunContentEvent
from agno.run.team import TeamRunOutputEvent
from agno.tools.function import Function, FunctionCall, FunctionExecutionResult, UserInputField
from agno.utils.log import log_debug, log_error, log_warning
from agno.utils.timer import Timer
from agno.utils.tools import get_function_call_for_tool_call, get_function_call_for_tool_execution


@dataclass
class MessageData:
    response_role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    response_content: Any = ""
    response_reasoning_content: Any = ""
    response_redacted_reasoning_content: Any = ""
    response_citations: Optional[Citations] = None
    response_tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    response_audio: Optional[Audio] = None
    response_image: Optional[Image] = None
    response_video: Optional[Video] = None
    response_file: Optional[File] = None

    response_provider_data: Optional[Dict[str, Any]] = None

    extra: Optional[Dict[str, Any]] = None


def _log_messages(messages: List[Message]) -> None:
    """
    Log messages for debugging.
    """
    for m in messages:
        print(type(m), m, '\n\n')
        m.log(metrics=False)


def _handle_agent_exception(a_exc: AgentRunException, additional_input: Optional[List[Message]] = None) -> None:
    """Handle AgentRunException and collect additional messages."""
    if additional_input is None:
        additional_input = []
    if a_exc.user_message is not None:
        msg = (
            Message(role="user", content=a_exc.user_message)
            if isinstance(a_exc.user_message, str)
            else a_exc.user_message
        )
        additional_input.append(msg)

    if a_exc.agent_message is not None:
        msg = (
            Message(role="assistant", content=a_exc.agent_message)
            if isinstance(a_exc.agent_message, str)
            else a_exc.agent_message
        )
        additional_input.append(msg)

    if a_exc.messages:
        for m in a_exc.messages:
            if isinstance(m, Message):
                additional_input.append(m)
            elif isinstance(m, dict):
                try:
                    additional_input.append(Message(**m))
                except Exception as e:
                    log_warning(f"Failed to convert dict to Message: {e}")

    if a_exc.stop_execution:
        for m in additional_input:
            m.stop_after_tool_call = True


@dataclass
class Model(ABC):
    id: str
    name: Optional[str] = None
    provider: Optional[str] = None

    supports_native_structured_outputs: bool = False
    
    supports_json_schema_outputs: bool = False


    _tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    system_prompt: Optional[str] = None
    instructions: Optional[List[str]] = None

    tool_message_role: str = "tool"
    assistant_message_role: str = "assistant"

    def __post_init__(self):
        if self.provider is None and self.name is not None:
            self.provider = f"{self.name} ({self.id})"

    def to_dict(self) -> Dict[str, Any]:
        fields = {"name", "id", "provider"}
        _dict = {field: getattr(self, field) for field in fields if getattr(self, field) is not None}
        return _dict

    def get_provider(self) -> str:
        return self.provider or self.name or self.__class__.__name__

    @abstractmethod
    def invoke(self, *args, **kwargs) -> ModelResponse:
        pass

    @abstractmethod
    async def ainvoke(self, *args, **kwargs) -> ModelResponse:
        pass

    @abstractmethod
    def invoke_stream(self, *args, **kwargs) -> Iterator[ModelResponse]:
        pass

    @abstractmethod
    def ainvoke_stream(self, *args, **kwargs) -> AsyncIterator[ModelResponse]:
        pass

    @abstractmethod
    def _parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        """
        Parse the raw response from the model provider into a ModelResponse.

        Args:
            response: Raw response from the model provider
            **kwargs: Additional keyword arguments

        Returns:
            ModelResponse: Parsed response data
        """
        pass

    @abstractmethod
    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        pass

    def response(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        functions: Optional[Dict[str, Function]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_call_limit: Optional[int] = None,
        run_response: Optional[RunOutput] = None,
        send_media_to_model: bool = True,
    ) -> ModelResponse:
        log_debug(f"{self.get_provider()} Response Start", center=True, symbol="-")
        log_debug(f"Model: {self.id}", center=True, symbol="-")

        _log_messages(messages)
        model_response = ModelResponse()

        function_call_count = 0

        while True:
            assistant_message = Message(role=self.assistant_message_role)
            self._process_model_response(
                messages=messages,
                assistant_message=assistant_message,
                model_response=model_response,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice or self._tool_choice,
                run_response=run_response,
            )

            messages.append(assistant_message)

            assistant_message.log(metrics=True)

            if assistant_message.tool_calls:
                function_calls_to_run = self._prepare_function_calls(
                    assistant_message=assistant_message,
                    messages=messages,
                    model_response=model_response,
                    functions=functions,
                )
                function_call_results: List[Message] = []

                for function_call_response in self.run_function_calls(
                    function_calls=function_calls_to_run,
                    function_call_results=function_call_results,
                    current_function_call_count=function_call_count,
                    function_call_limit=tool_call_limit,
                ):
                    if isinstance(function_call_response, ModelResponse):
                        if function_call_response.updated_session_state is not None:
                            model_response.updated_session_state = function_call_response.updated_session_state

                        if function_call_response.images is not None:
                            if model_response.images is None:
                                model_response.images = []
                            model_response.images.extend(function_call_response.images)

                        if function_call_response.audios is not None:
                            if model_response.audios is None:
                                model_response.audios = []
                            model_response.audios.extend(function_call_response.audios)

                        if function_call_response.videos is not None:
                            if model_response.videos is None:
                                model_response.videos = []
                            model_response.videos.extend(function_call_response.videos)

                        if function_call_response.files is not None:
                            if model_response.files is None:
                                model_response.files = []
                            model_response.files.extend(function_call_response.files)

                        if (
                            function_call_response.event
                            in [
                                ModelResponseEvent.tool_call_completed.value,
                                ModelResponseEvent.tool_call_paused.value,
                            ]
                            and function_call_response.tool_executions is not None
                        ):
                            if model_response.tool_executions is None:
                                model_response.tool_executions = []
                            model_response.tool_executions.extend(function_call_response.tool_executions)

                        elif function_call_response.event not in [
                            ModelResponseEvent.tool_call_started.value,
                            ModelResponseEvent.tool_call_completed.value,
                        ]:
                            if function_call_response.content:
                                model_response.content += function_call_response.content  # type: ignore

                function_call_count += len(function_call_results)

                self.format_function_call_results(
                    messages=messages, function_call_results=function_call_results, **model_response.extra or {}
                )

                if any(msg.images or msg.videos or msg.audio or msg.files for msg in function_call_results):
                    self._handle_function_call_media(
                        messages=messages,
                        function_call_results=function_call_results,
                        send_media_to_model=send_media_to_model,
                    )

                for function_call_result in function_call_results:
                    function_call_result.log(metrics=True)

                if any(m.stop_after_tool_call for m in function_call_results):
                    break

                if any(tc.requires_confirmation for tc in model_response.tool_executions or []):
                    break

                if any(tc.external_execution_required for tc in model_response.tool_executions or []):
                    break

                if any(tc.requires_user_input for tc in model_response.tool_executions or []):
                    break

                continue

            break

        log_debug(f"{self.get_provider()} Response End", center=True, symbol="-")
        return model_response

    async def aresponse(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        functions: Optional[Dict[str, Function]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_call_limit: Optional[int] = None,
        send_media_to_model: bool = True,
    ) -> ModelResponse:
        log_debug(f"{self.get_provider()} Async Response Start", center=True, symbol="-")
        log_debug(f"Model: {self.id}", center=True, symbol="-")
        _log_messages(messages)
        model_response = ModelResponse()

        function_call_count = 0

        while True:
            assistant_message = Message(role=self.assistant_message_role)
            await self._aprocess_model_response(
                messages=messages,
                assistant_message=assistant_message,
                model_response=model_response,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice or self._tool_choice,
            )

            messages.append(assistant_message)

            assistant_message.log(metrics=True)

            if assistant_message.tool_calls:
                function_calls_to_run = self._prepare_function_calls(
                    assistant_message=assistant_message,
                    messages=messages,
                    model_response=model_response,
                    functions=functions,
                )
                function_call_results: List[Message] = []

                async for function_call_response in self.arun_function_calls(
                    function_calls=function_calls_to_run,
                    function_call_results=function_call_results,
                    current_function_call_count=function_call_count,
                    function_call_limit=tool_call_limit,
                ):
                    if isinstance(function_call_response, ModelResponse):
                        if function_call_response.updated_session_state is not None:
                            model_response.updated_session_state = function_call_response.updated_session_state

                        if function_call_response.images is not None:
                            if model_response.images is None:
                                model_response.images = []
                            model_response.images.extend(function_call_response.images)

                        if function_call_response.audios is not None:
                            if model_response.audios is None:
                                model_response.audios = []
                            model_response.audios.extend(function_call_response.audios)

                        if function_call_response.videos is not None:
                            if model_response.videos is None:
                                model_response.videos = []
                            model_response.videos.extend(function_call_response.videos)

                        if function_call_response.files is not None:
                            if model_response.files is None:
                                model_response.files = []
                            model_response.files.extend(function_call_response.files)

                        if (
                            function_call_response.event
                            in [
                                ModelResponseEvent.tool_call_completed.value,
                                ModelResponseEvent.tool_call_paused.value,
                            ]
                            and function_call_response.tool_executions is not None
                        ):
                            if model_response.tool_executions is None:
                                model_response.tool_executions = []
                            model_response.tool_executions.extend(function_call_response.tool_executions)
                        elif function_call_response.event not in [
                            ModelResponseEvent.tool_call_started.value,
                            ModelResponseEvent.tool_call_completed.value,
                        ]:
                            if function_call_response.content:
                                model_response.content += function_call_response.content  

                function_call_count += len(function_call_results)

                self.format_function_call_results(
                    messages=messages, function_call_results=function_call_results, **model_response.extra or {}
                )

                if any(msg.images or msg.videos or msg.audio or msg.files for msg in function_call_results):
                    self._handle_function_call_media(
                        messages=messages,
                        function_call_results=function_call_results,
                        send_media_to_model=send_media_to_model,
                    )

                for function_call_result in function_call_results:
                    function_call_result.log(metrics=True)

                if any(m.stop_after_tool_call for m in function_call_results):
                    break

                if any(tc.requires_confirmation for tc in model_response.tool_executions or []):
                    break

                if any(tc.external_execution_required for tc in model_response.tool_executions or []):
                    break

                if any(tc.requires_user_input for tc in model_response.tool_executions or []):
                    break

                continue


            break

        log_debug(f"{self.get_provider()} Async Response End", center=True, symbol="-")
        return model_response

    def _process_model_response(
        self,
        messages: List[Message],
        assistant_message: Message,
        model_response: ModelResponse,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[RunOutput] = None,
    ) -> None:
        provider_response = self.invoke(
            assistant_message=assistant_message,
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice or self._tool_choice,
            run_response=run_response,
        )

        self._populate_assistant_message(assistant_message=assistant_message, provider_response=provider_response)

        if assistant_message.content is not None:
            if model_response.content is None:
                model_response.content = assistant_message.get_content_string()
            else:
                model_response.content += assistant_message.get_content_string()
        if assistant_message.reasoning_content is not None:
            model_response.reasoning_content = assistant_message.reasoning_content
        if assistant_message.redacted_reasoning_content is not None:
            model_response.redacted_reasoning_content = assistant_message.redacted_reasoning_content
        if assistant_message.citations is not None:
            model_response.citations = assistant_message.citations
        if assistant_message.audio_output is not None:
            if isinstance(assistant_message.audio_output, Audio):
                model_response.audio = assistant_message.audio_output
        if assistant_message.image_output is not None:
            model_response.images = [assistant_message.image_output]
        if assistant_message.video_output is not None:
            model_response.videos = [assistant_message.video_output]
        if provider_response.extra is not None:
            if model_response.extra is None:
                model_response.extra = {}
            model_response.extra.update(provider_response.extra)

    async def _aprocess_model_response(
        self,
        messages: List[Message],
        assistant_message: Message,
        model_response: ModelResponse,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[RunOutput] = None,
    ) -> None:
        provider_response = await self.ainvoke(
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice or self._tool_choice,
            assistant_message=assistant_message,
            run_response=run_response,
        )

        self._populate_assistant_message(assistant_message=assistant_message, provider_response=provider_response)

        if assistant_message.content is not None:
            if model_response.content is None:
                model_response.content = assistant_message.get_content_string()
            else:
                model_response.content += assistant_message.get_content_string()
        if assistant_message.reasoning_content is not None:
            model_response.reasoning_content = assistant_message.reasoning_content
        if assistant_message.redacted_reasoning_content is not None:
            model_response.redacted_reasoning_content = assistant_message.redacted_reasoning_content
        if assistant_message.citations is not None:
            model_response.citations = assistant_message.citations
        if assistant_message.audio_output is not None:
            if isinstance(assistant_message.audio_output, Audio):
                model_response.audio = assistant_message.audio_output
        if assistant_message.image_output is not None:
            model_response.images = [assistant_message.image_output]
        if assistant_message.video_output is not None:
            model_response.videos = [assistant_message.video_output]
        if provider_response.extra is not None:
            if model_response.extra is None:
                model_response.extra = {}
            model_response.extra.update(provider_response.extra)

    def _populate_assistant_message(
        self,
        assistant_message: Message,
        provider_response: ModelResponse,
    ) -> Message:
        if provider_response.role is not None:
            assistant_message.role = provider_response.role

        if provider_response.content is not None:
            assistant_message.content = provider_response.content

        if provider_response.tool_calls is not None and len(provider_response.tool_calls) > 0:
            assistant_message.tool_calls = provider_response.tool_calls

        if provider_response.audio is not None:
            assistant_message.audio_output = provider_response.audio

        if provider_response.images is not None:
            if provider_response.images:
                assistant_message.image_output = provider_response.images[-1]  

        if provider_response.videos is not None:
            if provider_response.videos:
                assistant_message.video_output = provider_response.videos[-1]  

        if provider_response.files is not None:
            if provider_response.files:
                assistant_message.file_output = provider_response.files[-1]  

        if provider_response.audios is not None:
            if provider_response.audios:
                assistant_message.audio_output = provider_response.audios[-1]  

        if provider_response.redacted_reasoning_content is not None:
            assistant_message.redacted_reasoning_content = provider_response.redacted_reasoning_content

        if provider_response.reasoning_content is not None:
            assistant_message.reasoning_content = provider_response.reasoning_content

        if provider_response.provider_data is not None:
            assistant_message.provider_data = provider_response.provider_data

        if provider_response.citations is not None:
            assistant_message.citations = provider_response.citations

        if provider_response.response_usage is not None:
            assistant_message.metrics += provider_response.response_usage

        return assistant_message

    def process_response_stream(
        self,
        messages: List[Message],
        assistant_message: Message,
        stream_data: MessageData,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[RunOutput] = None,
    ) -> Iterator[ModelResponse]:
        for response_delta in self.invoke_stream(
            messages=messages,
            assistant_message=assistant_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice or self._tool_choice,
            run_response=run_response,
        ):
            yield from self._populate_stream_data_and_assistant_message(
                stream_data=stream_data,
                assistant_message=assistant_message,
                model_response_delta=response_delta,
            )

        self._populate_assistant_message(assistant_message=assistant_message, provider_response=response_delta)

    def response_stream(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        functions: Optional[Dict[str, Function]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_call_limit: Optional[int] = None,
        stream_model_response: bool = True,
        run_response: Optional[RunOutput] = None,
        send_media_to_model: bool = True,
    ) -> Iterator[Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]]:
        log_debug(f"{self.get_provider()} Response Stream Start", center=True, symbol="-")
        log_debug(f"Model: {self.id}", center=True, symbol="-")
        _log_messages(messages)

        function_call_count = 0

        while True:
            assistant_message = Message(role=self.assistant_message_role)
            stream_data = MessageData()
            if stream_model_response:
                yield from self.process_response_stream(
                    messages=messages,
                    assistant_message=assistant_message,
                    stream_data=stream_data,
                    response_format=response_format,
                    tools=tools,
                    tool_choice=tool_choice or self._tool_choice,   
                    run_response=run_response,
                )

                if stream_data.response_content:
                    assistant_message.content = stream_data.response_content
                if stream_data.response_reasoning_content:
                    assistant_message.reasoning_content = stream_data.response_reasoning_content
                if stream_data.response_redacted_reasoning_content:
                    assistant_message.redacted_reasoning_content = stream_data.response_redacted_reasoning_content
                if stream_data.response_provider_data:
                    assistant_message.provider_data = stream_data.response_provider_data
                if stream_data.response_citations:
                    assistant_message.citations = stream_data.response_citations
                if stream_data.response_audio:
                    assistant_message.audio_output = stream_data.response_audio
                if stream_data.response_tool_calls and len(stream_data.response_tool_calls) > 0:
                    assistant_message.tool_calls = self.parse_tool_calls(stream_data.response_tool_calls)

            else:
                model_response = ModelResponse()
                self._process_model_response(
                    messages=messages,
                    assistant_message=assistant_message,
                    model_response=model_response,
                    response_format=response_format,
                    tools=tools,
                    tool_choice=tool_choice or self._tool_choice,
                )
                yield model_response

            messages.append(assistant_message)
            assistant_message.log(metrics=True)

            if assistant_message.tool_calls is not None:
                function_calls_to_run: List[FunctionCall] = self.get_function_calls_to_run(
                    assistant_message, messages, functions
                )
                function_call_results: List[Message] = []

                for function_call_response in self.run_function_calls(
                    function_calls=function_calls_to_run,
                    function_call_results=function_call_results,
                    current_function_call_count=function_call_count,
                    function_call_limit=tool_call_limit,
                ):
                    yield function_call_response

                function_call_count += len(function_call_results)

                if stream_data and stream_data.extra is not None:
                    self.format_function_call_results(
                        messages=messages, function_call_results=function_call_results, **stream_data.extra
                    )
                else:
                    self.format_function_call_results(messages=messages, function_call_results=function_call_results)

                if any(msg.images or msg.videos or msg.audio for msg in function_call_results):
                    self._handle_function_call_media(
                        messages=messages,
                        function_call_results=function_call_results,
                        send_media_to_model=send_media_to_model,
                    )

                for function_call_result in function_call_results:
                    function_call_result.log(metrics=True)

                if any(m.stop_after_tool_call for m in function_call_results):
                    break

                if any(fc.function.requires_confirmation for fc in function_calls_to_run):
                    break

                if any(fc.function.external_execution for fc in function_calls_to_run):
                    break

                if any(fc.function.requires_user_input for fc in function_calls_to_run):
                    break

                continue

            break

        log_debug(f"{self.get_provider()} Response Stream End", center=True, symbol="-")

    async def aprocess_response_stream(
        self,
        messages: List[Message],
        assistant_message: Message,
        stream_data: MessageData,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[RunOutput] = None,
    ) -> AsyncIterator[ModelResponse]:
        async for response_delta in self.ainvoke_stream(
            messages=messages,
            assistant_message=assistant_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice or self._tool_choice,
            run_response=run_response,
        ):  
            for model_response in self._populate_stream_data_and_assistant_message(
                stream_data=stream_data,
                assistant_message=assistant_message,
                model_response_delta=response_delta,
            ):
                yield model_response

        self._populate_assistant_message(assistant_message=assistant_message, provider_response=model_response)

    async def aresponse_stream(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        functions: Optional[Dict[str, Function]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_call_limit: Optional[int] = None,
        stream_model_response: bool = True,
        run_response: Optional[RunOutput] = None,
        send_media_to_model: bool = True,
    ) -> AsyncIterator[Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]]:
        log_debug(f"{self.get_provider()} Async Response Stream Start", center=True, symbol="-")
        log_debug(f"Model: {self.id}", center=True, symbol="-")
        _log_messages(messages)

        function_call_count = 0

        while True:
            
            assistant_message = Message(role=self.assistant_message_role)
            stream_data = MessageData()
            if stream_model_response:
                async for response in self.aprocess_response_stream(
                    messages=messages,
                    assistant_message=assistant_message,
                    stream_data=stream_data,
                    response_format=response_format,
                    tools=tools,
                    tool_choice=tool_choice or self._tool_choice,
                    run_response=run_response,
                ):
                    yield response

                if stream_data.response_content:
                    assistant_message.content = stream_data.response_content
                if stream_data.response_reasoning_content:
                    assistant_message.reasoning_content = stream_data.response_reasoning_content
                if stream_data.response_redacted_reasoning_content:
                    assistant_message.redacted_reasoning_content = stream_data.response_redacted_reasoning_content
                if stream_data.response_provider_data:
                    assistant_message.provider_data = stream_data.response_provider_data
                if stream_data.response_audio:
                    assistant_message.audio_output = stream_data.response_audio
                if stream_data.response_tool_calls and len(stream_data.response_tool_calls) > 0:
                    assistant_message.tool_calls = self.parse_tool_calls(stream_data.response_tool_calls)

            else:
                model_response = ModelResponse()
                await self._aprocess_model_response(
                    messages=messages,
                    assistant_message=assistant_message,
                    model_response=model_response,
                    response_format=response_format,
                    tools=tools,
                    tool_choice=tool_choice or self._tool_choice,
                    run_response=run_response,
                )
                yield model_response

            messages.append(assistant_message)
            assistant_message.log(metrics=True)

            if assistant_message.tool_calls is not None:
                function_calls_to_run: List[FunctionCall] = self.get_function_calls_to_run(
                    assistant_message, messages, functions
                )
                function_call_results: List[Message] = []

                async for function_call_response in self.arun_function_calls(
                    function_calls=function_calls_to_run,
                    function_call_results=function_call_results,
                    current_function_call_count=function_call_count,
                    function_call_limit=tool_call_limit,
                ):
                    yield function_call_response

                function_call_count += len(function_call_results)

                if stream_data and stream_data.extra is not None:
                    self.format_function_call_results(
                        messages=messages, function_call_results=function_call_results, **stream_data.extra
                    )
                else:
                    self.format_function_call_results(messages=messages, function_call_results=function_call_results)

                if any(msg.images or msg.videos or msg.audio for msg in function_call_results):
                    self._handle_function_call_media(
                        messages=messages,
                        function_call_results=function_call_results,
                        send_media_to_model=send_media_to_model,
                    )

                for function_call_result in function_call_results:
                    function_call_result.log(metrics=True)

                if any(m.stop_after_tool_call for m in function_call_results):
                    break

                if any(fc.function.requires_confirmation for fc in function_calls_to_run):
                    break

                if any(fc.function.external_execution for fc in function_calls_to_run):
                    break

                if any(fc.function.requires_user_input for fc in function_calls_to_run):
                    break

                continue

            break

        log_debug(f"{self.get_provider()} Async Response Stream End", center=True, symbol="-")

    def _populate_stream_data_and_assistant_message(
        self, stream_data: MessageData, assistant_message: Message, model_response_delta: ModelResponse
    ) -> Iterator[ModelResponse]:
        if model_response_delta.role is not None:
            assistant_message.role = model_response_delta.role

        should_yield = False
        if model_response_delta.content is not None:
            stream_data.response_content += model_response_delta.content
            should_yield = True

        if model_response_delta.reasoning_content is not None:
            stream_data.response_reasoning_content += model_response_delta.reasoning_content
            should_yield = True

        if model_response_delta.redacted_reasoning_content is not None:
            stream_data.response_redacted_reasoning_content += model_response_delta.redacted_reasoning_content
            should_yield = True

        if model_response_delta.citations is not None:
            stream_data.response_citations = model_response_delta.citations
            should_yield = True

        if model_response_delta.provider_data:
            if stream_data.response_provider_data is None:
                stream_data.response_provider_data = {}
            stream_data.response_provider_data.update(model_response_delta.provider_data)

        if model_response_delta.tool_calls is not None:
            if stream_data.response_tool_calls is None:
                stream_data.response_tool_calls = []
            stream_data.response_tool_calls.extend(model_response_delta.tool_calls)
            should_yield = True

        if model_response_delta.audio is not None and isinstance(model_response_delta.audio, Audio):
            if stream_data.response_audio is None:
                stream_data.response_audio = Audio(id=str(uuid4()), content="", transcript="")

            from typing import cast

            audio_response = cast(Audio, model_response_delta.audio)

            if audio_response.id is not None:
                stream_data.response_audio.id = audio_response.id  
            if audio_response.content is not None:
                stream_data.response_audio.content += audio_response.content  
            if audio_response.transcript is not None:
                stream_data.response_audio.transcript += audio_response.transcript  
            if audio_response.expires_at is not None:
                stream_data.response_audio.expires_at = audio_response.expires_at
            if audio_response.mime_type is not None:
                stream_data.response_audio.mime_type = audio_response.mime_type
            stream_data.response_audio.sample_rate = audio_response.sample_rate
            stream_data.response_audio.channels = audio_response.channels

            should_yield = True

        if model_response_delta.images:
            if stream_data.response_image is None:
                stream_data.response_image = model_response_delta.images[-1]
            should_yield = True

        if model_response_delta.videos:
            if stream_data.response_video is None:
                stream_data.response_video = model_response_delta.videos[-1]
            should_yield = True

        if model_response_delta.extra is not None:
            if stream_data.extra is None:
                stream_data.extra = {}
            for key in model_response_delta.extra:
                if isinstance(model_response_delta.extra[key], list):
                    if not stream_data.extra.get(key):
                        stream_data.extra[key] = []
                    stream_data.extra[key].extend(model_response_delta.extra[key])
                else:
                    stream_data.extra[key] = model_response_delta.extra[key]

        if should_yield:
            yield model_response_delta

    def parse_tool_calls(self, tool_calls_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return tool_calls_data

    def parse_tool_calls(self, tool_calls_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return tool_calls_data

    def get_function_call_to_run_from_tool_execution(
        self,
        tool_execution: ToolExecution,
        functions: Optional[Dict[str, Function]] = None,
    ) -> FunctionCall:
        function_call = get_function_call_for_tool_execution(
            tool_execution=tool_execution,
            functions=functions,
        )
        if function_call is None:
            raise ValueError("Function call not found")
        return function_call

    def get_function_calls_to_run(
        self,
        assistant_message: Message,
        messages: List[Message],
        functions: Optional[Dict[str, Function]] = None,
    ) -> List[FunctionCall]:
        function_calls_to_run: List[FunctionCall] = []
        if assistant_message.tool_calls is not None:
            for tool_call in assistant_message.tool_calls:
                _tool_call_id = tool_call.get("id")
                _function_call = get_function_call_for_tool_call(tool_call, functions)
                if _function_call is None:
                    messages.append(
                        Message(
                            role=self.tool_message_role,
                            tool_call_id=_tool_call_id,
                            content="Error: The requested tool does not exist or is not available.",
                        )
                    )
                    continue
                if _function_call.error is not None:
                    messages.append(
                        Message(role=self.tool_message_role, tool_call_id=_tool_call_id, content=_function_call.error)
                    )
                    continue
                function_calls_to_run.append(_function_call)
        return function_calls_to_run

    def create_function_call_result(
        self,
        function_call: FunctionCall,
        success: bool,
        output: Optional[Union[List[Any], str]] = None,
        timer: Optional[Timer] = None,
        function_execution_result: Optional[FunctionExecutionResult] = None,
    ) -> Message:
        kwargs = {}
        if timer is not None:
            kwargs["metrics"] = Metrics(duration=timer.elapsed)

        images = None
        videos = None
        audios = None

        if success and function_execution_result:
            images = function_execution_result.images
            videos = function_execution_result.videos
            audios = function_execution_result.audios

        return Message(
            role=self.tool_message_role,
            content=output if success else function_call.error,
            tool_call_id=function_call.call_id,
            tool_name=function_call.function.name,
            tool_args=function_call.arguments,
            tool_call_error=not success,
            stop_after_tool_call=function_call.function.stop_after_tool_call,
            images=images,
            videos=videos,
            audio=audios,
            **kwargs,  
        )

    def create_tool_call_limit_error_result(self, function_call: FunctionCall) -> Message:
        return Message(
            role=self.tool_message_role,
            content=f"Tool call limit reached. Tool call {function_call.function.name} not executed. Don't try to execute it again.",
            tool_call_id=function_call.call_id,
            tool_name=function_call.function.name,
            tool_args=function_call.arguments,
            tool_call_error=True,
        )

    def run_function_call(
        self,
        function_call: FunctionCall,
        function_call_results: List[Message],
        additional_input: Optional[List[Message]] = None,
    ) -> Iterator[Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]]:
        function_call_timer = Timer()
        function_call_timer.start()
        yield ModelResponse(
            content=function_call.get_call_str(),
            tool_executions=[
                ToolExecution(
                    tool_call_id=function_call.call_id,
                    tool_name=function_call.function.name,
                    tool_args=function_call.arguments,
                )
            ],
            event=ModelResponseEvent.tool_call_started.value,
        )


        function_execution_result: FunctionExecutionResult = FunctionExecutionResult(status="failure")
        try:
            function_execution_result = function_call.execute()
        except AgentRunException as a_exc:
            _handle_agent_exception(a_exc, additional_input)
        except Exception as e:
            log_error(f"Error executing function {function_call.function.name}: {e}")
            raise e

        function_call_success = function_execution_result.status == "success"

        function_call_timer.stop()

        function_call_output: str = ""

        if isinstance(function_execution_result.result, (GeneratorType, collections.abc.Iterator)):
            for item in function_execution_result.result:
                if isinstance(item, tuple(get_args(RunOutputEvent))) or isinstance(
                    item, tuple(get_args(TeamRunOutputEvent))
                ):
                    if isinstance(item, RunContentEvent) or isinstance(item, TeamRunContentEvent):
                        if item.content is not None and isinstance(item.content, BaseModel):
                            function_call_output += item.content.model_dump_json()
                        else:
                            function_call_output += item.content or ""

                        if function_call.function.show_result:
                            yield ModelResponse(content=item.content)

                        if isinstance(item, CustomEvent):
                            function_call_output += str(item)

                    yield item

                else:
                    function_call_output += str(item)
                    if function_call.function.show_result:
                        yield ModelResponse(content=str(item))
        else:
            from agno.tools.function import ToolResult

            if isinstance(function_execution_result.result, ToolResult):
                tool_result = function_execution_result.result
                function_call_output = tool_result.content

                if tool_result.images:
                    function_execution_result.images = tool_result.images
                if tool_result.videos:
                    function_execution_result.videos = tool_result.videos
                if tool_result.audios:
                    function_execution_result.audios = tool_result.audios
                if tool_result.files:
                    function_execution_result.files = tool_result.files
            else:
                function_call_output = str(function_execution_result.result) if function_execution_result.result else ""

            if function_call.function.show_result:
                yield ModelResponse(content=function_call_output)

        function_call_result = self.create_function_call_result(
            function_call,
            success=function_call_success,
            output=function_call_output,
            timer=function_call_timer,
            function_execution_result=function_execution_result,
        )
        yield ModelResponse(
            content=f"{function_call.get_call_str()} completed in {function_call_timer.elapsed:.4f}s. ",
            tool_executions=[
                ToolExecution(
                    tool_call_id=function_call_result.tool_call_id,
                    tool_name=function_call_result.tool_name,
                    tool_args=function_call_result.tool_args,
                    tool_call_error=function_call_result.tool_call_error,
                    result=str(function_call_result.content),
                    stop_after_tool_call=function_call_result.stop_after_tool_call,
                    metrics=function_call_result.metrics,
                )
            ],
            event=ModelResponseEvent.tool_call_completed.value,
            updated_session_state=function_execution_result.updated_session_state,
            images=function_execution_result.images,
            videos=function_execution_result.videos,
            audios=function_execution_result.audios,
            files=function_execution_result.files,
        )

        function_call_results.append(function_call_result)

    def run_function_calls(
        self,
        function_calls: List[FunctionCall],
        function_call_results: List[Message],
        additional_input: Optional[List[Message]] = None,
        current_function_call_count: int = 0,
        function_call_limit: Optional[int] = None,
    ) -> Iterator[Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]]:
        if additional_input is None:
            additional_input = []

        for fc in function_calls:
            if function_call_limit is not None:
                current_function_call_count += 1
                if current_function_call_count > function_call_limit:
                    function_call_results.append(self.create_tool_call_limit_error_result(fc))
                    continue

            paused_tool_executions = []

            if fc.function.requires_confirmation:
                paused_tool_executions.append(
                    ToolExecution(
                        tool_call_id=fc.call_id,
                        tool_name=fc.function.name,
                        tool_args=fc.arguments,
                        requires_confirmation=True,
                    )
                )
            if fc.function.requires_user_input:
                user_input_schema = fc.function.user_input_schema
                if fc.arguments and user_input_schema:
                    for name, value in fc.arguments.items():
                        for user_input_field in user_input_schema:
                            if user_input_field.name == name:
                                user_input_field.value = value

                paused_tool_executions.append(
                    ToolExecution(
                        tool_call_id=fc.call_id,
                        tool_name=fc.function.name,
                        tool_args=fc.arguments,
                        requires_user_input=True,
                        user_input_schema=user_input_schema,
                    )
                )
            if fc.function.name == "get_user_input" and fc.arguments and fc.arguments.get("user_input_fields"):
                user_input_schema = []
                for input_field in fc.arguments.get("user_input_fields", []):
                    field_type = input_field.get("field_type")
                    try:
                        python_type = eval(field_type) if isinstance(field_type, str) else field_type
                    except (NameError, SyntaxError):
                        python_type = str  # Default to str if type is invalid
                    user_input_schema.append(
                        UserInputField(
                            name=input_field.get("field_name"),
                            field_type=python_type,
                            description=input_field.get("field_description"),
                        )
                    )

                paused_tool_executions.append(
                    ToolExecution(
                        tool_call_id=fc.call_id,
                        tool_name=fc.function.name,
                        tool_args=fc.arguments,
                        requires_user_input=True,
                        user_input_schema=user_input_schema,
                    )
                )
            if fc.function.external_execution:
                paused_tool_executions.append(
                    ToolExecution(
                        tool_call_id=fc.call_id,
                        tool_name=fc.function.name,
                        tool_args=fc.arguments,
                        external_execution_required=True,
                    )
                )

            if paused_tool_executions:
                yield ModelResponse(
                    tool_executions=paused_tool_executions,
                    event=ModelResponseEvent.tool_call_paused.value,
                )
                continue

            yield from self.run_function_call(
                function_call=fc, function_call_results=function_call_results, additional_input=additional_input
            )

        if additional_input:
            function_call_results.extend(additional_input)

    async def arun_function_call(
        self,
        function_call: FunctionCall,
    ) -> Tuple[Union[bool, AgentRunException], Timer, FunctionCall, FunctionExecutionResult]:
        from inspect import isasyncgenfunction, iscoroutine, iscoroutinefunction

        function_call_timer = Timer()
        function_call_timer.start()
        success: Union[bool, AgentRunException] = False

        try:
            if (
                iscoroutinefunction(function_call.function.entrypoint)
                or isasyncgenfunction(function_call.function.entrypoint)
                or iscoroutine(function_call.function.entrypoint)
            ):
                result = await function_call.aexecute()
                success = result.status == "success"

            elif function_call.function.tool_hooks is not None and any(
                iscoroutinefunction(f) for f in function_call.function.tool_hooks
            ):
                result = await function_call.aexecute()
                success = result.status == "success"
            else:
                result = await asyncio.to_thread(function_call.execute)
                success = result.status == "success"
        except AgentRunException as e:
            success = e
        except Exception as e:
            log_error(f"Error executing function {function_call.function.name}: {e}")
            success = False
            raise e

        function_call_timer.stop()
        return success, function_call_timer, function_call, result

    async def arun_function_calls(
        self,
        function_calls: List[FunctionCall],
        function_call_results: List[Message],
        additional_input: Optional[List[Message]] = None,
        current_function_call_count: int = 0,
        function_call_limit: Optional[int] = None,
        skip_pause_check: bool = False,
    ) -> AsyncIterator[Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]]:
        if additional_input is None:
            additional_input = []

        function_calls_to_run = []
        for fc in function_calls:
            if function_call_limit is not None:
                current_function_call_count += 1
                if current_function_call_count > function_call_limit:
                    function_call_results.append(self.create_tool_call_limit_error_result(fc))
                    continue
            function_calls_to_run.append(fc)

        for fc in function_calls_to_run:
            paused_tool_executions = []
            if fc.function.requires_confirmation and not skip_pause_check:
                paused_tool_executions.append(
                    ToolExecution(
                        tool_call_id=fc.call_id,
                        tool_name=fc.function.name,
                        tool_args=fc.arguments,
                        requires_confirmation=True,
                    )
                )
            if fc.function.requires_user_input and not skip_pause_check:
                user_input_schema = fc.function.user_input_schema
                if fc.arguments and user_input_schema:
                    for name, value in fc.arguments.items():
                        for user_input_field in user_input_schema:
                            if user_input_field.name == name:
                                user_input_field.value = value

                paused_tool_executions.append(
                    ToolExecution(
                        tool_call_id=fc.call_id,
                        tool_name=fc.function.name,
                        tool_args=fc.arguments,
                        requires_user_input=True,
                        user_input_schema=user_input_schema,
                    )
                )
            if (
                fc.function.name == "get_user_input"
                and fc.arguments
                and fc.arguments.get("user_input_fields")
                and not skip_pause_check
            ):
                fc.function.requires_user_input = True
                user_input_schema = []
                for input_field in fc.arguments.get("user_input_fields", []):
                    field_type = input_field.get("field_type")
                    try:
                        python_type = eval(field_type) if isinstance(field_type, str) else field_type
                    except (NameError, SyntaxError):
                        python_type = str  # Default to str if type is invalid
                    user_input_schema.append(
                        UserInputField(
                            name=input_field.get("field_name"),
                            field_type=python_type,
                            description=input_field.get("field_description"),
                        )
                    )

                paused_tool_executions.append(
                    ToolExecution(
                        tool_call_id=fc.call_id,
                        tool_name=fc.function.name,
                        tool_args=fc.arguments,
                        requires_user_input=True,
                        user_input_schema=user_input_schema,
                    )
                )
            if fc.function.external_execution and not skip_pause_check:
                paused_tool_executions.append(
                    ToolExecution(
                        tool_call_id=fc.call_id,
                        tool_name=fc.function.name,
                        tool_args=fc.arguments,
                        external_execution_required=True,
                    )
                )

            if paused_tool_executions:
                yield ModelResponse(
                    tool_executions=paused_tool_executions,
                    event=ModelResponseEvent.tool_call_paused.value,
                )
                continue

            yield ModelResponse(
                content=fc.get_call_str(),
                tool_executions=[
                    ToolExecution(
                        tool_call_id=fc.call_id,
                        tool_name=fc.function.name,
                        tool_args=fc.arguments,
                    )
                ],
                event=ModelResponseEvent.tool_call_started.value,
            )

        if skip_pause_check:
            function_calls_to_run = function_calls_to_run
        else:
            function_calls_to_run = [
                fc
                for fc in function_calls_to_run
                if not (
                    fc.function.requires_confirmation
                    or fc.function.external_execution
                    or fc.function.requires_user_input
                )
            ]

        results = await asyncio.gather(
            *(self.arun_function_call(fc) for fc in function_calls_to_run), return_exceptions=True
        )

        for result in results:
            if isinstance(result, BaseException):
                log_error(f"Error during function call: {result}")
                raise result

            function_call_success, function_call_timer, function_call, function_execution_result = result

            updated_session_state = function_execution_result.updated_session_state

            if isinstance(function_call_success, AgentRunException):
                a_exc = function_call_success
                _handle_agent_exception(a_exc, additional_input)
                function_call_success = False

            function_call_output: str = ""
            if isinstance(function_call.result, (GeneratorType, collections.abc.Iterator)):
                for item in function_call.result:
                    if isinstance(item, tuple(get_args(RunOutputEvent))) or isinstance(
                        item, tuple(get_args(TeamRunOutputEvent))
                    ):
                        if isinstance(item, RunContentEvent) or isinstance(item, TeamRunContentEvent):
                            if item.content is not None and isinstance(item.content, BaseModel):
                                function_call_output += item.content.model_dump_json()
                            else:
                                function_call_output += item.content or ""

                            if function_call.function.show_result:
                                yield ModelResponse(content=item.content)
                                continue

                        yield item
                    else:
                        function_call_output += str(item)
                        if function_call.function.show_result:
                            yield ModelResponse(content=str(item))
            elif isinstance(function_call.result, (AsyncGeneratorType, collections.abc.AsyncIterator)):
                async for item in function_call.result:
                    if isinstance(item, tuple(get_args(RunOutputEvent))) or isinstance(
                        item, tuple(get_args(TeamRunOutputEvent))
                    ):
                        if isinstance(item, RunContentEvent) or isinstance(item, TeamRunContentEvent):
                            if item.content is not None and isinstance(item.content, BaseModel):
                                function_call_output += item.content.model_dump_json()
                            else:
                                function_call_output += item.content or ""

                            if function_call.function.show_result:
                                yield ModelResponse(content=item.content)
                                continue

                            if isinstance(item, CustomEvent):
                                function_call_output += str(item)

                        yield item

                    else:
                        function_call_output += str(item)
                        if function_call.function.show_result:
                            yield ModelResponse(content=str(item))
            else:
                from agno.tools.function import ToolResult

                if isinstance(function_execution_result.result, ToolResult):
                    tool_result = function_execution_result.result
                    function_call_output = tool_result.content

                    if tool_result.images:
                        function_execution_result.images = tool_result.images
                    if tool_result.videos:
                        function_execution_result.videos = tool_result.videos
                    if tool_result.audios:
                        function_execution_result.audios = tool_result.audios
                    if tool_result.files:
                        function_execution_result.files = tool_result.files
                else:
                    function_call_output = str(function_call.result)

                if function_call.function.show_result:
                    yield ModelResponse(content=function_call_output)

            function_call_result = self.create_function_call_result(
                function_call,
                success=function_call_success,
                output=function_call_output,
                timer=function_call_timer,
                function_execution_result=function_execution_result,
            )
            yield ModelResponse(
                content=f"{function_call.get_call_str()} completed in {function_call_timer.elapsed:.4f}s. ",
                tool_executions=[
                    ToolExecution(
                        tool_call_id=function_call_result.tool_call_id,
                        tool_name=function_call_result.tool_name,
                        tool_args=function_call_result.tool_args,
                        tool_call_error=function_call_result.tool_call_error,
                        result=str(function_call_result.content),
                        stop_after_tool_call=function_call_result.stop_after_tool_call,
                        metrics=function_call_result.metrics,
                    )
                ],
                event=ModelResponseEvent.tool_call_completed.value,
                updated_session_state=updated_session_state,
                images=function_execution_result.images,
                videos=function_execution_result.videos,
                audios=function_execution_result.audios,
                files=function_execution_result.files,
            )

            function_call_results.append(function_call_result)

        if additional_input:
            function_call_results.extend(additional_input)

    def _prepare_function_calls(
        self,
        assistant_message: Message,
        messages: List[Message],
        model_response: ModelResponse,
        functions: Optional[Dict[str, Function]] = None,
    ) -> List[FunctionCall]:
        if model_response.content is None:
            model_response.content = ""
        if model_response.tool_calls is None:
            model_response.tool_calls = []

        function_calls_to_run: List[FunctionCall] = self.get_function_calls_to_run(
            assistant_message, messages, functions
        )
        return function_calls_to_run

    def format_function_call_results(
        self, messages: List[Message], function_call_results: List[Message], **kwargs
    ) -> None:
        if len(function_call_results) > 0:
            messages.extend(function_call_results)

    def _handle_function_call_media(
        self, messages: List[Message], function_call_results: List[Message], send_media_to_model: bool = True
    ) -> None:
        if not function_call_results:
            return

        all_images: List[Image] = []
        all_videos: List[Video] = []
        all_audio: List[Audio] = []
        all_files: List[File] = []

        for result_message in function_call_results:
            if result_message.images:
                all_images.extend(result_message.images)
                result_message.images = None

            if result_message.videos:
                all_videos.extend(result_message.videos)
                result_message.videos = None

            if result_message.audio:
                all_audio.extend(result_message.audio)
                result_message.audio = None

            if result_message.files:
                all_files.extend(result_message.files)
                result_message.files = None

        if send_media_to_model and (all_images or all_videos or all_audio or all_files):
            media_message = Message(
                role="user",
                content="Take note of the following content",
                images=all_images if all_images else None,
                videos=all_videos if all_videos else None,
                audio=all_audio if all_audio else None,
                files=all_files if all_files else None,
            )
            messages.append(media_message)

    def get_system_message_for_model(self, tools: Optional[List[Any]] = None) -> Optional[str]:
        return self.system_prompt

    def get_instructions_for_model(self, tools: Optional[List[Any]] = None) -> Optional[List[str]]:
        return self.instructions

    def __deepcopy__(self, memo):
        from copy import copy, deepcopy

        cls = self.__class__
        new_model = cls.__new__(cls)
        memo[id(self)] = new_model

        for k, v in self.__dict__.items():
            if k in {"response_format", "_tools", "_functions"}:
                continue
            try:
                setattr(new_model, k, deepcopy(v, memo))
            except Exception:
                try:
                    setattr(new_model, k, copy(v))
                except Exception:
                    setattr(new_model, k, v)

        return new_model
