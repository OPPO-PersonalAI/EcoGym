#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by OPPO
# Licensed under the Apache License, Version 2.0 (the "License");

import random
import json
import inspect
import time
from typing import List, Callable, Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

    class Fore:
        CYAN = ""
        BRIGHT = ""
    class Style:
        BRIGHT = ""
        RESET_ALL = ""

try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessage, ChatCompletion
except ImportError:
    OpenAI = None
    ChatCompletionMessage = None
    ChatCompletion = None


class SimpleRunStatus(str, Enum):
    pending = "PENDING"
    running = "RUNNING"
    completed = "COMPLETED"
    paused = "PAUSED"
    cancelled = "CANCELLED"
    error = "ERROR"


@dataclass
class SimpleMetrics:
    duration: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class SimpleMessage:
    role: str
    content: Optional[Any] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[SimpleMetrics] = None
    reasoning_content: Optional[str] = None


@dataclass
class SimpleToolExecution:
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_call_error: Optional[bool] = None
    result: Optional[Any] = None


@dataclass
class SimpleRunOutput:
    content: Optional[Any] = None
    messages: Optional[List[SimpleMessage]] = None
    metrics: Optional[SimpleMetrics] = None
    tools: Optional[List[SimpleToolExecution]] = None
    status: SimpleRunStatus = SimpleRunStatus.completed
    reasoning_content: Optional[str] = None
    
    def get_content_as_string(self, indent: int = 2) -> str:
        if isinstance(self.content, str):
            return self.content
        else:
            return json.dumps(self.content, indent=indent, ensure_ascii=False)


def function_to_schema(func: Callable) -> dict:
    """
    Convert a function to OpenAI function calling schema.
    
    Handles wrapped functions:
    - If function has __wrapped__ attribute, use original function signature
    - Exclude framework-injected parameters (session_state, agent, team, etc.)
    - Extract parameter descriptions from docstring (supports Google, NumPy, Sphinx styles)
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null"
    }

    original_func = func
    if hasattr(func, '__wrapped__'):
        original_func = func.__wrapped__
    
    sig = inspect.signature(original_func)

    excluded_params = {"session_state", "agent", "team", "self", "images", "videos", "audios", "files", "dependencies"}
    
    param_descriptions = {}
    try:
        from docstring_parser import parse
        from inspect import getdoc
        
        if docstring := getdoc(original_func):
            parsed_doc = parse(docstring)
            if parsed_doc.params:
                for param in parsed_doc.params:
                    param_name = param.arg_name
                    param_type_name = param.type_name
                    description = param.description or f"Parameter {param_name}"
                    
                    if param_type_name:
                        param_descriptions[param_name] = f"({param_type_name}) {description}"
                    else:
                        param_descriptions[param_name] = description
    except Exception:
        pass
    
    parameters = {}
    required = []
    for name, param in sig.parameters.items():
        if name in excluded_params:
            continue
        
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        
        param_type = type_map.get(param.annotation, "string")
        
        description = param_descriptions.get(name, f"Parameter {name}")
        parameters[name] = {"type": param_type, "description": description}
        
        if param.default == inspect.Parameter.empty:
            required.append(name)
    
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (original_func.__doc__ or func.__doc__ or "").strip(),
            "parameters": {"type": "object", "properties": parameters, "required": required}
        }
    }

class Agent:
    def __init__(self, 
        model,
        tools: List[Callable], 
        instructions: str,
        model_id: Optional[str] = None,
        initial_session_state: Optional[Dict[str, Any]] = None,
        history_limit: Optional[int] = None,
        use_responses_api: Optional[bool] = None):

        self.model = model
        self.model_id = model_id
        self.instructions = instructions
        self.messages = [{"role": "system", "content": self.instructions}]
        self.response_inputs: List[Any] = []
        
        self.history_limit = history_limit
        
        self.session_state = dict(initial_session_state) if initial_session_state else {}
        
        self.tool_map = {}
        self.tool_schemas = []
        
        for tool in tools:
            if hasattr(tool, 'tools'):
                toolkit = tool
                for toolkit_tool in toolkit.tools:
                    if callable(toolkit_tool):
                        wrapped_func = self._wrap_tool_with_session_state(toolkit_tool)
                        self.tool_map[toolkit_tool.__name__] = wrapped_func
                        self.tool_schemas.append(function_to_schema(wrapped_func))
            elif callable(tool):
                wrapped_func = self._wrap_tool_with_session_state(tool)
                self.tool_map[tool.__name__] = wrapped_func
                self.tool_schemas.append(function_to_schema(wrapped_func))
        
        self.is_openai_client = OpenAI is not None and isinstance(model, OpenAI)
        self.max_retries = 5
        self.retry_delay = 2

        self.use_responses_api = (
            bool(use_responses_api)
            if use_responses_api is not None
            else (self.is_openai_client and isinstance(self.model_id, str) and self.model_id.startswith("closed_5.2_calling_pipeline"))
        )
        self.response_tool_schemas = self._convert_tools_for_responses(self.tool_schemas)

    def update_system_prompt(self, new_system_prompt: str):
        self.instructions = new_system_prompt
        
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = new_system_prompt
        else:
            self.messages.insert(0, {"role": "system", "content": new_system_prompt})
    
    def _wrap_tool_with_session_state(self, tool: Callable) -> Callable:
        from inspect import signature
        from functools import wraps
        
        try:
            sig = signature(tool)
            has_session_state = 'session_state' in sig.parameters
            session_state_param = sig.parameters.get('session_state') if has_session_state else None
        except Exception:
            has_session_state = False
            session_state_param = None
        
        if has_session_state:
            @wraps(tool)
            def wrapper(*args, **kwargs):
                kwargs.pop('session_state', None)
                
                if not isinstance(self.session_state, dict):
                    raise TypeError(
                        f"session_state must be a dict, got {type(self.session_state).__name__}: {self.session_state}"
                    )
                
                if hasattr(tool, '__self__'):
                    return tool(self.session_state, *args, **kwargs)
                else:
                    return tool(self.session_state, *args, **kwargs)
            return wrapper
        else:
            return tool
    
    def _convert_tools_for_responses(self, tool_schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        responses_tools: List[Dict[str, Any]] = []
        for schema in tool_schemas:
            if isinstance(schema, dict) and "function" in schema:
                func_obj = schema.get("function", {})
                responses_tools.append({
                    "type": "function",
                    "name": func_obj.get("name"),
                    "description": func_obj.get("description", ""),
                    "parameters": func_obj.get("parameters", {"type": "object", "properties": {}}),
                })
            elif schema:
                responses_tools.append(schema)
        return responses_tools

    def _parse_tool_arguments(self, args_raw: Any) -> Dict[str, Any]:
        if isinstance(args_raw, str):
            s = args_raw
            args: Dict[str, Any] = {}
            for _ in range(2):
                try:
                    parsed = json.loads(s)
                except Exception:
                    break
                if isinstance(parsed, dict):
                    args = parsed
                    break
                if isinstance(parsed, list):
                    args = {}
                    break
                if isinstance(parsed, str):
                    s = parsed
                    continue
                args = {}
                break
            if not isinstance(args, dict):
                args = {}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}
        return args

    def _serialize_tool_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)

    def _get_response_item_type(self, item: Any) -> Optional[str]:
        if isinstance(item, dict):
            return item.get("type") or item.get("role")
        return getattr(item, "type", None) or getattr(item, "role", None)
    
    def _print_session_state(self):
        try:
            state_json = json.dumps(self.session_state, ensure_ascii=False, indent=2)
            
            section = "\n" + "=" * 80 + "\n"
            section += "📊 [Session State After Tools]\n"
            section += "-" * 80 + "\n"
            section += state_json + "\n"
            section += "-" * 80 + "\n"
            
            if COLORAMA_AVAILABLE:
                print(f"{Fore.CYAN}{Style.BRIGHT}{section}{Style.RESET_ALL}")
            else:
                print(section)
        except Exception as e:
            print(f"\n📊 [Session State After Tools] (Error formatting: {e})\n{str(self.session_state)}\n")

    def run(self, user_query: str = "", session_id: Optional[str] = None) -> SimpleRunOutput:
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages = [{"role": "system", "content": self.instructions}]
        
        if self.use_responses_api:
            return self._run_with_responses_api(user_query=user_query, session_id=session_id)
        
        if self.history_limit is not None and self.history_limit > 0:
            current_len = len(self.messages)
            if current_len > self.history_limit + 1:
                kept_slice = self.messages[-self.history_limit:]
                
                while kept_slice and kept_slice[0].get("role") == "tool":
                    kept_slice.pop(0)
                
                self.messages = [self.messages[0]] + kept_slice
        
        last_message_role = self.messages[-1].get("role") if len(self.messages) > 1 else None
        
        if user_query or last_message_role != "tool":
            if not user_query and len(self.messages) == 1:
                user_query = "Please proceed with your task according to the instructions."
            
            if user_query:
                self.messages.append({"role": "user", "content": user_query})
                print(f"🤖 User: {user_query}")

        all_messages = [
            SimpleMessage(role="system", content=self.instructions),
            SimpleMessage(role="user", content=user_query)
        ]
        all_tools = []
        total_metrics = SimpleMetrics()
        final_content = None
        
        start_time = time.time()
        call_start_time = time.time()
        response = None
        message = None
        
        for attempt in range(self.max_retries):
            try:
                if self.is_openai_client:
                    response = self.model.chat.completions.create(
                        model=self.model_id,
                        messages=self.messages,
                        tools=self.tool_schemas if self.tool_schemas else None
                    )
                    message = response.choices[0].message
                elif hasattr(self.model, 'chat'):
                    message = self.model.chat(
                        messages=self.messages, 
                        tools=self.tool_schemas
                    )
                else:
                    raise AttributeError(f"Model object {type(self.model)} has no 'chat' method and is not an OpenAI client")
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"{Fore.RED}🔥 LLM call completely failed (Error: {e}). Attempting to execute emergency circuit breaker procedure...{Style.RESET_ALL}")
                    
                    target_task_id = None
                    for msg in reversed(self.messages):
                        if msg.get("role") == "tool":
                            content = str(msg.get("content", ""))
                            if '"task_id":' in content and '"status": "selected"' in content:
                                try:
                                    data = json.loads(content)
                                    target_task_id = data.get("task_id")
                                    if target_task_id:
                                        break
                                except:
                                    pass
                    
                    if target_task_id and "solution_submit" in self.tool_map:
                        print(f"{Fore.YELLOW}🛡️ Detected active task ID: {target_task_id}. Forcing submission to skip this task...{Style.RESET_ALL}")
                        
                        try:
                            fake_reasoning = f"System Error encountered. Forcing submission to skip Task {target_task_id}."
                            
                            force_args = {
                                "task_id": str(target_task_id),
                                "solution_text": "SYSTEM_FORCE_QUIT: API Error / Loop Detected. Skipping."
                            }
                            
                            tool_result = self.tool_map["solution_submit"](**force_args)
                            
                            mock_tool_call_id = f"call_force_quit_{int(time.time())}"
                            mock_tool_call = {
                                "id": mock_tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": "solution_submit",
                                    "arguments": json.dumps(force_args)
                                }
                            }
                            
                            self.messages.append({
                                "role": "assistant",
                                "content": fake_reasoning,
                                "tool_calls": [mock_tool_call]
                            })
                            
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": mock_tool_call_id,
                                "content": str(tool_result)
                            })
                            
                            all_messages.append(SimpleMessage(
                                role="assistant", 
                                content=fake_reasoning,
                                tool_calls=[mock_tool_call]
                            ))
                            all_messages.append(SimpleMessage(
                                role="tool",
                                content=str(tool_result)
                            ))
                            
                            force_tool_exec = SimpleToolExecution(
                                tool_call_id=mock_tool_call_id,
                                tool_name="solution_submit",
                                tool_args=force_args,
                                tool_call_error=False,
                                result=tool_result
                            )
                            
                            print(f"{Fore.GREEN}✅ Emergency fallback succeeded! Task {target_task_id} has been forcibly removed.{Style.RESET_ALL}")
                            
                            return SimpleRunOutput(
                                content=fake_reasoning,
                                messages=all_messages,
                                metrics=total_metrics,
                                tools=[force_tool_exec],
                                status=SimpleRunStatus.completed
                            )
                        except Exception as inner_e:
                            print(f"{Fore.RED}❌ Emergency fallback execution failed: {inner_e}{Style.RESET_ALL}")
                            raise e
                    else:
                        print(f"{Fore.RED}❌ Unable to find active task ID or missing solution_submit tool, fallback is not possible.{Style.RESET_ALL}")
                        raise e
                
                wait_time = self.retry_delay
                print(f"{Fore.RED}⚠️ LLM call exception: {e}, retrying attempt {attempt + 1} (waiting {wait_time:.1f}s)...{Style.RESET_ALL}")
                time.sleep(wait_time)

        call_duration = time.time() - call_start_time
        
        if total_metrics.duration is None:
            total_metrics.duration = 0
        total_metrics.duration += call_duration
        
        if self.is_openai_client and response and hasattr(response, 'usage'):
            usage = response.usage
            if usage:
                total_metrics.input_tokens += getattr(usage, 'prompt_tokens', 0)
                total_metrics.output_tokens += getattr(usage, 'completion_tokens', 0)
                total_metrics.total_tokens += getattr(usage, 'total_tokens', 0)
                if total_metrics.total_tokens == 0:
                    total_metrics.total_tokens = total_metrics.input_tokens + total_metrics.output_tokens
        elif hasattr(message, 'usage'):
            usage = message.usage
            if usage:
                if hasattr(usage, 'prompt_tokens'):
                    total_metrics.input_tokens += usage.prompt_tokens
                if hasattr(usage, 'completion_tokens'):
                    total_metrics.output_tokens += usage.completion_tokens
                if hasattr(usage, 'total_tokens'):
                    total_metrics.total_tokens += usage.total_tokens
                else:
                    total_metrics.total_tokens = total_metrics.input_tokens + total_metrics.output_tokens
        
        msg_metrics = SimpleMetrics(duration=call_duration)
        
        tool_calls_list = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls_list = []
            for tc in message.tool_calls:
                if isinstance(tc, dict):
                    tool_calls_list.append(tc)
                else:
                    tc_dict = {
                        "id": getattr(tc, 'id', None),
                        "type": getattr(tc, 'type', 'function'),
                        "function": {
                            "name": getattr(tc.function, 'name', None) if hasattr(tc, 'function') else None,
                            "arguments": getattr(tc.function, 'arguments', None) if hasattr(tc, 'function') else None
                        }
                    }
                    tool_calls_list.append(tc_dict)
        
        if not tool_calls_list and hasattr(message, 'content') and message.content:
            try:
                content_text = message.content.strip()
                if "```json" in content_text:
                    content_text = content_text.split("```json")[1].split("```")[0].strip()
                elif "```" in content_text:
                    content_text = content_text.split("```")[1].split("```")[0].strip()
                
                if content_text.startswith("{") and content_text.endswith("}"):
                    parsed_data = json.loads(content_text)
                    
                    if isinstance(parsed_data, dict) and "tool_calls" in parsed_data:
                        print(f"{Fore.YELLOW}⚠️ Detected that the model output tool call JSON in Content, performing manual parsing...{Style.RESET_ALL}")
                        raw_calls = parsed_data["tool_calls"]
                        if isinstance(raw_calls, list):
                            tool_calls_list = []
                            for tc in raw_calls:
                                if "function" in tc:
                                    func_obj = tc["function"]
                                    if isinstance(func_obj.get("arguments"), dict):
                                        func_obj["arguments"] = json.dumps(func_obj["arguments"])
                                    
                                    tool_calls_list.append({
                                        "id": tc.get("id", f"call_fallback_{int(time.time())}"),
                                        "type": tc.get("type", "function"),
                                        "function": func_obj
                                    })
                    elif isinstance(parsed_data, dict) and "function" in parsed_data:
                        pass
            except json.JSONDecodeError:
                pass
            except Exception:
                pass
        
        simple_message = SimpleMessage(
            role="assistant",
            content=message.content if hasattr(message, 'content') else "",
            tool_calls=tool_calls_list,
            metrics=msg_metrics,
            reasoning_content=getattr(message, 'reasoning_content', None)
        )
        all_messages.append(simple_message)
        
        content_val = message.content if hasattr(message, 'content') else ""
        if content_val is None:
            content_val = ""

        if tool_calls_list:
            last_assistant_msg = None
            for i in range(len(self.messages) - 1, -1, -1):
                if self.messages[i]["role"] == "assistant":
                    last_assistant_msg = self.messages[i]
                    break
            
            is_repetition = False
            if last_assistant_msg and "tool_calls" in last_assistant_msg:
                last_tools = last_assistant_msg["tool_calls"]
                if len(last_tools) == len(tool_calls_list):
                    current_dump = json.dumps(tool_calls_list, sort_keys=True)
                    last_dump = json.dumps(last_tools, sort_keys=True)
                    if current_dump == last_dump:
                        is_repetition = True

            if is_repetition:
                print(f"{Fore.RED}🛑 Agent detected an infinite loop (repetitive identical tool calls), forcibly intercepting.{Style.RESET_ALL}")
                
                assistant_msg_dict = {"role": "assistant", "content": message.content or "", "tool_calls": tool_calls_list}
                self.messages.append(assistant_msg_dict)
                all_messages.append(SimpleMessage(role="assistant", content=message.content, tool_calls=tool_calls_list, metrics=msg_metrics))
                
                for tc in tool_calls_list:
                    tool_call_id = tc.get("id")
                    error_content = "SYSTEM ERROR: You executed this exact tool with the same arguments in the previous turn. This is invalid. You must proceed to the NEXT step (e.g., solve the problem or submit the solution) instead of repeating the inspection."
                    
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": error_content
                    })
                    all_messages.append(SimpleMessage(role="tool", content=error_content))
                    print(f"   🚫 Blocked Repetition: {error_content}")

                return SimpleRunOutput(
                    content=message.content,
                    messages=all_messages,
                    metrics=total_metrics,
                    tools=[],
                    status=SimpleRunStatus.completed
                )

        assistant_msg_dict = {
            "role": "assistant",
            "content": content_val
        }
        if tool_calls_list:
            assistant_msg_dict["tool_calls"] = tool_calls_list
        self.messages.append(assistant_msg_dict)

        if tool_calls_list:
            print(f"🤔 Agent decides to call {len(tool_calls_list)} tool(s)")
            
            for tool_call in tool_calls_list:
                if isinstance(tool_call, dict):
                    func_info = tool_call.get('function', {})
                    func_name = func_info.get('name', 'unknown')
                    args_raw = func_info.get('arguments', '{}')
                    tool_call_id = tool_call.get('id', None)
                else:
                    func_name = getattr(tool_call.function, 'name', 'unknown') if hasattr(tool_call, 'function') else 'unknown'
                    args_raw = getattr(tool_call.function, 'arguments', '{}') if hasattr(tool_call, 'function') else '{}'
                    tool_call_id = getattr(tool_call, 'id', None)
                
                if isinstance(args_raw, str):
                    s = args_raw
                    args = {}
                    for _ in range(2):
                        try:
                            parsed = json.loads(s)
                        except Exception:
                            break
                        if isinstance(parsed, dict):
                            args = parsed
                            break
                        if isinstance(parsed, list):
                            args = {}
                            break
                        if isinstance(parsed, str):
                            s = parsed
                            continue
                        args = {}
                        break
                    if not isinstance(args, dict):
                        args = {}
                else:
                    if isinstance(args_raw, dict):
                        args = args_raw
                    else:
                        args = {}
                
                skip_execution = False
                mock_result = None

                if func_name.startswith("functions."):
                    func_name = func_name.replace("functions.", "", 1)

                args = self._parse_tool_arguments(args_raw)

                if func_name == "task_inspect":
                    target_task_id = args.get("task_id")
                    if target_task_id:
                        for msg in self.messages:
                            if msg.get("role") == "tool":
                                content = msg.get("content", "")
                                if f'"{target_task_id}"' in content and "selected" in content:
                                    print(f"{Fore.RED}🚫 Loop intercepted: Task {target_task_id} is already in the context, repeated inspection is forbidden!{Style.RESET_ALL}")
                                    skip_execution = True
                                    mock_result = (
                                        f"SYSTEM WARNING: You have ALREADY inspected Task {target_task_id} above. "
                                        f"The details are in your conversation history. "
                                        f"DO NOT inspect it again. "
                                        f"You MUST strictly proceed to solve it using 'solution_submit' now, or give up."
                                    )
                                    break
                
                output_str = f"   ⚙️  Executing: {func_name}({args})"[:100]
                print(output_str)
                
                tool_exec = SimpleToolExecution(
                    tool_call_id=tool_call_id,
                    tool_name=func_name,
                    tool_args=args
                )
                
                if skip_execution:
                    result = mock_result
                    tool_exec.result = result
                    tool_exec.tool_call_error = True
                elif func_name in self.tool_map:
                    try:
                        result = self.tool_map[func_name](**args)
                        tool_exec.result = result
                        tool_exec.tool_call_error = False
                    except Exception as e:
                        result = f"Error: {e}"
                        tool_exec.result = result
                        tool_exec.tool_call_error = True
                else:
                    result = "Error: Tool not found"
                    tool_exec.result = result
                    tool_exec.tool_call_error = True
                
                all_tools.append(tool_exec)
                
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(result)
                }
                self.messages.append(tool_message)
                
                all_messages.append(SimpleMessage(
                    role="tool",
                    content=str(result)
                ))
                
                log_result = str(result)
                if len(log_result) > 200:
                    log_result = log_result[:200] + "..."
                print(f"   👀 Observation: {log_result}")
            
        final_content = message.content if hasattr(message, 'content') else None
        if final_content is not None:
            print(f"🤖 Agent: {final_content[:100]}")
        else:
            print(f"🤖 Agent: {final_content}")
        
        return SimpleRunOutput(
            content=final_content,
            messages=all_messages,
            metrics=total_metrics,
            tools=all_tools if all_tools else None,
            status=SimpleRunStatus.completed,
            reasoning_content=getattr(message, 'reasoning_content', None)
        )
    
    def _run_with_responses_api(self, user_query: str = "", session_id: Optional[str] = None) -> SimpleRunOutput:
        if self.history_limit is not None and self.history_limit > 0 and len(self.response_inputs) > self.history_limit:
            kept_slice = self.response_inputs[-self.history_limit:]
            while kept_slice and self._get_response_item_type(kept_slice[0]) == "function_call_output":
                kept_slice.pop(0)
            self.response_inputs = kept_slice

        add_user_msg = False
        if user_query:
            add_user_msg = True
        elif not self.response_inputs:
            user_query = "Please proceed with your task strictly according to the instructions without any useless descriptive sentences."
            add_user_msg = True
        else:
            add_user_msg = True

        all_messages = [SimpleMessage(role="system", content=self.instructions)]
        if add_user_msg:
            user_msg = {"role": "user", "content": user_query}
            self.response_inputs.append(user_msg)
            self.messages.append(user_msg)
            all_messages.append(SimpleMessage(role="user", content=user_query))
            if user_query:
                print(f"🤖 User: {user_query}")

        all_tools: List[SimpleToolExecution] = []
        total_metrics = SimpleMetrics()
        call_start_time = time.time()
        response = None

        for attempt in range(self.max_retries):
            try:
                response = self.model.responses.create(
                    model=self.model_id,
                    input=self.response_inputs,
                    tools=self.response_tool_schemas if self.response_tool_schemas else None,
                    instructions=self.instructions
                )
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"{Fore.RED}🔥 LLM call completely failed (Error: {e}). Attempting to execute emergency circuit breaker procedure...{Style.RESET_ALL}")

                    target_task_id = None
                    for msg in reversed(self.messages):
                        if msg.get("role") == "tool":
                            content = str(msg.get("content", ""))
                            if '"task_id":' in content and '"status": "selected"' in content:
                                try:
                                    data = json.loads(content)
                                    target_task_id = data.get("task_id")
                                    if target_task_id:
                                        break
                                except:
                                    pass

                    if target_task_id and "solution_submit" in self.tool_map:
                        print(f"{Fore.GREEN}✅ Emergency circuit breaker succeeded! Task {target_task_id} has been forcibly removed.{Style.RESET_ALL}")

                        try:
                            fake_reasoning = f"System Error encountered. Forcing submission to skip Task {target_task_id}."

                            force_args = {
                                "task_id": str(target_task_id),
                                "solution_text": "SYSTEM_FORCE_QUIT: API Error / Loop Detected. Skipping."
                            }

                            tool_result = self.tool_map["solution_submit"](**force_args)

                            mock_tool_call_id = f"call_force_quit_{int(time.time())}"

                            mock_tool_call = {
                                "id": mock_tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": "solution_submit",
                                    "arguments": json.dumps(force_args)
                                }
                            }

                            self.messages.append({
                                "role": "assistant",
                                "content": fake_reasoning,
                                "tool_calls": [mock_tool_call]
                            })

                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": mock_tool_call_id,
                                "content": str(tool_result)
                            })

                            serialized_result = self._serialize_tool_result(tool_result)
                            self.response_inputs.append({
                                "type": "function_call",
                                "call_id": mock_tool_call_id,
                                "name": "solution_submit",
                                "arguments": json.dumps(force_args)
                            })
                            self.response_inputs.append({
                                "type": "function_call_output",
                                "call_id": mock_tool_call_id,
                                "output": serialized_result
                            })

                            all_messages.append(SimpleMessage(
                                role="assistant",
                                content=fake_reasoning,
                                tool_calls=[mock_tool_call]
                            ))
                            all_messages.append(SimpleMessage(
                                role="tool",
                                content=serialized_result
                            ))

                            force_tool_exec = SimpleToolExecution(
                                tool_call_id=mock_tool_call_id,
                                tool_name="solution_submit",
                                tool_args=force_args,
                                tool_call_error=False,
                                result=tool_result
                            )

                            print(f"{Fore.GREEN}✅ Emergency circuit breaker succeeded! Task {target_task_id} has been forcibly removed.{Style.RESET_ALL}")

                            return SimpleRunOutput(
                                content=fake_reasoning,
                                messages=all_messages,
                                metrics=total_metrics,
                                tools=[force_tool_exec],
                                status=SimpleRunStatus.completed
                            )

                        except Exception as inner_e:
                            print(f"{Fore.RED}❌ Emergency circuit breaker execution failed: {inner_e}{Style.RESET_ALL}")
                            raise e
                    else:
                        print(f"{Fore.RED}❌ Unable to find active task ID or missing solution_submit tool, circuit breaker cannot proceed.{Style.RESET_ALL}")
                        raise e

                wait_time = self.retry_delay
                print(f"{Fore.RED}⚠️ LLM call exception: {e}. Retrying attempt {attempt + 1} (waiting {wait_time:.1f}s)...{Style.RESET_ALL}")
                time.sleep(self.retry_delay)

        call_duration = time.time() - call_start_time
        if total_metrics.duration is None:
            total_metrics.duration = 0
        total_metrics.duration += call_duration

        if response and hasattr(response, "usage"):
            usage = response.usage
            if usage:
                total_metrics.input_tokens += getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0
                total_metrics.output_tokens += getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0
                if getattr(usage, "total_tokens", None) is not None:
                    total_metrics.total_tokens += getattr(usage, "total_tokens", 0) or 0
                else:
                    total_metrics.total_tokens = total_metrics.input_tokens + total_metrics.output_tokens
        if total_metrics.total_tokens == 0:
            total_metrics.total_tokens = total_metrics.input_tokens + total_metrics.output_tokens

        output_items = list(getattr(response, "output", []) or [])

        tool_calls_list = []
        final_content = getattr(response, "output_text", None) if response is not None else None
        msg_metrics = SimpleMetrics(duration=call_duration)

        for item in output_items:
            item_type = self._get_response_item_type(item)
            if item_type == "function_call":
                func_name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else "unknown")
                args_raw = getattr(item, "arguments", "{}") if not isinstance(item, dict) else item.get("arguments", "{}")
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                if isinstance(item, dict):
                    call_id = item.get("call_id") or item.get("id")
                if not call_id:
                    call_id = f"call_{int(time.time() * 1000)}"

                self.response_inputs.append(item)

                if func_name.startswith("functions."):
                    func_name = func_name.replace("functions.", "", 1)

                tool_calls_list.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": args_raw
                    }
                })

                args = self._parse_tool_arguments(args_raw)
                output_str = f"   ⚙️  Executing: {func_name}({args})"[:100]
                print(output_str)

                tool_exec = SimpleToolExecution(
                    tool_call_id=call_id,
                    tool_name=func_name,
                    tool_args=args
                )

                if func_name in self.tool_map:
                    try:
                        result = self.tool_map[func_name](**args)
                        tool_exec.result = result
                        tool_exec.tool_call_error = False
                    except Exception as e:
                        result = f"Error: {e}"
                        tool_exec.result = result
                        tool_exec.tool_call_error = True
                else:
                    result = "Error: Tool not found"
                    tool_exec.result = result
                    tool_exec.tool_call_error = True

                all_tools.append(tool_exec)

                tool_output_payload = self._serialize_tool_result(result)
                self.response_inputs.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": tool_output_payload
                })
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_output_payload
                })
                all_messages.append(SimpleMessage(
                    role="tool",
                    content=tool_output_payload
                ))
                print(f"   👀 Observation: {tool_output_payload[:100]}")

        assistant_msg_dict = {
            "role": "assistant",
            "content": final_content or ""
        }
        if tool_calls_list:
            assistant_msg_dict["tool_calls"] = tool_calls_list
        self.messages.append(assistant_msg_dict)

        simple_message = SimpleMessage(
            role="assistant",
            content=final_content if final_content is not None else "",
            tool_calls=tool_calls_list or None,
            metrics=msg_metrics
        )
        all_messages.append(simple_message)

        if final_content is not None:
            print(f"🤖 Agent: {final_content[:100]}")
        else:
            print(f"🤖 Agent: {final_content}")

        return SimpleRunOutput(
            content=final_content,
            messages=all_messages,
            metrics=total_metrics,
            tools=all_tools if all_tools else None,
            status=SimpleRunStatus.completed,
            reasoning_content=None
        )