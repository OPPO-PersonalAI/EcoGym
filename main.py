#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import yaml
import argparse
import logging
import uuid
import re
import atexit
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Callable, Dict, Any, Optional
from textwrap import dedent
from functools import partial

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = script_dir
agno_lib_path = os.path.join(repo_root, "libs", "agno")
if agno_lib_path not in sys.path:
    sys.path.insert(0, agno_lib_path)

sys.path.append(os.path.join(script_dir, "config"))

from agno.agent.agent import Agent
from openai import OpenAI
from agno.tools.vending.timer import TimerTools
from agno.run.base import RunStatus

sys.path.insert(0, repo_root)
from memory.manager import MemoryManager

sys.path.append(os.path.join(repo_root, "utils"))
from colored_logging import setup_colored_logging, add_file_handler
from stdout_filter import install_stdout_filter
from session_manager import SessionManager

load_dotenv()

install_stdout_filter()

NO_CALL_WARNING = dedent("""
    ⚠️ IMPORTANT: Your previous response did not include any tool calls. You MUST use the structured tool_calls/function calling mechanism with JSON format (following the tool schema) to call tools. Do NOT write function names in plain text - that will NOT be executed.
    
    CORRECT FORMAT - You must use function calling/tool_calls:
    
    Your response must include tool_calls in the structured format. Example:
    
    {
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "<tool_name>",
                    "arguments": "<json_string_of_arguments>"
                }
            }
            ]
    }
    
    Key points:
    - You MUST call exactly ONE tool per turn (the tool_calls array should contain only one element)
    - Use the tool_calls array in your response
    - Each tool call must have: id, type="function", and function object
    - The function object must have: name (exact tool name from schema) and arguments (JSON string)
    - If a tool has no parameters, use "arguments": "{}"
    - Arguments must be a valid JSON string matching the tool's parameter schema
    - Check the available tools and their schemas provided in the system prompt
""")


def format_log_content(content: Any, max_length: int = 500) -> str:
    """Format content for logging with truncation if too long"""
    if content is None:
        return "None"

    if isinstance(content, str):
        formatted = content
    elif isinstance(content, (dict, list)):
        try:
            formatted = json.dumps(content, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            formatted = str(content)
    elif hasattr(content, 'model_dump_json'):
        try:
            formatted = content.model_dump_json(indent=2, exclude_none=True)
        except:
            if hasattr(content, 'content'):
                formatted = str(content.content) if content.content else str(content)
            else:
                formatted = str(content)
    elif hasattr(content, 'content') and not isinstance(content, type):
        formatted = format_log_content(content.content, max_length=max_length)
    else:
        formatted = str(content)

    if len(formatted) > max_length:
        return formatted[:max_length] + f"\n... (truncated, total length: {len(formatted)} chars)"

    return formatted


def format_state_summary(state: Dict[str, Any], key_fields: Optional[List[str]] = None, max_length: int = 400) -> str:
    """Format state for logging with truncation for large dicts/lists"""
    if key_fields is None:
        key_fields = ["money", "day", "inventory", "price_by_sku", "qty_by_sku"]

    state_summary = {}
    for key in key_fields:
        if key in state:
            value = state[key]
            if isinstance(value, dict):
                if len(value) == 0:
                    state_summary[key] = "{}"
                elif len(value) <= 5:
                    state_summary[key] = value
                else:
                    items = list(value.items())[:3]
                    state_summary[key] = f"{{... {len(value)} items, first 3: {dict(items)}}}"
            elif isinstance(value, list):
                if len(value) == 0:
                    state_summary[key] = "[]"
                elif len(value) <= 5:
                    state_summary[key] = value
                else:
                    state_summary[key] = f"[... {len(value)} items, first 3: {value[:3]}]"
            else:
                state_summary[key] = value

    return format_log_content(state_summary, max_length=max_length)


class BenchmarkLauncher(object):
    """Universal benchmark launcher that supports multiple benchmark types"""

    def __init__(
        self,
        benchmark_type: str,
        env_path: str,
        model_name: str,
        tools: List[Callable],
        state: Dict[str, Any],
        is_finished: Callable,
        cal_metric: Callable,
        system_prompt: str,
        state_context_keys: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        memory_config={},
        resume_session_id: Optional[str] = None,
        session_config: Optional[Dict[str, Any]] = None,
        env_model_name: Optional[str] = None,
        max_actions_per_day: Optional[int] = None,
        token_pricing: Optional[Dict[str, Dict[str, float]]] = None,
        history_limit: Optional[int] = None,
        excluded_tools_from_count: Optional[List[str]] = None,
        original_config_path: Optional[str] = None,
    ):
        """
        Args:
            benchmark_type: Type of benchmark (e.g., "vending_bench", "operation_bench")
            state_context_keys: Optional list of state keys to filter for logging and state checks.
            timeout: Optional timeout for LLM API calls in seconds. Defaults to 120.0 if not specified.
            resume_session_id: Optional session ID to resume from.
            session_config: Optional configuration dict to save with session metadata.
            env_model_name: Optional model name for environment tools. If None, uses model_name.
            max_actions_per_day: Optional maximum number of actions allowed per day.
            token_pricing: Optional token pricing configuration.
            history_limit: Optional maximum number of messages to keep in conversation history.
            excluded_tools_from_count: Optional list of tool names to exclude from daily action count.
            original_config_path: Optional path to original config file to copy to session directory.
        """
        self.benchmark_type = benchmark_type
        self.logger = logging.getLogger(benchmark_type)

        self.max_actions_per_day = max_actions_per_day
        self.daily_action_count = 0
        self.current_day = 0

        self.excluded_tools_from_count = excluded_tools_from_count if excluded_tools_from_count else []

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.token_pricing = token_pricing if token_pricing else {}
        self.model_name = model_name
        self.env_model_name_for_pricing = env_model_name

        try:
            self.get_token_price(model_name)
        except ValueError as e:
            self.logger.error(str(e))
            raise

        if env_model_name and env_model_name != model_name:
            try:
                self.get_token_price(env_model_name)
            except ValueError as e:
                self.logger.error(str(e))
                raise

        load_dotenv(env_path)
        
        model_supplier = self._get_model_supplier(model_name)
        env_model_supplier = self._get_model_supplier(env_model_name) if env_model_name else model_supplier
        
        api_key = self._get_api_credentials(model_supplier)['key']
        api_url = self._get_api_credentials(model_supplier)['url']
        env_api_key = self._get_api_credentials(env_model_supplier)['key']
        env_api_url = self._get_api_credentials(env_model_supplier)['url']
        
        timeout = timeout if timeout is not None else 120.0

        self.env_model_name = env_model_name

        model = OpenAI(api_key=api_key, base_url=api_url, timeout=timeout)
        env_model = OpenAI(api_key=env_api_key, base_url=env_api_url, timeout=timeout)

        self.use_memory = memory_config.get('use_memory', False)
        if self.use_memory:
            self.memory_manager = MemoryManager(model_name=self.env_model_name, memory_config=memory_config, llm_client=env_model)

        self.state_context_keys = state_context_keys

        self.start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.is_resuming = resume_session_id is not None
        self.session_id = resume_session_id if resume_session_id else f"{benchmark_type}_{self.start_time}"
        self.logger.info(f"Session ID: {self.session_id}")

        self.session_manager = SessionManager(self.session_id)

        self.session_log_dir = str(self.session_manager.session_dir)

        self.is_freelance = (self.benchmark_type == "freelance_bench")
        if self.is_freelance:
            self.logger.info("Initializing FreeLance specific logging paths...")
            log_dir = Path(self.session_log_dir)
            self.state_log_path = log_dir / "state_trace.jsonl"
            self.daily_log_path = log_dir / "daily_trace.jsonl"
            self.token_step_log_path = log_dir / "token_step_trace.jsonl"
            self.token_daily_log_path = log_dir / "token_daily_trace.jsonl"
            self.tasks_db_path = log_dir / "tasks_db_snapshot.json"

            self.daily_token_stats = {
                "input_tokens": 0, "output_tokens": 0,
                "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0
            }
            self.last_recorded_day = -1
            if self.is_resuming:
                 self.last_recorded_day = state.get('day', 0) - 1

        log_file_path = str(Path(self.session_log_dir) / "detailed.log")
        add_file_handler(logging.getLogger(), log_file_path)

        if self.is_resuming:
            resume_info = self.session_manager.get_resume_info()
            if resume_info:
                initial_state = resume_info['state']
                
                if self.is_freelance:
                    if "all_tasks_db" not in initial_state or not initial_state["all_tasks_db"]:
                        if os.path.exists(self.tasks_db_path):
                            self.logger.info(f"Restoring 'all_tasks_db' from snapshot: {self.tasks_db_path}")
                            try:
                                with open(self.tasks_db_path, 'r', encoding='utf-8') as f:
                                    snapshot_db = json.load(f)
                                    initial_state["all_tasks_db"] = snapshot_db
                            except Exception as e:
                                self.logger.error(f"Failed to restore tasks_db snapshot: {e}")
                        else:
                            self.logger.warning("Warning: Resuming freelance but tasks_db_snapshot.json not found!")
                
                self.resumed_from_step = resume_info['last_step']
                self.logger.info(f"Resuming session - continuing from Step {self.resumed_from_step}")
            else:
                self.logger.warning(f"Unable to resume session {self.session_id}, starting from scratch")
                self.is_resuming = False
                initial_state = dict(state) if state else {}
                self.resumed_from_step = 0
        else:
            initial_state = dict(state) if state else {}
            self.resumed_from_step = 0

            self.session_manager.init_session(config=session_config or {}, initial_state=initial_state)
            
            if session_config and 'full_config' in session_config:
                config_yaml_path = Path(self.session_log_dir) / "config.yaml"
                try:
                    with open(config_yaml_path, 'w', encoding='utf-8') as f:
                        yaml.dump(session_config['full_config'], f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                    self.logger.info(f"Applied configuration (with overrides) saved to: {config_yaml_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save applied config: {e}")
            elif original_config_path and os.path.exists(original_config_path):
                config_yaml_path = Path(self.session_log_dir) / "config.yaml"
                try:
                    shutil.copy2(original_config_path, config_yaml_path)
                    self.logger.info(f"Original configuration file copied to: {config_yaml_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to copy original config file: {e}")

        use_responses_api = model_name.startswith("closed_5.2_calling_pipeline")
        self.agent = Agent(
            model=model,
            tools=tools,
            instructions=system_prompt,
            model_id=model_name,
            initial_session_state=initial_state,
            history_limit=history_limit,
            use_responses_api=use_responses_api,
        )

        self.timer_tools = None
        for tool in tools:
            if hasattr(tool, '__self__'):
                tool_instance = tool.__self__
                if hasattr(tool_instance, 'task_done') and callable(getattr(tool_instance, 'task_done', None)):
                    self.timer_tools = tool_instance
                    break

        self.is_finished = is_finished
        self.cal_metric = cal_metric
        self.system_prompt = system_prompt
        self.base_system_prompt = system_prompt

        if self.agent.messages and len(self.agent.messages) > 0:
            system_msg = self.agent.messages[0]
            if system_msg.get("role") == "system":
                system_content = system_msg.get("content", "")
                self.logger.info("\n" + "=" * 60)
                self.logger.info("Agent System Prompt (complete):")
                self.logger.info("=" * 60)
                self.logger.info(system_content)
                self.logger.info("=" * 60 + "\n")

        if hasattr(self.agent, 'tool_schemas') and self.agent.tool_schemas:
            self.logger.info("\n" + "=" * 60)
            self.logger.info(f"Tool Schemas (total {len(self.agent.tool_schemas)}):")
            self.logger.info("=" * 60 + "\n")

            for idx, tool_schema in enumerate(self.agent.tool_schemas, 1):
                schema_str = json.dumps(tool_schema, indent=2, ensure_ascii=False)
                tool_name = (
                    tool_schema.get('function', {}).get('name', 'unknown')
                    if isinstance(tool_schema, dict)
                    else 'unknown'
                )

                self.logger.info(f"\n[{idx}] {tool_name}")
                self.logger.info(f"Schema:\n{schema_str}")

            self.logger.info("=" * 60 + "\n")
        else:
            self.logger.info("No tool schemas (agent.tool_schemas is empty)\n")

    def append_to_jsonl(self, path: Path, data: Dict):
        """Helper to append data to JSONL file"""
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to append to jsonl {path}: {e}")

    def _save_tasks_db_snapshot(self, tasks_db: Dict[str, Any]):
        """Helper to save the heavy tasks_db separately"""
        if not tasks_db: return
        try:
            with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(self.tasks_db_path), encoding='utf-8', delete=False) as tmp_f:
                json.dump(tasks_db, tmp_f, ensure_ascii=False)
                tmp_path = tmp_f.name

            if os.path.exists(self.tasks_db_path):
                os.remove(self.tasks_db_path)
            os.replace(tmp_path, self.tasks_db_path)
            self.logger.info(f"Saved tasks_db snapshot to {self.tasks_db_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save tasks_db snapshot: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _get_freelance_light_state(self, full_state: Dict[str, Any]) -> Dict[str, Any]:
        """Strip heavy database from state for logging"""
        light_state = full_state.copy()
        if "all_tasks_db" in light_state:
            light_state.pop("all_tasks_db")
        return light_state

    def _get_model_supplier(self, model_name: str) -> str:
        """Get supplier for specified model from token_pricing config"""
        if not self.token_pricing or model_name not in self.token_pricing:
            self.logger.warning(f"Model '{model_name}' not found in pricing config, defaulting to 'mtu' supplier")
            return "mtu"
        
        pricing_config = self.token_pricing[model_name]
        supplier = pricing_config.get("supplier", "mtu")
        return supplier
    
    def _get_api_credentials(self, supplier: str) -> Dict[str, str]:
        """Get API credentials for specified supplier"""
        supplier_upper = supplier.upper()
        
        # Try supplier-specific credentials (supports both BASE_URL and API_URL for compatibility)
        api_key = os.getenv(f"{supplier_upper}_API_KEY")
        api_url = os.getenv(f"{supplier_upper}_BASE_URL") or os.getenv(f"{supplier_upper}_API_URL")
        
        if not api_key or not api_url:
            # Fallback to standard OpenAI environment variables
            self.logger.warning(
                f"Supplier-specific credentials ({supplier_upper}_API_KEY, {supplier_upper}_BASE_URL) not found, "
                f"falling back to OPENAI_API_KEY and OPENAI_BASE_URL"
            )
            api_key = os.getenv("OPENAI_API_KEY")
            api_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_URL")
        
        if not api_key or not api_url:
            raise ValueError(
                f"API credentials not found for supplier '{supplier}'. "
                f"Please set {supplier_upper}_API_KEY and {supplier_upper}_BASE_URL in .env file, "
                f"or set OPENAI_API_KEY and OPENAI_BASE_URL as fallback."
            )
        
        self.logger.info(f"Using {supplier} supplier with URL: {api_url}")
        return {"key": api_key, "url": api_url}

    def get_token_price(self, model_name: str) -> Dict[str, float]:
        """Get token pricing for specified model"""
        if not self.token_pricing:
            raise ValueError(
                f"Token pricing configuration is empty. Please configure pricing for model '{model_name}' in model_config.token_pricing."
            )

        if model_name in self.token_pricing:
            pricing = self.token_pricing[model_name]
            if not isinstance(pricing, dict) or "input" not in pricing or "output" not in pricing:
                raise ValueError(f"Model '{model_name}' pricing configuration is incomplete. Must include 'input' and 'output' fields.")
            return pricing

        available_models = ", ".join(self.token_pricing.keys())
        raise ValueError(
            f"Model '{model_name}' has no token pricing configured.\n"
            f"Please add pricing configuration for this model in model_config.token_pricing.\n"
            f"Available models: {available_models}"
        )

    def calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """Calculate token usage cost"""
        pricing = self.get_token_price(model_name)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def get_full_state(self) -> Dict[str, Any]:
        """Get the complete unfiltered state from agent's session_state"""
        return dict(self.agent.session_state)

    def get_state(self) -> Dict[str, Any]:
        """Get filtered state (for logging only)"""
        if self.state_context_keys is None:
            return dict(self.agent.session_state)

        return {key: self.agent.session_state[key] for key in self.state_context_keys if key in self.agent.session_state}

    def get_next_action(self) -> str:
        """Generate the next action prompt

        Returns empty string to enable pure tool calling loop without user prompts.
        The first call will automatically get an initial prompt from Agent.
        """
        return ""

    def run(self, max_steps: int = 10) -> Dict[str, Any]:
        self.logger.info("=" * 60)
        if self.is_resuming:
            self.logger.info(f"Resuming simulation - continuing from Step {self.resumed_from_step + 1}")
        else:
            self.logger.info("Starting simulation")
        self.logger.info("=" * 60)

        if self.is_freelance and not self.is_resuming:
            initial_state = self.get_full_state()
            if "all_tasks_db" in initial_state:
                self._save_tasks_db_snapshot(initial_state["all_tasks_db"])

        start_step = self.resumed_from_step if self.is_resuming else 0

        previous_warning = None

        interrupted_by_user = False

        for step in range(start_step, max_steps):
            full_state = self.get_full_state()
            if self.is_finished(full_state):
                self.logger.info("Simulation stopped (termination condition met)")
                self.session_manager.mark_completed()
                break

            current_state = self.get_state()

            if self.max_actions_per_day is not None:
                state_day = full_state.get('day', 0)

                if state_day != self.current_day:
                    self.current_day = state_day
                    self.daily_action_count = 0

                if self.daily_action_count >= self.max_actions_per_day:
                    force_task_done_msg = (
                        f"[Daily Action Limit] Current day (Day {state_day}) has executed {self.daily_action_count} actions, "
                        f"reached max {self.max_actions_per_day} actions per day, automatically calling task_done() to advance to next day"
                    )
                    self.logger.info(force_task_done_msg)

                    if self.timer_tools:
                        try:
                            task_done_result = self.timer_tools.task_done(full_state)
                            self.logger.info(f"[Auto-call task_done] {task_done_result}")

                            self.agent.session_state.update(full_state)

                            self.current_day = full_state.get('day', 0)
                            self.daily_action_count = 0

                            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                            assistant_msg = {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [{
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "task_done",
                                        "arguments": "{}"
                                    }}]}

                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": task_done_result
                            }

                            self.agent.messages.append(assistant_msg)
                            self.agent.messages.append(tool_msg)

                            if getattr(self.agent, "use_responses_api", False):
                                try:
                                    serialized_result = self.agent._serialize_tool_result(task_done_result)
                                except Exception:
                                    serialized_result = str(task_done_result)

                                response_tool_call = {
                                    "type": "function_call",
                                    "call_id": tool_call_id,
                                    "name": "task_done",
                                    "arguments": "{}",
                                }
                                self.agent.response_inputs.append(response_tool_call)
                                self.agent.response_inputs.append({
                                    "type": "function_call_output",
                                    "call_id": tool_call_id,
                                    "output": serialized_result,
                                })

                            self.logger.info(f"\nStep {step + 1}/{max_steps}")
                            self.logger.info("=" * 80)
                            self.logger.info("[State] " + format_state_summary(current_state, key_fields=self.state_context_keys))
                            self.logger.info("[Agent Input] SYSTEM AUTO-EXECUTION: Daily action limit reached.")
                            self.logger.info("[Agent Output] No output")
                            self.logger.info(f"[Tool Call] task_done with args: {{}}")
                            self.logger.info(f"[Tool Output] task_done: {format_log_content(task_done_result, max_length=1000)}")

                            auto_state_before = self._get_freelance_light_state(current_state) if self.is_freelance else current_state
                            
                            auto_state_after_full = self.get_state()
                            auto_state_after = self._get_freelance_light_state(auto_state_after_full) if self.is_freelance else auto_state_after_full
                            
                            step_data = {
                                'step': step + 1,
                                'session_id': self.session_id,
                                'state_before': dict(auto_state_before),
                                'input_prompt': 'SYSTEM AUTO-EXECUTION: Daily action limit reached.',
                                'tools_called': [{
                                    'tool_name': 'task_done',
                                    'tool_args': {},
                                    'result': task_done_result,
                                    'error': None
                                }],
                                'state_after': dict(auto_state_after),
                                'is_finished': False
                            }
                            self.session_manager.save_step(step_data)
                            self.session_manager.save_state(self.get_full_state(), step + 1)

                            continue
                        except Exception as e:
                            error_msg = f"[Error] Auto-call task_done failed: {e}"
                            self.logger.error(error_msg)
                    else:
                        warning_msg = "[Warning] Unable to find TimerTools instance, cannot auto-call task_done"
                        self.logger.warning(warning_msg)

            log_state_before = self._get_freelance_light_state(current_state) if self.is_freelance else current_state
            step_data = {
                'step': step + 1,
                'session_id': self.session_id,
                'state_before': dict(log_state_before),
                'is_finished': False,
            }

            try:
                current_action = self.get_next_action()

                if self.use_memory:
                    query = current_action
                    
                    if not query and self.agent.messages:
                        for msg in reversed(self.agent.messages):
                            msg_dict = msg if isinstance(msg, dict) else msg.model_dump() if hasattr(msg, 'model_dump') else msg.__dict__
                            
                            content = msg_dict.get('content')
                            role = msg_dict.get('role')
                            
                            if role == 'system' or not content:
                                continue
                                
                            raw_text = str(content).strip()
                            if raw_text and raw_text != "None":
                                query = raw_text[-200:]
                                break
                    
                    retrieved_context = self.memory_manager.retrieve(query)
                    
                    if retrieved_context:
                        context_prompt = (
                            f"{self.base_system_prompt}\n\n"
                            f"====== DYNAMIC MEMORY CONTEXT ======\n"
                            f"{retrieved_context}\n"
                            f"====================================\n"
                        )
                        self.agent.update_system_prompt(context_prompt)
                        self.logger.info(f"[Memory] Injected context length: {len(retrieved_context)}")

                if previous_warning:
                    current_action = f"{current_action}\n\n{previous_warning}"
                    previous_warning = None

                current_state = self.get_state()

                self.logger.info(f"\nStep {step + 1}/{max_steps}")
                self.logger.info("=" * 80)
                self.logger.info("[State] " + format_state_summary(current_state, key_fields=self.state_context_keys))
                self.logger.info(f"[Agent Input] {format_log_content(current_action, max_length=1000)}")

                step_data['input_prompt'] = current_action

                result = self.agent.run(current_action, session_id=self.session_id)

                has_valid_tool_call = False
                tools_counted = 0

                if self.is_freelance and result.metrics:
                    i_tokens = result.metrics.input_tokens or 0
                    o_tokens = result.metrics.output_tokens or 0

                    try:
                        price = self.get_token_price(self.model_name)
                        i_cost = (i_tokens / 1_000_000) * price["input"]
                        o_cost = (o_tokens / 1_000_000) * price["output"]
                        t_cost = i_cost + o_cost

                        token_step_record = {
                            "step": step + 1,
                            "timestamp": datetime.now().isoformat(),
                            "tokens": {"input": i_tokens, "output": o_tokens, "total": i_tokens + o_tokens},
                            "cost": {"input": round(i_cost, 6), "output": round(o_cost, 6), "total": round(t_cost, 6)},
                            "cumulative_total_cost": round(self.total_cost, 6)
                        }
                        self.append_to_jsonl(self.token_step_log_path, token_step_record)

                        self.daily_token_stats["input_tokens"] += i_tokens
                        self.daily_token_stats["output_tokens"] += o_tokens
                        self.daily_token_stats["input_cost"] += i_cost
                        self.daily_token_stats["output_cost"] += o_cost
                        self.daily_token_stats["total_cost"] += t_cost

                    except Exception as e:
                        self.logger.error(f"Error logging freelance token stats: {e}")

                if result.tools:
                    for tool in result.tools:
                        if tool.tool_name:
                            has_valid_tool_call = True
                            if tool.tool_name not in self.excluded_tools_from_count:
                                tools_counted += 1

                if tools_counted > 0 and self.max_actions_per_day is not None:
                    self.daily_action_count += tools_counted
                    state_day = self.agent.session_state.get('day', 0)
                    self.logger.info(
                        f"[Daily Action Limit] Current day (Day {state_day}) has executed {self.daily_action_count}/{self.max_actions_per_day} actions (+{tools_counted} from this step)"
                    )

                if not has_valid_tool_call:
                    warning_msg = f"[Warning] Step {step + 1} did not detect valid tool calls. Agent must call at least one tool to continue. Please use JSON format tool schema (tool_calls/function calling) to call tools, do not write function names in plain text."
                    self.logger.warning(warning_msg)
                    step_data['warning'] = "No valid tool calls detected"

                    previous_warning = NO_CALL_WARNING

                step_data['result_content'] = result.get_content_as_string(indent=2) if result.content else None
                step_data['status'] = result.status.value if hasattr(result.status, 'value') else str(result.status)

                if self.use_memory:                    
                    new_messages = []
                    if current_action:
                        new_messages.append({"role": "user", "content": current_action})
                    
                    if result.messages:
                        for msg in result.messages:
                            new_messages.append(msg)
                            
                    self.memory_manager.add(new_messages, step_index=step+1)

                if result.messages:
                    step_data['messages'] = []
                    for msg in result.messages:
                        msg_data = {
                            'role': msg.role,
                            'content': msg.content if isinstance(msg.content, str) else str(msg.content)[:500]
                        }
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            msg_data['tool_calls'] = [
                                {
                                    'name': tc.get('function', {}).get('name', 'unknown') if isinstance(tc, dict) else 'unknown',
                                    'arguments': tc.get('function', {}).get('arguments', '{}') if isinstance(tc, dict) else '{}'
                                }
                                for tc in msg.tool_calls
                            ]
                        step_data['messages'].append(msg_data)

                llm_api_duration = None
                if result.messages:
                    for msg in result.messages:
                        if msg.role == "assistant":
                            if hasattr(msg, 'metrics') and msg.metrics and msg.metrics.duration is not None:
                                if llm_api_duration is None:
                                    llm_api_duration = msg.metrics.duration
                                    break

                if llm_api_duration is not None:
                    llm_time_msg = f"[LLM API Duration] {llm_api_duration:.3f}s (pure API call time, excluding tool execution)"
                    self.logger.info(llm_time_msg)
                else:
                    if result.metrics and result.metrics.duration is not None:
                        llm_time_msg = f"[LLM API Duration] {result.metrics.duration:.3f}s (from result.metrics, may include tool execution time)"
                        self.logger.info(llm_time_msg)

                if result.metrics:
                    step_data['metrics'] = {
                        'duration': result.metrics.duration if hasattr(result.metrics, 'duration') else None,
                        'input_tokens': result.metrics.input_tokens if hasattr(result.metrics, 'input_tokens') else 0,
                        'output_tokens': result.metrics.output_tokens if hasattr(result.metrics, 'output_tokens') else 0,
                        'total_tokens': result.metrics.total_tokens if hasattr(result.metrics, 'total_tokens') else 0,
                    }

                    if result.metrics.input_tokens > 0 or result.metrics.output_tokens > 0:
                        self.total_input_tokens += result.metrics.input_tokens
                        self.total_output_tokens += result.metrics.output_tokens

                        step_cost = self.calculate_cost(result.metrics.input_tokens, result.metrics.output_tokens, self.model_name)
                        self.total_cost += step_cost

                        step_data['metrics']['cost'] = step_cost

                        token_info = f"[Token Usage] Input: {result.metrics.input_tokens}, Output: {result.metrics.output_tokens}, Total: {result.metrics.total_tokens}"
                        cost_info = f"[Step Cost] ${step_cost:.6f} (Cumulative: ${self.total_cost:.6f})"
                        self.logger.info(token_info)
                        self.logger.info(cost_info)

                if result.messages:
                    for msg in result.messages:
                        if msg.role == "assistant":
                            if msg.content:
                                if isinstance(msg.content, str) and msg.content.strip():
                                    content_preview = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                                    self.logger.info(f"[Model Pre-Tool Output] {content_preview}")
                                elif isinstance(msg.content, list):
                                    content_str = str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content)
                                    self.logger.info(f"[Model Pre-Tool Output] {content_str}")

                            if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                                reasoning_preview = (
                                    str(msg.reasoning_content)[:1000] + "..."
                                    if len(str(msg.reasoning_content)) > 1000
                                    else str(msg.reasoning_content)
                                )
                                self.logger.info(f"[Model Reasoning] {reasoning_preview}")

                            if msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    if isinstance(tool_call, dict):
                                        func_info = tool_call.get('function', {})
                                        tool_name = func_info.get('name', 'unknown')
                                        tool_args_raw = func_info.get('arguments', '{}')
                                        if isinstance(tool_args_raw, str):
                                            try:
                                                tool_args = json.loads(tool_args_raw)
                                            except:
                                                tool_args = tool_args_raw
                                        else:
                                            tool_args = tool_args_raw
                                        tool_args_str = json.dumps(tool_args, ensure_ascii=False, indent=2)
                                        tool_args_preview = tool_args_str[:300] + "..." if len(tool_args_str) > 300 else tool_args_str
                                        self.logger.info(f"[Model Tool Call Request] {tool_name} - Args: {tool_args_preview}")

                if hasattr(result, 'reasoning_content') and result.reasoning_content:
                    reasoning_preview = (
                        str(result.reasoning_content)[:1000]
                        if len(str(result.reasoning_content)) > 1000
                        else str(result.reasoning_content)
                    )
                    self.logger.info(f"[Agent Reasoning] {reasoning_preview}")

                if result.content is not None:
                    try:
                        agent_output = result.get_content_as_string(indent=2)
                    except:
                        agent_output = str(result.content) if result.content else "No output"
                else:
                    agent_output = "No output"
                self.logger.info(f"[Agent Output] {format_log_content(agent_output, max_length=1000)}")

                agent_output_for_log = agent_output
                max_output_length = 2000
                if len(agent_output) > max_output_length:
                    agent_output_for_log = (
                        agent_output[:max_output_length] + f"... (content too long, truncated, total length: {len(agent_output)} chars)"
                    )
                self.logger.info(f"[Agent Output] {agent_output_for_log}")

                if result.tools:
                    step_data['tools_called'] = []

                    self.logger.info(f"[Tools Summary] Total {len(result.tools)} tool calls")
                    for tool in result.tools:
                        if tool.tool_name:
                            tool_data = {
                                'tool_name': tool.tool_name,
                                'tool_args': tool.tool_args,
                                'result': str(tool.result)[:500] if tool.result else None,
                                'error': tool.tool_call_error,
                            }
                            step_data['tools_called'].append(tool_data)

                            tool_args_formatted = format_log_content(tool.tool_args, max_length=300) if tool.tool_args else 'None'
                            self.logger.info(f"[Tool Call] {tool.tool_name} with args: {tool_args_formatted}")

                            tool_args_str = json.dumps(tool.tool_args, ensure_ascii=False, indent=2) if tool.tool_args else "None"
                            self.logger.info(f"[Tool Call] {tool.tool_name} - Args: {tool_args_str}")

                            tool_result = tool.result if tool.result else "No result"
                            status = "ERROR" if tool.tool_call_error else "SUCCESS"
                            tool_result_formatted = format_log_content(tool_result, max_length=1000)
                            self.logger.info(f"[Tool Output] {tool.tool_name} - Status: {status}, Result: {tool_result_formatted}")

                            tool_result_str = str(tool_result)[:500] if tool_result else "No result"
                            if len(str(tool_result)) > 500:
                                tool_result_str += "... (content too long, truncated)"
                            self.logger.info(f"[Tool Output] {tool.tool_name} - Result: {tool_result_str}")

                self.logger.debug(f"Step {step + 1} result: {result}")

                if result.status == RunStatus.cancelled:
                    self.logger.warning("Simulation interrupted by user (Ctrl+C)")
                    self.session_manager.mark_interrupted()
                    break

                full_state_after = self.get_full_state()
                self.cal_metric(full_state_after)

                if self.is_freelance:
                    light_state = self._get_freelance_light_state(full_state_after)
                    self.append_to_jsonl(self.state_log_path, {"step": step + 1, "state": light_state})

                if self.is_freelance:
                    new_day = full_state_after.get('day', 0)
                    if new_day > self.last_recorded_day:
                        daily_summary = {
                            "day": new_day,
                            "step": step + 1,
                            "summary": {k: full_state_after.get(k) for k in ['money', 'energy', 'stress', 'skill_rating'] if k in full_state_after}
                        }
                        self.append_to_jsonl(self.daily_log_path, daily_summary)

                        token_day_record = {
                            "day": new_day,
                            "step_end": step + 1,
                            "daily_stats": {
                                "input_tokens": self.daily_token_stats["input_tokens"],
                                "output_tokens": self.daily_token_stats["output_tokens"],
                                "input_cost": round(self.daily_token_stats["input_cost"], 6),
                                "output_cost": round(self.daily_token_stats["output_cost"], 6),
                                "total_cost": round(self.daily_token_stats["total_cost"], 6)
                            },
                            "session_cumulative": {
                                "total_cost": round(self.total_cost, 6)
                            }
                        }
                        self.append_to_jsonl(self.token_daily_log_path, token_day_record)

                        self.daily_token_stats = {
                            "input_tokens": 0, "output_tokens": 0,
                            "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0
                        }
                        self.last_recorded_day = new_day

                updated_state = self.get_state()
                log_state_after = self._get_freelance_light_state(updated_state) if self.is_freelance else updated_state
                step_data['state_after'] = dict(log_state_after)
                step_data['is_finished'] = self.is_finished(full_state_after)

                self.session_manager.save_step(step_data)

                full_state = self.get_full_state()
                save_state_payload = self._get_freelance_light_state(full_state) if self.is_freelance else full_state
                self.session_manager.save_state(save_state_payload, step + 1)

            except KeyboardInterrupt:
                interrupted_by_user = True
                self.session_manager.mark_interrupted()
                break
            except Exception as e:
                self.logger.exception(f"Step {step + 1} execution error: {e}")
                step_data['error'] = str(e)
                
                err_state = self.get_state()
                err_log_state = self._get_freelance_light_state(err_state) if self.is_freelance else err_state
                step_data['state_after'] = dict(err_log_state)

                self.session_manager.save_step(step_data)
                self.session_manager.mark_interrupted()
                raise

        if not interrupted_by_user:
            final_full_state = self.get_full_state()
            final_state = self.get_state()
            self.logger.info("=" * 60)
            if self.is_finished(final_full_state):
                self.logger.info("✅ Success: Task completed")
                self.session_manager.mark_completed()
                
                try:
                    if self.cal_metric is not None:
                        final_metrics = self.cal_metric(final_full_state)
                        if final_metrics:
                            self.session_manager.update_final_metrics(final_metrics)
                            self.logger.info("Final metrics saved to metadata")
                except Exception as e:
                    self.logger.warning(f"Failed to calculate or save final metrics: {e}")
            else:
                self.logger.warning("❌ Failed: Task not completed within max steps")
                metadata = self.session_manager.load_metadata()
                if metadata and metadata.get('status') == 'running':
                    self.session_manager.mark_interrupted()

            self.logger.info(f"Final state: {final_state}")
            self.logger.info(f"Session data saved to: {self.session_manager.session_dir}")

            self.logger.info("=" * 60)
            self.logger.info("Token Usage Statistics:")
            self.logger.info(f"  Total Input Tokens: {self.total_input_tokens:,}")
            self.logger.info(f"  Total Output Tokens: {self.total_output_tokens:,}")
            self.logger.info(f"  Total Tokens: {self.total_input_tokens + self.total_output_tokens:,}")
            self.logger.info(f"  Model: {self.model_name}")

            pricing = self.get_token_price(self.model_name)
            self.logger.info(f"  Pricing: Input ${pricing['input']:.2f}/1M tokens, Output ${pricing['output']:.2f}/1M tokens")
            self.logger.info(f"  💰 Total Cost: ${self.total_cost:.6f}")
            self.logger.info("=" * 60)

        if self.total_input_tokens > 0 or self.total_output_tokens > 0:
            try:
                pricing = self.get_token_price(self.model_name)
                self.session_manager.update_cost_info(
                    total_input_tokens=self.total_input_tokens,
                    total_output_tokens=self.total_output_tokens,
                    total_cost=self.total_cost,
                    model_name=self.model_name,
                    pricing=pricing
                )
            except Exception as e:
                self.logger.warning(f"Failed to save cost info to metadata: {e}")

        final_full_state = self.get_full_state()
        final_state = self.get_state()
        return final_state


def load_config(config_path_name: str) -> Dict[str, Any]:
    """Load YAML config with support for !include references."""
    config_path = Path(config_path_name)
    config_dir = config_path.parent

    with open(config_path_name, 'r', encoding="utf-8") as f:
        content = f.read()

    include_placeholders = {}

    def replace_include(match):
        placeholder = f"__INCLUDE_PLACEHOLDER_{uuid.uuid4().hex[:16]}__"
        include_placeholders[placeholder] = match.group(0)
        return placeholder

    content = re.sub(r'!include:[^\s\n]+', replace_include, content)

    config = yaml.safe_load(content)

    if 'task_config' in config and 'system_prompt' in config['task_config']:
        prompt = config['task_config']['system_prompt']

        if isinstance(prompt, str) and prompt in include_placeholders:
            include_spec = include_placeholders[prompt].replace('!include:', '').strip()
        elif isinstance(prompt, str) and prompt.strip().startswith('!include:'):
            include_spec = prompt.strip()[9:].strip()
        else:
            include_spec = None

        if include_spec:
            if ':' in include_spec:
                file_path, key = include_spec.split(':', 1)
            else:
                file_path = include_spec
                key = 'base_system_prompt'

            include_file = config_dir / file_path.strip()
            if include_file.exists():
                with open(include_file, 'r', encoding="utf-8") as f:
                    shared_config = yaml.safe_load(f)
                    if key in shared_config:
                        included_prompt = shared_config[key]
                        if 'system_prompt_prefix' in config.get('task_config', {}):
                            prefix = config['task_config']['system_prompt_prefix']
                            config['task_config']['system_prompt'] = f"{prefix}\n\n{included_prompt}"
                            del config['task_config']['system_prompt_prefix']
                        else:
                            config['task_config']['system_prompt'] = included_prompt
                    else:
                        raise ValueError(f"Key '{key}' not found in {include_file}")
            else:
                raise FileNotFoundError(f"Included file not found: {include_file}")

    return config


def load_benchmark_specific_modules(benchmark_type: str, config: Dict[str, Any], logger: logging.Logger, config_file_path: str = None) -> Dict[str, Any]:
    """
    Load benchmark-specific modules dynamically

    Returns:
        Dictionary containing:
        - tools: List of tool functions
        - is_finished: Termination condition function
        - cal_metric: Metric calculation function
        - timer_tools_class: Optional custom TimerTools class
    """
    if benchmark_type == "vending_bench":
        from agno.tools.vending.supplier import SupplierCommunicationTools
        from agno.tools.vending.seller import SalesTools, SalesModel
        from agno.tools.vending.timer import TimerTools
        from vending_bench_utils import vending_bench_is_finished, vending_bench_cal_metric

        supplier_config = config.get("tools_config", {}).get("supplier", {})
        
        model_config = config.get("model_config", {})
        model_pricing_path = model_config.get("token_pricing_file") or config_file_path
        if model_pricing_path and not Path(model_pricing_path).is_absolute():
            if config_file_path:
                config_dir = Path(config_file_path).parent
                model_pricing_path = str(config_dir / model_pricing_path)
        
        supplier_tools = SupplierCommunicationTools(
            product_db_path=supplier_config.get("product_db_path", "data/vending/products.jsonl"),
            use_embeddings=supplier_config.get("use_embeddings", True),
            embedding_model=supplier_config.get("embedding_model", "text-embedding-3-small"),
            model_pricing_config_path=model_pricing_path,
        )

        data_config = config.get("data_config", {})
        demand_structure_path = data_config.get("demand_structure_path") or config.get("task_config", {}).get("demand_structure_path")
        product_catalog_path = data_config.get("product_catalog_path") or config.get("task_config", {}).get("product_catalog_path")

        sales_model = SalesModel(
            demand_structure_path=demand_structure_path,
            product_catalog_path=product_catalog_path,
        )
        sales_tools = SalesTools(sales_model)
        timer_tools = TimerTools(sales_tools=sales_tools, supplier_tools=supplier_tools)

        all_tools = supplier_tools.tools + timer_tools.tools + sales_tools.tools

        no_sales_days_threshold = config.get("run_settings", {}).get("no_sales_days_threshold", 5)
        max_days = config.get("run_settings", {}).get("max_days", None)

        logger.info(f"Termination condition - No sales days threshold: {no_sales_days_threshold} days")
        if max_days:
            logger.info(f"Max simulation days: {max_days}")

        is_finished_func = partial(
            vending_bench_is_finished,
            no_sales_days_threshold=no_sales_days_threshold,
            max_days=max_days,
        )

        return {
            'tools': all_tools,
            'is_finished': is_finished_func,
            'cal_metric': vending_bench_cal_metric,
            'timer_tools_class': None,
        }

    elif benchmark_type == "operation_bench":
        from agno.tools.operation.platform_operator import PlatformOperatorTools
        from agno.tools.operation.timer import TimerTools
        from operation_bench_utils import operation_bench_is_finished, operation_bench_cal_metric

        platform_dynamics = config.get("platform_dynamics_config", {})
        logger.info(f"Platform dynamics configuration loaded: {len(platform_dynamics)} categories")

        platform_tools = PlatformOperatorTools(platform_dynamics=platform_dynamics)
        timer_tools = TimerTools()
        all_tools = platform_tools.tools + timer_tools.tools

        max_days = config.get("run_settings", {}).get("max_days", None)
        min_dau_threshold = config.get("run_settings", {}).get("min_dau_threshold", 100)

        if max_days:
            logger.info(f"Termination condition - Max days: {max_days}, Min DAU threshold: {min_dau_threshold}")
        else:
            logger.info(f"Termination condition - Min DAU threshold: {min_dau_threshold}")

        is_finished_func = partial(
            operation_bench_is_finished,
            max_days=max_days,
            min_dau_threshold=min_dau_threshold
        )

        if "initial_state" in config:
            config["initial_state"]["_platform_dynamics"] = platform_dynamics

        return {
            'tools': all_tools,
            'is_finished': is_finished_func,
            'cal_metric': operation_bench_cal_metric,
            'timer_tools_class': None,
        }

    elif benchmark_type == "freelance_bench":
        from agno.tools.freelance.task_pool import TaskManagementTools
        from agno.tools.freelance.task_settle import TaskExecutionTools
        from agno.tools.freelance.relax import RelaxTools
        from agno.tools.freelance.timer import TimerTools
        from freelance_bench_utils import freelance_is_finished, freelance_cal_metric, set_seed
        import tempfile

        seed = config.get("run_settings", {}).get("seed")
        if seed is not None:
            set_seed(seed)
            logger.info(f"Random seed set to: {seed}")

        if config_file_path is None:
            raise ValueError("config_file_path is required for freelance_bench")

        temp_config_file = None
        processed_config_path = config_file_path
        try:
            temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
            yaml.dump(config, temp_config_file, allow_unicode=True, default_flow_style=False, sort_keys=False)
            temp_config_file.close()
            processed_config_path = temp_config_file.name
            logger.info(f"Created temporary processed config file: {processed_config_path}")

            def cleanup_temp_config():
                if os.path.exists(processed_config_path):
                    try:
                        os.unlink(processed_config_path)
                        logger.info(f"Cleaned up temporary config file: {processed_config_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary config file {processed_config_path}: {e}")
            atexit.register(cleanup_temp_config)
        except Exception as e:
            logger.warning(f"Failed to create temporary config file, using original: {e}")
            processed_config_path = config_file_path

        dataset_path = config.get("system_config", {}).get("dataset_path", "./data/freelance/tasks.jsonl")

        task_mgmt_tools = TaskManagementTools(dataset_path=dataset_path, config_path=processed_config_path)

        if "initial_state" not in config:
            config["initial_state"] = {}

        logger.info(f"Preloading task database from {dataset_path} into initial state...")
        if hasattr(task_mgmt_tools, "_ensure_dataset_loaded"):
            try:
                task_mgmt_tools._ensure_dataset_loaded(config["initial_state"])
                db_size = len(config["initial_state"].get("all_tasks_db", {}))
                logger.info(f"✓ Successfully preloaded {db_size} tasks into state.")
            except Exception as e:
                logger.error(f"Failed to preload dataset: {e}")
        else:
            logger.warning("TaskManagementTools does not have _ensure_dataset_loaded method.")

        task_exec_tools = TaskExecutionTools(config_path=processed_config_path)
        relax_tools = RelaxTools(config_path=processed_config_path)
        timer_tools = TimerTools(config_path=processed_config_path)

        all_tools = (
            task_mgmt_tools.tools +
            task_exec_tools.tools +
            relax_tools.tools +
            timer_tools.tools
        )

        max_days = config.get("run_settings", {}).get("max_days", 100)
        logger.info(f"Termination condition - Max days: {max_days}")

        is_finished_func = partial(
            freelance_is_finished,
            max_days=max_days
        )

        return {
            'tools': all_tools,
            'is_finished': is_finished_func,
            'cal_metric': freelance_cal_metric,
            'timer_tools_class': None,
        }

    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")


def apply_config_overrides(config: Dict[str, Any], overrides: List[str], logger: logging.Logger) -> Dict[str, Any]:
    """
    Apply configuration overrides from command line arguments.

    Args:
        config: The configuration dictionary to modify
        overrides: List of override strings in format "key=value" or "path.to.key=value"
        logger: Logger instance for warnings

    Returns:
        Modified configuration dictionary

    Examples:
        --override "run_settings.max_actions_per_day=5"
        --override "model_config.timeout=60.0"
        --override "run_settings.max_days=100"
    """
    if not overrides:
        return config

    for override_str in overrides:
        if '=' not in override_str:
            logger.warning(f"Invalid override format: '{override_str}'. Expected format: 'key=value' or 'path.to.key=value'")
            continue

        key_path, value_str = override_str.split('=', 1)
        key_path = key_path.strip()
        value_str = value_str.strip()

        try:
            if '.' in value_str:
                try:
                    value = float(value_str)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    value = value_str
            else:
                try:
                    value = int(value_str)
                except ValueError:
                    if value_str.lower() in ('true', 'yes', 'on', '1'):
                        value = True
                    elif value_str.lower() in ('false', 'no', 'off', '0'):
                        value = False
                    else:
                        value = value_str
        except Exception:
            value = value_str

        keys = key_path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                logger.warning(f"Cannot override '{key_path}': '{'.'.join(keys[:keys.index(key)+1])}' is not a dictionary")
                break
            current = current[key]
        else:
            final_key = keys[-1]
            old_value = current.get(final_key, "N/A")
            current[final_key] = value
            logger.info(f"Config override: {key_path} = {value} (was: {old_value})")

    return config


def load_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a universal benchmark simulation.")
    parser.add_argument("--type", "-t", type=str, required=True, choices=["vending", "freelance", "operation", "v", "f", "o"],
                       dest="benchmark_type", help="Benchmark type: vending, freelance, or operation.")
    parser.add_argument("--config_path", type=str, default="./config/", help="Path to the configuration directory (default: ./config/).")
    parser.add_argument("--model_name", type=str, help="Override the model name from the config file.")
    parser.add_argument("--max_steps", type=int, help="Override the max steps from the config file.")

    parser.add_argument("--override", type=str, action="append", dest="overrides", default=[],
                       help="Override config values. Format: 'key=value' or 'path.to.key=value'. "
                            "Can be used multiple times. Examples: "
                            "--override 'run_settings.max_actions_per_day=5' "
                            "--override 'model_config.timeout=60.0'")

    parser.add_argument("--resume", type=str, help="Resume from a specific session ID.")
    parser.add_argument("--list-sessions", action="store_true", help="List all resumable sessions.")

    args = parser.parse_args()
    
    if args.benchmark_type == 'v': args.benchmark_type = 'vending'
    elif args.benchmark_type == 'f': args.benchmark_type = 'freelance'
    elif args.benchmark_type == 'o': args.benchmark_type = 'operation'

    benchmark_map = {
        "vending": ("vending_bench", "vending_config.yaml"),
        "freelance": ("freelance_bench", "freelance_config.yaml"),
        "operation": ("operation_bench", "operation_config.yaml")
    }

    args.benchmark_type_full, args.config_name = benchmark_map[args.benchmark_type]

    if not args.config_path.endswith("/"):
        args.config_path += "/"

    return args


if __name__ == '__main__':
    args = load_params()

    logger = setup_colored_logging(use_colors=True)

    try:
        if args.list_sessions:
            sessions = SessionManager.list_resumable_sessions()
            if not sessions:
                logger.info("No resumable sessions")
            else:
                logger.info(f"\nFound {len(sessions)} resumable sessions:\n")
                logger.info(f"{'No.':<6} {'Session ID':<40} {'Last Update':<25} {'Step':<8} {'Status':<12}")
                logger.info("=" * 100)
                for idx, session_info in enumerate(sessions, 1):
                    metadata = session_info['metadata']
                    session_id = session_info['session_id']
                    last_update = metadata.get('last_update', 'N/A')[:19]
                    last_step = metadata.get('last_step', 0)
                    status = metadata.get('status', 'unknown')
                    logger.info(f"{idx:<6} {session_id:<40} {last_update:<25} {last_step:<8} {status:<12}")
                logger.info("\nUse --resume <session_id> to resume a specific session\n")
            sys.exit(0)

        config = load_config(args.config_path + args.config_name)

        if args.overrides:
            config = apply_config_overrides(config, args.overrides, logger)

        resume_session_id = args.resume
        if resume_session_id:
            session_mgr = SessionManager(resume_session_id)
            if not session_mgr.can_resume():
                logger.error(f"Unable to resume session {resume_session_id}, session does not exist or status does not allow resume")
                logger.error(f"\nError: Unable to resume session {resume_session_id}")
                logger.error("Use --list-sessions to view all resumable sessions\n")
                sys.exit(1)

            resume_info = session_mgr.get_resume_info()
            logger.info(f"Resuming session: {resume_session_id}")
            logger.info(f"Last run stopped at Step {resume_info['last_step']}")

            saved_config = resume_info['metadata'].get('config', {})
            if not args.model_name and saved_config:
                model_name = saved_config.get('model_name', config["model_config"]["model_name"])
            else:
                model_name = args.model_name if args.model_name else config["model_config"]["model_name"]
        else:
            model_name = args.model_name if args.model_name else config["model_config"]["model_name"]

        max_steps = args.max_steps if args.max_steps else config["run_settings"]["max_steps"]

        timeout = config.get("model_config", {}).get("timeout", 120.0)
        env_model_name = config.get("model_config", {}).get("env_model_name", model_name)

        config_file_path = args.config_path + args.config_name
        benchmark_modules = load_benchmark_specific_modules(args.benchmark_type_full, config, logger, config_file_path)
        all_tools = benchmark_modules['tools']
        is_finished_func = benchmark_modules['is_finished']
        cal_metric_func = benchmark_modules['cal_metric']

        state_context_keys = config.get("state_context_keys")

        pricing_config_path = config.get("model_config", {}).get("pricing_config_path")
        if pricing_config_path:
            with open(pricing_config_path, 'r', encoding='utf-8') as f:
                pricing_data = yaml.safe_load(f)
                token_pricing = pricing_data.get("token_pricing", {})
        else:
            token_pricing = config.get("model_config", {}).get("token_pricing", {})

        max_actions_per_day = config.get("run_settings", {}).get("max_actions_per_day", None)

        excluded_tools_from_count = config.get("run_settings", {}).get("excluded_tools_from_count", ["task_done"])

        history_limit = config.get("run_settings", {}).get("history_limit", None)

        system_prompt = dedent(config["task_config"]["system_prompt"])
        if max_actions_per_day is not None:
            system_prompt = system_prompt.format(max_actions_per_day=max_actions_per_day)
        else:
            system_prompt = system_prompt.replace("{max_actions_per_day}", "unlimited")

        session_config = {
            'benchmark_type': args.benchmark_type_full,
            'model_name': model_name,
            'env_model_name': env_model_name,
            'max_steps': max_steps,
            'config_file': args.config_path + args.config_name,
            'timeout': timeout,
            'max_actions_per_day': max_actions_per_day,
            'full_config': config
        }

        original_config_path = args.config_path + args.config_name
        
        benchmark_launcher = BenchmarkLauncher(
            benchmark_type=args.benchmark_type_full,
            env_path=config["run_settings"]["env_path"],
            model_name=model_name,
            tools=all_tools,
            state=config["initial_state"],
            is_finished=is_finished_func,
            cal_metric=cal_metric_func,
            system_prompt=system_prompt,
            state_context_keys=state_context_keys,
            timeout=timeout,
            memory_config=config['memory_config'],
            resume_session_id=resume_session_id,
            session_config=session_config,
            env_model_name=env_model_name,
            max_actions_per_day=max_actions_per_day,
            token_pricing=token_pricing,
            history_limit=history_limit,
            excluded_tools_from_count=excluded_tools_from_count,
            original_config_path=original_config_path if not resume_session_id else None,
        )
        final_state = benchmark_launcher.run(max_steps=max_steps)

    except Exception as e:
        logger.exception(f"Program exited with error: {e}")
        raise

