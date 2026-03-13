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
import os
import json
import yaml
import time
import re
import statistics
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

from agno.tools.toolkit import Toolkit


class TaskSettlementTools(Toolkit):
    """
    Toolkit for finalizing task outcomes and processing rewards in Freelance-Bench.
    It acts as the 'Payroll System', handling effort calculation and salary negotiation.

    Key Features:
    1. Effort Analysis: Calculates the actual effort cost based on the length of the agent's execution trajectory.
    2. Automated Negotiation: Simulates a negotiation process between the worker and the system to determine the final payment.
    3. Failure Handling: Enforces zero payment policies for failed tasks.
    """
    def __init__(self, config_path: str = "./freelance_bench_config.yaml", api_key: Optional[str] = None, api_url: Optional[str] = None, add_instructions: bool = True, **kwargs: Any):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        env_path = self.config.get("system_config", {}).get("env_path")
        if env_path: load_dotenv(env_path)
            
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.environ.get("API_KEY")
        self.api_url = api_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_URL") or os.getenv("API_URL")
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        else:
            self.client = None
            print("[Warn] TaskSettlementTools: OpenAI API Key is missing.")

        self.sys_conf = self.config.get("system_config", {})
        self.temperature = self.sys_conf.get("temperature", 1.0)
        self.top_p = self.sys_conf.get("top_p", 0.95)
        self.max_retries = self.sys_conf.get("max_retries", 3)

        super().__init__(name="task_settlement_tools", tools=[self.settle_task_payment], add_instructions=add_instructions, auto_register=True, **kwargs)

    def settle_task_payment(self, session_state: Dict[str, Any], question: str, category: str, complexity_hint: str, agent_trajectory: str, is_success: bool, init_money: Optional[float] = None, init_effort: Optional[float] = None) -> str:
        """Calculate the final payment and effort cost for a completed task.
        
        This tool MUST be called immediately after a task is finished (whether successfully or not).
        It triggers an automated negotiation to determine the final reward based on the work done.

        Args:
            question: The original task description or prompt.
            category: The category of the task (e.g., "Data Entry", "Creative Writing").
            complexity_hint: The difficulty level indicator (e.g., "Easy", "Hard").
            agent_trajectory: The full history/log of steps taken to solve the task. Used to calculate 'final_effort'.
            is_success: Boolean indicating if the task was completed correctly. If False, payment is 0.
            init_money: (Optional) The base reward promised. Defaults to session_state['init_money'].
            init_effort: (Optional) The estimated effort cost. Defaults to session_state['init_effort'].

        Returns Example:
            A JSON string with the financial results:
            {
                "status": "success",
                "final_money": 8.50,       # The actual amount earned (added to wallet)
                "final_effort": 6.2,       # The effort consumed (deducted from energy)
                "negotiation_history": "Round 1: Agent asked $10..." 
            }
        """
        if not self.client:
            return json.dumps({
                "status": "error", 
                "message": "OpenAI Client not initialized."
            }, ensure_ascii=False)

        if session_state:
            if init_money is None: init_money = session_state.get("init_money", 5.0)
            if init_effort is None: init_effort = session_state.get("init_effort", 5.0)
        
        init_money = float(init_money) if init_money is not None else 5.0
        init_effort = float(init_effort) if init_effort is not None else 5.0

        chars_unit = self.config.get("effort_calculation_config", {}).get("chars_per_unit_effort", 100)
        traj_len = len(agent_trajectory)
        
        if traj_len > 6000:
            traj_snippet = agent_trajectory[:3000] + "\n...[SNIP]...\n" + agent_trajectory[-3000:]
        else:
            traj_snippet = agent_trajectory

        final_effort = round(max(1.0, min(10.0, traj_len / chars_unit)), 2)

        if not is_success:
            print("--- [TaskSettlement] Task Failed. Zero Payment. ---")
            return json.dumps({
                "status": "success", 
                "final_money": 0.0, 
                "final_effort": final_effort, 
                "message": "Task failed."
            }, ensure_ascii=False)

        defaults = self.sys_conf.get("defaults", {})
        agent_model = defaults.get("agent_model", "gpt-4o-mini")
        system_model = defaults.get("system_model", "gpt-4o-mini")
        
        neg_conf = self.config.get("negotiation_config", {})
        max_rounds = neg_conf.get("max_rounds", 3)
        deal_threshold = neg_conf.get("deal_threshold", 0.1)

        agent_tpl = neg_conf.get("agent_prompt_template", "")
        sys_tpl = neg_conf.get("system_prompt_template", "")

        negotiation_history_lines = []
        negotiation_history_text = "No prior negotiation."
        current_system_offer = init_money
        final_money = init_money

        print(f"\n--- [TaskSettlement] Negotiation Start (Init: ${init_money}) ---")

        for round_idx in range(max_rounds):
            agent_prompt = agent_tpl.format(
                question=question, category=category, complexity_hint=complexity_hint, 
                init_money=init_money, init_effort=init_effort, 
                agent_trajectory=traj_snippet, final_effort=final_effort, 
                negotiation_history=negotiation_history_text
            )
            agent_res = self._call_llm(agent_prompt, model_name=agent_model)
            agent_ask = float(agent_res.get("proposed_money", current_system_offer))
            agent_reason = agent_res.get("reasoning", "No reason")[:100]

            log_entry = f"Round {round_idx+1} [Agent]: Asking ${agent_ask:.2f}. Reason: {agent_reason}"
            negotiation_history_lines.append(log_entry)
            negotiation_history_text = "\n".join(negotiation_history_lines)
            print(f"> R{round_idx+1} [Agent]: ${agent_ask:.2f}")

            if agent_ask <= current_system_offer + deal_threshold:
                final_money = current_system_offer
                print(">>> Deal Struck! Agent accepted offer.")
                break

            sys_prompt = sys_tpl.format(
                question=question, category=category, complexity_hint=complexity_hint, 
                init_money=init_money, init_effort=init_effort, 
                agent_trajectory=traj_snippet, final_effort=final_effort, 
                negotiation_history=negotiation_history_text
            )
            sys_res = self._call_llm(sys_prompt, model_name=system_model)
            system_offer = float(sys_res.get("proposed_money", init_money))
            sys_reason = sys_res.get("reasoning", "No reason")[:100]

            log_entry = f"Round {round_idx+1} [System]: Offering ${system_offer:.2f}. Reason: {sys_reason}"
            negotiation_history_lines.append(log_entry)
            negotiation_history_text = "\n".join(negotiation_history_lines)
            print(f"> R{round_idx+1} [System]: ${system_offer:.2f}")

            if system_offer >= agent_ask - deal_threshold:
                final_money = system_offer
                print(">>> Deal Struck! System accepted ask.")
                break

            if abs(system_offer - current_system_offer) < 0.01 and round_idx > 0:
                final_money = system_offer
                print(">>> Negotiation Stalled.")
                break

            current_system_offer = system_offer
            if round_idx == max_rounds - 1:
                final_money = max(init_money, current_system_offer)

        return json.dumps({
            "status": "success", 
            "final_money": round(final_money, 2), 
            "final_effort": final_effort
        }, ensure_ascii=False)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path): return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        return {}

    def _call_llm(self, prompt: str, model_name: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                )
                return self._extract_json(response.choices[0].message.content)
            except Exception as e:
                print(f"[Retry {attempt+1}] {model_name} error: {e}")
                time.sleep(0.5 * (attempt + 1))
        return {}