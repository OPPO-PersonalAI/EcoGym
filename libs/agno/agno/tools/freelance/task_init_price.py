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

class TaskEstimationTools(Toolkit):
    """
    Toolkit for pre-task assessment and market pricing in Freelance-Bench.
    Uses an ensemble (voting) of LLM models to objectively estimate the difficulty and fair reward for a given task.
    
    This establishes the 'Contract' baseline (init_money, init_effort) before the agent commits to the work.
    Dependencies: Requires OpenAI API key to run the voting models.
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
            print("[Warn] TaskEstimationTools: OpenAI API Key is missing.")

        self.sys_conf = self.config.get("system_config", {})
        self.temperature = self.sys_conf.get("temperature", 1.0)
        self.top_p = self.sys_conf.get("top_p", 0.95)
        self.max_retries = self.sys_conf.get("max_retries", 3)

        super().__init__(name="task_estimation_tools", tools=[self.estimate_task_price], add_instructions=add_instructions, auto_register=True, **kwargs)

    def _remove_outliers(self, values: List[float], threshold: float = 2.5) -> List[float]:
        """Remove outliers using MAD (Median Absolute Deviation) method.

        Args:
            values: List of numerical values to filter.
            threshold: Number of MADs away from median to consider as outlier.

        Returns:
            List of values with outliers removed.
        """
        if len(values) < 2:
            return values

        median = statistics.median(values)
        deviations = [abs(v - median) for v in values]
        mad = statistics.median(deviations)

        if mad == 0:
            return values

        modified_z_scores = [0.6745 * abs(v - median) / mad for v in values]

        filtered = [v for i, v in enumerate(values) if modified_z_scores[i] < threshold]

        if len(filtered) < max(2, len(values) // 2):
            return values

        return filtered if filtered else values

    def estimate_task_price(self, session_state: Dict[str, Any], question: str, category: str, complexity_hint: str) -> str:
        """Calculate the initial market rate and expected effort for a task.
        
        This tool MUST be called BEFORE starting to solve a task.
        It polls multiple models to find a consensus on price and difficulty.
        
        Args:
            question: The task description or prompt to evaluate.
            category: The domain of the task (e.g., "Math", "Creative Writing").
            complexity_hint: A hint provided by the system regarding difficulty.

        Returns Example:
            A JSON string containing the estimated baseline values:
            {
                "status": "success",
                "base_payment": 12.50,       # The average price voted by models
                "estimated_effort": 7.2      # The average effort score (1-10)
            }
            These values act as the baseline for the final settlement negotiation.
        """
        if not self.client:
            return json.dumps({
                "status": "error", 
                "message": "OpenAI Client not initialized."
            }, ensure_ascii=False)

        defaults = self.sys_conf.get("defaults", {})
        model_names_list = defaults.get("initial_voting_models", ["gpt-4o", "gpt-4o-mini"])

        prompt_tpl = self.config.get("initial_pricing_config", {}).get("prompt_template", "")
        if not prompt_tpl:
            return json.dumps({
                "status": "error", 
                "message": "Prompt template missing in YAML."
            }, ensure_ascii=False)

        prompt = prompt_tpl.format(question=question, category=category, complexity_hint=complexity_hint)

        valid_efforts = []
        valid_payments = []

        for i, model in enumerate(model_names_list):
            result = self._call_llm(prompt, model_name=model)

            if result:
                e = float(result.get("estimated_effort", 0.0))
                p = float(result.get("base_payment", 0.0))

                if e > 0 and p > 0:
                    valid_efforts.append(e)
                    valid_payments.append(p)
            else:
                print(f"[Warn] Model {model} failed or returned invalid JSON.")

        if not valid_efforts:
            print("[Error] No valid responses. Using fallback values.")
            final_payment, final_effort = 5.0, 5.0
        else:
            filtered_efforts = self._remove_outliers(valid_efforts)
            filtered_payments = self._remove_outliers(valid_payments)

            final_effort = round(statistics.mean(filtered_efforts), 2)
            final_payment = round(statistics.mean(filtered_payments), 2)

            if len(filtered_efforts) < len(valid_efforts):
                print(f"[Info] Removed {len(valid_efforts) - len(filtered_efforts)} outlier(s) from effort estimates.")
            if len(filtered_payments) < len(valid_payments):
                print(f"[Info] Removed {len(valid_payments) - len(filtered_payments)} outlier(s) from payment estimates.")

        if session_state is not None:
            session_state["init_money"] = final_payment
            session_state["init_effort"] = final_effort

        return json.dumps({
            "status": "success", 
            "base_payment": final_payment, 
            "estimated_effort": final_effort
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