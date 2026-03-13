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
import statistics
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from agno.tools.toolkit import Toolkit

from agno.tools.freelance.task_final_price import TaskSettlementTools

class TaskExecutionTools(Toolkit):
    """
    Toolkit for the core workflow of performing tasks in Freelance-Bench.
    It integrates execution, evaluation, and automatic reward settlement into a single atomic action.
    
    Key Features:
    1. Solution Evaluation: Uses LLM Judges or Keyword Matching to grade the agent's work.
    2. Resource Management: Deducts Energy and adjusts Stress/Skills based on performance.
    3. Auto-Settlement: Automatically triggers the negotiation process to deposit earnings immediately after success.
    4. Daily Limits: Enforces the maximum number of tasks allowed per day.
    """
    def __init__(self, config_path: str = "./freelance_bench_config.yaml", api_key: Optional[str] = None, api_url: Optional[str] = None, add_instructions: bool = True, **kwargs: Any):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        env_path = self.config.get("system_config", {}).get("env_path")
        if env_path: load_dotenv(env_path)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.environ.get("API_KEY")
        self.api_url = api_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_URL") or os.getenv("API_URL")
        
        if self.api_key: self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        else: self.client = None
            
        self.model = self.config.get("system_config", {}).get("defaults", {}).get("system_model", "gpt-4o-mini")
        
        self.settlement_tool = TaskSettlementTools(config_path=self.config_path, api_key=self.api_key, api_url=self.api_url)

        super().__init__(
            name="task_execution_tools",
            tools=[self.solution_submit],
            add_instructions=add_instructions,
            auto_register=True,
            **kwargs,
        )

    def solution_submit(self, session_state: Dict[str, Any], task_id: str, solution_text: str) -> str:
        """Submit completed task solution for evaluation and payment processing.
        
        Args:
            task_id: Task ID from active pool (must be inspected first)
            solution_text: Final answer or generated content for the task
        
        Returns:
            A JSON string with evaluation and settlement results:
            {
                "status": "success",
                "is_success": true,
                "execution_stats": {
                    "energy_consumed": 12,
                    "current_stress": 25,
                    "skill_avg": 60.5
                },
                "settlement": {
                    "final_payment": 15.0,
                    "current_balance": 120.0
                },
                "message": "..."
            }
        
        Note:
            - Effect: Consumes energy, evaluates solution, updates stress/skills/money
            - Success reduces stress/increases skill, failure increases stress/reduces skill
        """
        max_daily_tasks = self.config.get("task_settings_config", {}).get("max_tasks_per_day", 3)
        tasks_today = session_state.get("tasks_completed_today", 0)
        
        if tasks_today >= max_daily_tasks:
            return json.dumps({
                "status": "blocked", 
                "reason": "daily_limit_reached", 
                "message": f"You have reached the daily limit of {max_daily_tasks} tasks. Please use 'task_done' to rest."
            }, ensure_ascii=False)
            
        db = session_state.get("all_tasks_db", {})
        task = db.get(task_id)  
        category = task.get("category", "General")
        
        if task_id not in session_state.get("task_pool", []):
            return json.dumps({
                "status": "error", 
                "message": "Task ID not in active pool."
            }, ensure_ascii=False)


        energy = session_state.get("energy", 0)
        stress = session_state.get("stress", 0)
        
        if "skill_rating" not in session_state or not isinstance(session_state["skill_rating"], dict):
            session_state["skill_rating"] = {}
        skills_dict = session_state["skill_rating"]
        current_skill = float(skills_dict.get(category, 60.0))        
        
        if task.get("init_effort"):
            task_effort = float(task["init_effort"])
        else:
            comp = task.get("complexity", "Medium")
            task_effort = 3.0 if comp == "Low" else (8.5 if comp == "High" else 5.5)

        skill_req = task_effort * 10
        cost_modifier = max(0.4, 1.0 - (current_skill - skill_req) / 100.0)
        energy_cost = int((2 + task_effort * 0.8) * cost_modifier)
        
        if energy < energy_cost:
            return json.dumps({
                "status": "failed", 
                "reason": "exhaustion", 
                "message": f"Need {energy_cost} energy."
            }, ensure_ascii=False)

        exec_stress_increase = int((1 + task_effort * 0.5) * cost_modifier)
        session_state["energy"] = max(0, energy - energy_cost)
        session_state["stress"] = min(100, stress + exec_stress_increase)

        is_success = self._evaluate_solution(task, solution_text)
        
        learning_mod = 1.0 if current_skill == 0 else max(0.1, min(2.0, skill_req / current_skill))

        if is_success:
            stress_drop = int((2 + task_effort * 0.5) * cost_modifier)
            session_state["stress"] = max(0, session_state["stress"] - stress_drop)
            
            skill_gain = (1.0 + task_effort * 0.25) * learning_mod
            skills_dict[category] = round(min(100.0, current_skill + skill_gain), 2)
            
            result_msg = "Success"
        else:
            stress_add = int((5 + task_effort * 1.0) * cost_modifier)
            session_state["stress"] = min(100, session_state["stress"] + stress_add)
            
            skill_penalty = 1.5 + task_effort * 0.1
            skills_dict[category] = round(max(0.0, current_skill - skill_penalty), 2)
            
            result_msg = "Failure"
        
        session_state["skill_rating"] = skills_dict

        trajectory = f"Task: {task.get('question', '')}\nSolution: {solution_text}\nResult: {result_msg}"
        
        settlement_json_str = self.settlement_tool.settle_task_payment(
            session_state=session_state,
            question=task.get("question", ""),
            category=task.get("category", "General"),
            complexity_hint=str(task.get("complexity", "Medium")),
            agent_trajectory=trajectory,
            is_success=is_success,
            init_money=task.get("init_payment", 5.0),
            init_effort=task.get("init_effort", 5.0)
        )
        
        final_money = 0.0
        negotiation_log = "Settlement failed or error."
        try:
            settlement_res = json.loads(settlement_json_str)
            if settlement_res.get("status") == "success":
                final_money = float(settlement_res.get("final_money", 0.0))
                negotiation_log = settlement_res.get("negotiation_history", "No negotiation.")
        except Exception as e:
            print(f"[Error] Parsing settlement result: {e}")

        if final_money > 0:
            session_state["money"] = round(session_state.get("money", 0) + final_money, 2)

        session_state["task_pool"] = [tid for tid in session_state["task_pool"] if tid != task_id]
        if "task_history" not in session_state: session_state["task_history"] = []
        session_state["task_history"].append(task_id)
        
        session_state["tasks_completed_today"] = tasks_today + 1
        
        print(f'[Task {is_success}], Money Changed: {final_money:.2f}')
        
        return json.dumps({
            "status": "success" if is_success else "failed",
            "is_success": is_success,
            "execution_stats": {
                "energy_consumed": energy_cost,
                "current_stress": session_state["stress"],
                "skill_avg": round(statistics.mean(session_state["skill_rating"].values()), 2) if session_state["skill_rating"] else 60.0
            },
            "settlement": {
                "final_payment": final_money,
                "current_balance": session_state["money"],
            },
            "message": f"Task {result_msg}. Payment: ${final_money:.2f}. Balance: ${session_state['money']:.2f}"
        }, ensure_ascii=False)

    def _evaluate_solution(self, task: Dict, solution: str) -> bool:
        method = task.get("eval_method", "LLM_JUDGE").upper()
        ground_truth = task.get("answer", "")
        question = task.get("question", "")
        
        if "EXACT_MATCH" in method or "KEYWORD" in method:
            return self._robust_keyword_match(solution, ground_truth)
        else:
            return self._llm_eval(question, solution, ground_truth)

    def _robust_keyword_match(self, user_sol: str, ref_sol: str) -> bool:
        if not ref_sol or not user_sol: return False
        def normalize(s: str) -> str:
            return re.sub(r'\s+', '', s).lower()
        return normalize(ref_sol) in normalize(user_sol)

    def _llm_eval(self, question: str, solution: str, reference: str) -> bool:
        if not self.client: return False
        prompt = f"""
        Compare the User Solution to the Reference Answer.
        Question: {question}
        Reference: {reference}
        User Solution: {solution}
        
        Output JSON: {{"is_correct": true}} or false.
        """
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = res.choices[0].message.content
            
            if "true" in content.lower(): return True
            
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match: 
                return json.loads(match.group()).get("is_correct", False)
            
            return False
        except: 
            return False

    def _load_config(self, path: str) -> Dict:
        if os.path.exists(path):
            with open(path, 'r') as f: return yaml.safe_load(f)
        return {}