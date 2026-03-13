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
import random
import concurrent.futures
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from agno.tools.toolkit import Toolkit

from agno.tools.freelance.task_init_price import TaskEstimationTools 

class TaskManagementTools(Toolkit):
    """
    Toolkit for managing the job market and task availability in Freelance-Bench.
    It acts as the 'Job Board' where the agent can browse, select, or hunt for new work.

    Key Features:
    1. Task Browsing: View currently valid tasks and their expiration times.
    2. Detail Retrieval: Fetching full task details triggers the 'TaskEstimationTools' to generate dynamic pricing.
    3. Headhunting (Refresh): Allows the agent to spend resources (Money/Energy) to find new task opportunities from the database.
    """
    def __init__(self, dataset_path: str = "./data/freelance/tasks.jsonl", config_path: str = "./freelance_bench_config.yaml", add_instructions: bool = True, **kwargs: Any):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        sys_conf = self.config.get("system_config", {})
        self.dataset_path = sys_conf.get("dataset_path", dataset_path)
        if sys_conf.get("env_path"):
            load_dotenv(sys_conf["env_path"])

        self.max_pool_size = self.config.get("task_settings_config", {}).get("max_pool_size", 10)
        
        self.pricing_tool = TaskEstimationTools(config_path=self.config_path)

        super().__init__(
            name="task_management_tools",
            tools=[self.tasks_browse, self.task_inspect, self.tasks_discover],
            add_instructions=add_instructions,
            auto_register=True,
            **kwargs,
        )

    def _ensure_dataset_loaded(self, session_state: Dict[str, Any]):
        """
        Internal: Loads JSONL into session_state['all_tasks_db'] if not present.
        Assigns simple sequential IDs.
        """
        if "all_tasks_db" not in session_state:
            session_state["all_tasks_db"] = {}

            if "task_pool" not in session_state: session_state["task_pool"] = []    
            if "task_history" not in session_state: session_state["task_history"] = [] 

            if not os.path.exists(self.dataset_path):
                print(f"[Error] Dataset not found: {self.dataset_path}")
                return

            try:
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                for idx, line in enumerate(lines):
                    data = json.loads(line)
                    task_id = str(idx)
                    data["id"] = task_id

                    if "init_payment" not in data: data["init_payment"] = None
                    if "init_effort" not in data: data["init_effort"] = None
                    
                    session_state["all_tasks_db"][task_id] = data
                    
            except Exception as e:
                print(f"[Error] Failed to load dataset: {e}")

    def tasks_browse(self, session_state: Dict[str, Any]) -> str:
        """View currently available tasks in the job market pool.
        
        Returns:
            A JSON string with task summaries:
            [
                {
                    "task_id": "102",
                    "category": "Data Entry",
                    "complexity": "Low",
                    "status": "Available",
                    "days_left": 3,
                    "estimated_payment": "Calculate in details"
                },
                ...
            ]
        
        Note:
            - Effect: Lists active tasks with expiration times, filters expired tasks
            - Tasks expire after complexity-based lifespan (Low: 3d, Medium: 5d, High: 7d)
        """
        if session_state is None:
            return json.dumps({
                "status": "error", 
                "message": "Internal Error: Session state missing."
            }, ensure_ascii=False)

        self._ensure_dataset_loaded(session_state)
        
        current_day = session_state.get("day", 0)
        active_ids = session_state.get("task_pool", [])
        
        valid_ids = []
        display_list = []
        
        db = session_state["all_tasks_db"]
        
        for tid in active_ids:
            task = db.get(tid)
            if not task: continue
            
            end_day = task.get("end_day", 999)

            if end_day >= current_day:
                valid_ids.append(tid)
                display_list.append({
                    "task_id": tid,
                    "category": task.get("category", "General"),
                    "complexity": task.get("complexity", "Medium"),
                    "status": "Available",
                    "days_left": end_day - current_day,
                    "                    estimated_payment": task.get("init_payment", "Calculate in details"),
                })
        
        session_state["task_pool"] = valid_ids
        
        if not display_list:
            return json.dumps({
                "status": "empty", 
                "message": "Task pool is empty or all tasks expired. Please use 'tasks_discover'."
            }, ensure_ascii=False)

        return json.dumps(display_list, ensure_ascii=False)

    def task_inspect(self, session_state: Dict[str, Any], task_id: str) -> str:
        """Retrieve full details and pricing for a specific task.
        
        Args:
            task_id: Task ID from tasks_browse() results
        
        Returns:
            A JSON string with complete task information:
            {
                "status": "selected",
                "task_id": "102",
                "category": "Data Entry",
                "complexity": "Low",
                "init_payment": 15.0,
                "init_effort": 6.5,
                "question": "...",
                "end_day": 5
            }
        
        Note:
            - Effect: Calculates payment/effort if not cached, no resource changes
            - Uses TaskEstimationTools for dynamic pricing based on complexity
        """
        if session_state is None:
            return json.dumps({
                "status": "error", 
                "message": "Internal Error: Session state missing."
            }, ensure_ascii=False)

        self._ensure_dataset_loaded(session_state)
        
        if task_id not in session_state.get("task_pool", []):
            db_task = session_state["all_tasks_db"].get(task_id)
            if db_task and db_task.get("end_day", 0) < session_state.get("day", 0):
                return json.dumps({
                    "status": "error", 
                    "message": f"Task ID {task_id} has expired."
                }, ensure_ascii=False)
                
            return json.dumps({
                "status": "error", 
                "message": f"Task ID {task_id} not in active pool. Please check 'tasks_browse' again."
            }, ensure_ascii=False)
        
        task = session_state["all_tasks_db"].get(task_id)
        if not task:
            return json.dumps({
                "status": "error", 
                "message": "Task data corrupted."
            }, ensure_ascii=False)

        if task.get("init_payment") is None or task.get("init_effort") is None:
            print(f"--- [TaskManagement] Task {task_id} missing pricing. Calling Estimation Tool... ---")
            try:
                price_json_str = self.pricing_tool.estimate_task_price(
                    session_state=session_state,
                    question=task.get("question", ""),
                    category=task.get("category", "General"),
                    complexity_hint=task.get("complexity", "Medium")
                )
                price_data = json.loads(price_json_str)
                
                if price_data.get("status") == "success":
                    task["init_payment"] = float(price_data.get("base_payment", 5.0))
                    task["init_effort"] = float(price_data.get("estimated_effort", 5.0))
                    session_state["all_tasks_db"][task_id] = task
                else:
                    task["init_payment"] = 5.0
                    task["init_effort"] = 5.0
            except Exception as e:
                print(f"[Error] calling estimation tool: {e}")
                task["init_payment"] = 5.0
                task["init_effort"] = 5.0

        return json.dumps({
            "status": "selected",
            "task_id": task_id,
            "category": task.get("category"),
            "complexity": task.get("complexity"),
            "init_payment": task["init_payment"],
            "init_effort": task["init_effort"],
            "question": task.get("question"),
            "end_day": task.get("end_day"),
            "message": "Task details retrieved. You can now solve and submit this task."
        }, ensure_ascii=False)

    def tasks_discover(self, session_state: Dict[str, Any], refresh_type: str) -> str:
        """Search for new tasks to replenish the available task pool.
        
        Args:
            refresh_type: Search mode determining resource costs
                - "free": No cost, high energy (~8), low stress (~1), finds ~4 tasks
                - "paid": Costs $2, lower energy (~5), higher stress (~3), finds ~6 tasks
        
        Returns:
            A JSON string with refresh results:
            {
                "status": "success",
                "added_count": 4,
                "current_pool_size": 8,
                "message": "Pool refreshed..."
            }
        
        Note:
            - Effect: Adds new tasks, prices them automatically, consumes money/energy
            - Pool size capped at max_pool_size (oldest tasks removed if exceeded)
        """
        if session_state is None:
            return json.dumps({
                "status": "error", 
                "message": "Internal Error: Session state missing."
            }, ensure_ascii=False)

        self._ensure_dataset_loaded(session_state)
        rtype = refresh_type.lower()
        
        if rtype == "paid":
            cost_money, cost_energy, add_stress, count = 2, 5, 3, 6
        else: # free
            cost_money, cost_energy, add_stress, count = 0, 8, 1, 4

        if session_state.get("money", 0) < cost_money:
            return json.dumps({
                "status": "error", 
                "message": f"Insufficient Money. Need ${cost_money}."
            }, ensure_ascii=False)
            
        if session_state.get("energy", 0) < cost_energy:
            return json.dumps({
                "status": "error", 
                "message": f"Insufficient Energy. Need {cost_energy} energy."
            }, ensure_ascii=False)

        session_state["money"] = round(session_state["money"] - cost_money, 2)
        session_state["energy"] = max(0, session_state["energy"] - cost_energy)
        session_state["stress"] = min(100, session_state.get("stress", 0) + add_stress)

        all_ids = set(session_state["all_tasks_db"].keys())
        history_ids = set(session_state.get("task_history", []))
        current_pool_ids = set(session_state.get("task_pool", []))
        
        available_candidates = list(all_ids - history_ids - current_pool_ids)
        
        if not available_candidates:
            return json.dumps({
                "status": "warning", 
                "message": "No new tasks available in dataset."
            }, ensure_ascii=False)

        sample_size = min(count, len(available_candidates))
        new_ids = random.sample(available_candidates, sample_size)
        
        print(f"--- [TaskManagement] Parallel pricing for {len(new_ids)} new tasks... ---")

        def _estimate_single_task(tid):
            task_data = session_state["all_tasks_db"].get(tid)
            if not task_data: return tid, 5.0, 5.0
            
            if task_data.get("init_payment") is not None and task_data.get("init_effort") is not None:
                return tid, task_data["init_payment"], task_data["init_effort"]

            try:
                price_json_str = self.pricing_tool.estimate_task_price(
                    session_state=session_state, 
                    question=task_data.get("question", ""),
                    category=task_data.get("category", "General"),
                    complexity_hint=task_data.get("complexity", "Medium")
                )
                price_data = json.loads(price_json_str)
                if price_data.get("status") == "success":
                    return tid, float(price_data.get("base_payment", 5.0)), float(price_data.get("estimated_effort", 5.0))
            except Exception:
                pass
            return tid, 5.0, 5.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_tid = {executor.submit(_estimate_single_task, tid): tid for tid in new_ids}
            for future in concurrent.futures.as_completed(future_to_tid):
                tid, payment, effort = future.result()
                session_state["all_tasks_db"][tid]["init_payment"] = payment
                session_state["all_tasks_db"][tid]["init_effort"] = effort
        
        current_day = session_state.get("day", 0)
        for tid in new_ids:
            task = session_state["all_tasks_db"][tid]
            comp = task.get("complexity", "Medium")
            if comp == "Low": lifespan = 3
            elif comp == "High": lifespan = 7
            else: lifespan = 5
            task["end_day"] = current_day + lifespan
            session_state["all_tasks_db"][tid] = task

        current_pool_list = session_state.get("task_pool", [])
        current_pool_list.extend(new_ids)
        
        if len(current_pool_list) > self.max_pool_size:
            overflow = len(current_pool_list) - self.max_pool_size
            current_pool_list = current_pool_list[overflow:] 
            
        session_state["task_pool"] = current_pool_list
        
        return json.dumps({
            "status": "success", 
            "added_count": len(new_ids), 
            "current_pool_size": len(current_pool_list), 
            "message": "Pool refreshed. Warning: Old tasks may have been removed."
        }, ensure_ascii=False)

    def _load_config(self, path: str) -> Dict:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)
        return {}