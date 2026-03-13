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
from typing import Dict, Any, Optional, List
from agno.tools.toolkit import Toolkit

class TimerTools(Toolkit):
    """
    Toolkit for managing time progression and daily life cycle in the Freelance-Bench simulation.
    Implements the core game loop mechanic: advancing to the next day.
    
    This toolkit handles:
    1. Resource Recovery: Restoring Energy and reducing Stress upon sleeping.
    2. Economic Simulation: Deducting daily living costs automatically.
    3. Task Maintenance: Resetting daily counters and removing expired tasks from the pool.
    4. Game State Monitoring: Checks for bankruptcy (Money < 0) or time limit (Max Days) to trigger Game Over.
    """
    def __init__(self, config_path: str = "./freelance_bench_config.yaml", add_instructions: bool = True, **kwargs: Any):
        self.config = self._load_config(config_path)
        
        self.sys_conf = self.config.get("run_settings", {})
        self.max_days = self.sys_conf.get("max_days", 365)
        
        self.living_conf = self.config.get("living_settings", {})
        self.daily_living_cost = self.living_conf.get("daily_living_cost", 10)
        self.daily_energy_rec = self.living_conf.get("daily_energy_rec", 30)
        self.daily_stress_red = self.living_conf.get("daily_stress_red", 10)
        
        self.max_daily_tasks = self.config.get("task_settings_config", {}).get("max_tasks_per_day", 3)

        super().__init__(
            name="timer_tools",
            tools=[self.task_done],
            add_instructions=add_instructions,
            auto_register=True,
            show_result_tools=["task_done"],
            **kwargs,
        )

    def task_done(self, session_state: Dict[str, Any]) -> str:
        """Complete current day operations and advance to next day.
        
        Returns:
            A JSON string with day transition summary:
            {
                "status": "success",
                "event": "new_day",
                "current_day": 2,
                "daily_summary": {
                    "living_cost_deducted": 10.0,
                    "energy_recovered": 30,
                    "stress_relieved": 5,
                    "expired_tasks_removed": 0
                },
                "current_state": {...}
            }
        
        Note:
            - Effect: Advances day, recovers energy, reduces stress, deducts living cost
            - Resets daily task/action counters, removes expired tasks from pool
        """
        if "day" not in session_state: session_state["day"] = 0
        
        current_day = int(session_state["day"])
        
        if current_day >= self.max_days:
            return json.dumps({
                "status": "game_over",
                "reason": "max_days_reached",
                "message": f"Simulation ended. You reached Day {current_day}."
            }, ensure_ascii=False)

        current_money = float(session_state.get("money", 0.0))
        new_money = round(current_money - self.daily_living_cost, 2)
        
        current_energy = int(session_state.get("energy", 0))
        current_stress = int(session_state.get("stress", 0))
        
        new_energy = min(100, current_energy + self.daily_energy_rec)
        energy_gain = new_energy - current_energy
        
        new_stress = max(0, current_stress - self.daily_stress_red)
        stress_drop = current_stress - new_stress

        next_day = current_day + 1
        
        session_state["day"] = next_day
        session_state["money"] = new_money
        session_state["energy"] = new_energy
        session_state["stress"] = new_stress
        
        session_state["tasks_completed_today"] = 0 
        session_state["actions_completed_today"] = 0
        
        task_pool = session_state.get("task_pool", [])
        active_tasks = []
        expired_count = 0
        
        all_tasks_db = session_state.get("all_tasks_db", {})
        
        for task_id in task_pool:
            # task_pool contains task IDs (strings), need to look up task object from all_tasks_db
            task_obj = all_tasks_db.get(task_id, {})
            if task_obj.get("end_day", 999) >= next_day:
                active_tasks.append(task_id)
            else:
                expired_count += 1
        
        session_state["task_pool"] = active_tasks

        if "day_history" not in session_state: session_state["day_history"] = []
        session_state["day_history"].append({
            "day": current_day,
            "money": current_money,
            "energy": current_energy,
            "stress": current_stress
        })

        if new_money < 0:
             return json.dumps({
                "status": "game_over",
                "reason": "bankruptcy",
                "day": next_day,
                "final_money": new_money,
                "message": "You have run out of money to pay for living costs. Game Over."
            }, ensure_ascii=False)

        return json.dumps({
            "status": "success",
            "current_day": next_day,
            "events": {
                "living_cost_deducted": self.daily_living_cost,
                "energy_recovered": energy_gain,
                "stress_relieved": stress_drop,
                "expired_tasks_removed": expired_count,
                "daily_task_limit_reset": True
            },
            "current_state": {
                "money": new_money,
                "energy": new_energy,
                "stress": new_stress,
                "active_tasks_count": len(active_tasks)
            },
            "summary": f"Day {next_day}: Money=${new_money:.2f}, Energy={new_energy}/100, Stress={new_stress}/100"
        }, ensure_ascii=False)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            return {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except:
            return {}

