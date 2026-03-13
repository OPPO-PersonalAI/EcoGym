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
import json
from typing import Dict, Any, Optional
from agno.tools.toolkit import Toolkit

class StatusTools(Toolkit):
    """
    Toolkit for the agent to actively check its own status and dashboard.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(
            name="status_tools",
            tools=[self.check_status],
            instructions="Use 'check_status' to view your current Money, Energy, Stress, Day, and Active Tasks.",
            **kwargs,
        )

    def check_status(self, session_state: Dict[str, Any]) -> str:
        """View current freelancer status and resource dashboard.
        
        Returns:
            A JSON string with status metrics:
            {
                "current_day": 2,
                "resources": {
                    "money": "$90.00",
                    "energy": "80/100",
                    "stress": "10/100",
                    "skills": {...}
                },
                "daily_progress": {...},
                "market_status": {...},
                "alerts": [...]
            }
        
        Note:
            - Effect: Read-only status check, no state changes
            - Provides alerts for critical resource levels (low money, energy, high stress)
        """
        day = session_state.get("day", 0)
        money = session_state.get("money", 0.0)
        energy = session_state.get("energy", 0)
        stress = session_state.get("stress", 0)
        skill = session_state.get("skill_rating", {})
        
        actions_today = session_state.get("actions_completed_today", 0)
        tasks_completed = session_state.get("tasks_completed_today", 0)
        
        task_pool = session_state.get("task_pool", [])
        active_task_count = len(task_pool)
        
        status_report = {
            "current_day": day,
            "resources": {
                "money": f"${money:.2f}",
                "energy": f"{energy}/100",
                "stress": f"{stress}/100",
                "skills": skill
            },
            "daily_progress": {
                "actions_used": actions_today,
                "tasks_finished_today": tasks_completed
            },
            "market_status": {
                "available_tasks_count": active_task_count,
                "message": "Call 'get_available_tasks' to see details." if active_task_count > 0 else "Task pool is empty."
            },
            "status": "active"
        }
        
        alerts = []
        if money < 20: alerts.append("CRITICAL: Low Money! Bankruptcy risk.")
        if energy < 20: alerts.append("WARNING: Low Energy. Rest recommended.")
        if stress > 80: alerts.append("WARNING: High Stress. Burnout risk.")
        
        if alerts:
            status_report["alerts"] = alerts

        return json.dumps(status_report, ensure_ascii=False)