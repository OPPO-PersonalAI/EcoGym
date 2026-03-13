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
from typing import Any, Dict, List
import json

from agno.tools.toolkit import Toolkit


class TimerTools(Toolkit):
    """
    Timer toolkit for platform operation scenario.
    
    Controls day progression and triggers platform dynamics simulation.
    """

    def __init__(
        self,
        add_instructions: bool = True,
        **kwargs: Any,
    ):
        tools = [self.task_done]

        super().__init__(
            name="timer_tools",
            tools=tools,
            add_instructions=add_instructions,
            auto_register=True,
            show_result_tools=["task_done"],
            **kwargs,
        )

    def task_done(self, session_state: Dict[str, Any]) -> str:
        """Complete current day operations, simulate platform evolution and advance to next day.
        
        Returns:
            A JSON string with day transition summary including:
            - New day number
            - DAU changes
            - Retention rate
            - Content ecosystem updates
            - Platform health metrics
        
        Note:
            - Effect: Advances day, simulates platform dynamics
            - Automatically updates DAU, content quality, creator activity, engagement
        """
        if session_state is None:
            session_state = {}

        if "day" not in session_state:
            session_state["day"] = 0

        current_day = int(session_state["day"])

        # Import here to avoid circular dependency
        from agno.tools.operation.platform_operator import simulate_platform_day
        
        day_result = simulate_platform_day(session_state)

        new_day = current_day + 1
        session_state["day"] = new_day

        day_history: List[int] = session_state.setdefault("day_history", [])
        day_history.append(new_day)

        current_state = {
            "dau": session_state.get("dau", 0),
            "content_volume": session_state.get("content_volume", 0),
            "content_quality": round(session_state.get("content_quality", 0.5), 3),
            "creator_activity": round(session_state.get("creator_activity", 0.5), 3),
            "engagement_level": round(session_state.get("engagement_level", 0), 3)
        }

        return json.dumps(
            {
                "status": "success",
                "current_day": new_day,
                "events": day_result,
                "current_state": current_state,
                "summary": f"Day {new_day}: DAU={day_result['dau']}, Change={day_result['dau_change']:+d}, Retention Rate={day_result['retention_rate']:.1%}",
            },
            ensure_ascii=False,
            indent=2,
        )

