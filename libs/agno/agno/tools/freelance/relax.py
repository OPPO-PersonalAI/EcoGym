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
from typing import Dict, Any, Optional
from agno.tools.toolkit import Toolkit

class RelaxTools(Toolkit):
    """
    Toolkit for managing player well-being through paid relaxation activities in Freelance-Bench.
    Allows the agent to trade Money for Energy recovery and Stress reduction.
    
    This toolkit helps prevent game-over scenarios caused by 0 Energy or Max Stress.
    It offers tiered options (Low, Medium, High) to suit different budget constraints.
    """
    def __init__(self, config_path: str = "./freelance_bench_config.yaml", add_instructions: bool = True, **kwargs: Any):
        self.config = self._load_config(config_path)
        self.relax_config = self.config.get("relaxation_config", {})
        
        self.tiers = {
            "low": {
                "cost": 15.0, 
                "energy_gain": 20, 
                "stress_drop": 15,
                "desc": "Low cost relaxation (e.g., Coffee break, Short nap)"
            },
            "medium": {
                "cost": 60.0, 
                "energy_gain": 55, 
                "stress_drop": 40,
                "desc": "Medium cost relaxation (e.g., Movie, Nice meal)"
            },
            "high": {
                "cost": 150.0, 
                "energy_gain": 90, 
                "stress_drop": 80,
                "desc": "High cost relaxation (e.g., Spa day, Short trip)"
            }
        }
        
        if "tiers" in self.relax_config:
            self.tiers.update(self.relax_config["tiers"])

        super().__init__(name="relax_tools", tools=[self.energy_restore], add_instructions=add_instructions, auto_register=True, **kwargs)

    def energy_restore(self, session_state: Dict[str, Any], level: str) -> str:
        """Perform relaxation activity to recover energy and reduce stress.
        
        Args:
            level: Relaxation tier determining cost and recovery amount
                - "low": Cost $15, recovers +20 energy, reduces -15 stress
                - "medium": Cost $60, recovers +55 energy, reduces -40 stress
                - "high": Cost $150, recovers +90 energy, reduces -80 stress
        
        Returns:
            A JSON string with transaction results:
            {
                "status": "success",
                "action": "relax_low",
                "description": "...",
                "changes": {...},
                "current_state": {
                    "money": 985.0,
                    "energy": 80,
                    "stress": 10
                }
            }
        
        Note:
            - Effect: Trades money for energy recovery and stress reduction
            - Recovery amounts capped by max/min bounds (0-100)
        """
        level_key = level.lower().strip()
        
        if level_key not in self.tiers:
            return json.dumps({
                "status": "error", 
                "message": f"Invalid relaxation level '{level}'. Valid options: {list(self.tiers.keys())}"
            }, ensure_ascii=False)

        tier = self.tiers[level_key]
        cost = tier["cost"]
        e_gain = tier["energy_gain"]
        s_drop = tier["stress_drop"]

        current_money = float(session_state.get("money", 1000.0))
        current_energy = int(session_state.get("energy", 80))
        current_stress = int(session_state.get("stress", 10))

        if current_money < cost:
            return json.dumps({
                "status": "failed",
                "reason": "insufficient_funds",
                "cost_required": cost,
                "current_money": current_money,
                "message": f"You cannot afford {level} relaxation."
            }, ensure_ascii=False)

        new_money = round(current_money - cost, 2)

        new_energy = min(100, current_energy + e_gain)
        actual_e_gain = new_energy - current_energy
        
        new_stress = max(0, current_stress - s_drop)
        actual_s_drop = current_stress - new_stress

        session_state["money"] = new_money
        session_state["energy"] = new_energy
        session_state["stress"] = new_stress

        return json.dumps({
            "status": "success",
            "action": f"relax_{level_key}",
            "description": tier["desc"],
            "changes": {
                "money_change": -cost,
                "energy_change": f"+{actual_e_gain}",
                "stress_change": f"-{actual_s_drop}"
            },
            "current_state": {
                "money": new_money,
                "energy": new_energy,
                "stress": new_stress
            },
            "message": f"Relaxation complete. Recovered {actual_e_gain} Energy and reduced {actual_s_drop} Stress."
        }, ensure_ascii=False)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path): return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[Warn] RelaxTools failed to load config: {e}")
            return {}