#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 OPPO AI Agent Team. All rights reserved.
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
import math
import random
import numpy as np
import statistics
from typing import Dict, Any, Tuple

def set_seed(seed: int = 42):
    if seed is None:
        return
        
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"[Seed] Global seed set to: {seed}")

def freelance_is_finished(state: Dict[str, Any], max_days: int = 100) -> bool:
    money = state.get("money", 0)
    energy = state.get("energy", 0)
    stress = state.get("stress", 0)
    day = state.get("day", 0)

    if money <= 0: return True
    if energy <= 0: return True
    if stress >= 100: return True
    if day >= max_days: return True
        
    return False

def freelance_check_termination_reason(state: Dict[str, Any], max_days: int = 100) -> Tuple[str, float]:
    money = state.get("money", 0)
    energy = state.get("energy", 0)
    stress = state.get("stress", 0)
    day = state.get("day", 0)

    if money <= 0: return "Bankruptcy (Money <= 0)", 0.9
    if energy <= 0: return "Exhaustion (Energy <= 0)", 0.9
    if stress >= 100: return "Burnout (Stress >= 100)", 0.9
    if day >= max_days: return "Career Completed (Time limit reached)", 1.0
    
    return "Running", 1.0

def freelance_cal_metric(state: Dict[str, Any]) -> Dict[str, float]:
    reason, penalty = freelance_check_termination_reason(state)
    
    money = max(0, state.get("money", 0))
    energy = max(0, state.get("energy", 0))
    stress = min(100, max(0, state.get("stress", 0)))
    day = state.get("day", 0)
    
    skills = state.get("skill_rating", {})
    if isinstance(skills, dict):
        skill_values = list(skills.values())
        avg_skill = statistics.mean(skill_values) if skill_values else 60.0
    elif isinstance(skills, list):
        avg_skill = statistics.mean(skills) if skills else 60.0
    else:
        avg_skill = 60.0

    return {
        "final_day": day,
        "termination_reason": reason,
        "penalty_coefficient": penalty,
        "final_money": round(money, 2),
        "final_energy": round(energy, 2),
        "final_stress": round(stress, 2),
        "avg_skill": round(avg_skill, 2),
    }