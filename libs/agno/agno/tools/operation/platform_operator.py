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
from typing import Dict, Any, Optional
import json
import math
import random
from agno.tools.toolkit import Toolkit


class PlatformOperatorTools(Toolkit):
    """Platform operation toolkit for managing user growth, content ecosystem, and community health"""

    def __init__(
        self,
        add_instructions: bool = True,
        platform_dynamics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        # Store platform dynamics config (will be passed to session_state)
        self.platform_dynamics = platform_dynamics or {}
        
        # Initialize random number generator for noise
        noise_config = self.platform_dynamics.get("noise_config", {})
        seed = noise_config.get("seed", None)
        self._rng = random.Random(seed)

        tools = [
            self.acquisition_boost,
            self.engagement_tune,
            self.creator_incentive,
            self.moderation_tighten,
        ]

        super().__init__(
            name="platform_operator_tools",
            tools=tools,
            add_instructions=add_instructions,
            auto_register=True,
            show_result_tools=["acquisition_boost", "engagement_tune", "creator_incentive", "moderation_tighten"],
            **kwargs,
        )
    
    def _apply_action_noise(self, value: float, action_name: str, session_state: Dict[str, Any]) -> float:
        """Apply mixed noise (multiplicative + additive) to action effects
        
        This uses a hybrid approach to ensure noise has meaningful impact on both
        large and small values:
        - Multiplicative noise: value * (1 ± relative_fluctuation)
        - Additive noise: ± absolute_scale
        
        Args:
            value: The base value to add noise to
            action_name: Name of the action (e.g., 'acquisition_campaign', 'creator_incentive')
            session_state: State dictionary containing noise config
            
        Returns:
            Value with noise applied
        """
        dynamics = session_state.get("_platform_dynamics", {})
        noise_config = dynamics.get("noise_config", {})
        
        if not noise_config.get("enabled", False):
            return value
        
        action_noise_config = noise_config.get("action_noise", {})
        fluctuation = action_noise_config.get(action_name, 0.0)
        
        if fluctuation <= 0:
            return value
        
        absolute_noise_config = noise_config.get("action_noise_absolute", {})
        absolute_scale = absolute_noise_config.get(action_name, 0.3)
        
        relative_component = value * self._rng.uniform(-fluctuation, fluctuation)
        
        absolute_component = value * absolute_scale * self._rng.uniform(-1.0, 1.0)
        
        noisy_value = value + relative_component + absolute_component
        
        return max(0.0, noisy_value)

    def acquisition_boost(self, session_state: Dict[str, Any]) -> str:
        """Invest in user acquisition campaigns to rapidly increase DAU.
        
        Returns:
            A JSON string with action results:
            {
                "status": "success",
                "action": "acquisition_boost",
                "new_users_acquired": 130
            }
        
        Note:
            - Effect: Increases DAU immediately, effectiveness scales with content quality
        """
        # Get config parameters
        dynamics = session_state.get("_platform_dynamics", {})
        action_params = dynamics.get("actions", {}).get("acquisition_boost", {})
        base_new_users = action_params.get("base_new_users", 100)
        quality_bonus_rate = action_params.get("quality_bonus_rate", 0.3)
        
        # Calculate new users (base + quality bonus)
        quality = session_state.get("content_quality", 0.5)
        quality_bonus = int(base_new_users * quality * quality_bonus_rate)
        new_users_base = base_new_users + quality_bonus
        
        new_users = int(self._apply_action_noise(new_users_base, "acquisition_campaign", session_state))
        
        history = session_state.setdefault("action_history", [])
        history.append({
            "day": session_state.get("day", 0),
            "action": "acquisition_boost",
            "new_users": new_users
        })
        
        return json.dumps({
            "status": "success",
            "action": "acquisition_boost",
            "new_users_acquired": new_users
        }, ensure_ascii=False, indent=2)

    def engagement_tune(self, session_state: Dict[str, Any]) -> str:
        """Optimize recommendation algorithm to boost user engagement and retention.
        
        Returns:
            A JSON string with action results:
            {
                "status": "success",
                "action": "engagement_tune",
                "engagement_level": 0.20,
                "content_quality": 0.62
            }
        
        Note:
            - Effect: Boosts retention rate short-term, reduces content quality
        """
        # Get config parameters
        dynamics = session_state.get("_platform_dynamics", {})
        action_params = dynamics.get("actions", {}).get("engagement_tune", {})
        engagement_boost_base = action_params.get("engagement_boost", 0.2)
        quality_penalty_base = action_params.get("quality_penalty", 0.08)
        
        engagement_boost = self._apply_action_noise(engagement_boost_base, "engagement_tune", session_state)
        quality_penalty = self._apply_action_noise(quality_penalty_base, "engagement_tune", session_state)
        
        current_engagement_level = session_state.get("engagement_level", 0)
        session_state["engagement_level"] = min(1.0, current_engagement_level + engagement_boost)
        
        quality = session_state.get("content_quality", 0.5)
        session_state["content_quality"] = max(0.0, quality - quality_penalty)
        
        history = session_state.setdefault("action_history", [])
        history.append({
            "day": session_state.get("day", 0),
            "action": "engagement_tune",
            "engagement_boost": engagement_boost,
            "quality_decay": quality_penalty
        })
        
        return json.dumps({
            "status": "success",
            "action": "engagement_tune",
            "engagement_level": session_state["engagement_level"],
            "content_quality": round(session_state["content_quality"], 3)
        }, ensure_ascii=False, indent=2)

    def creator_incentive(self, session_state: Dict[str, Any]) -> str:
        """Invest in creator programs to increase content supply and creator activity.
        
        Returns:
            A JSON string with action results:
            {
                "status": "success",
                "action": "creator_incentive",
                "creator_activity": 0.65,
                "content_added": 32,
                "total_content": 132
            }
        
        Note:
            - Effect: Increases creator activity and content volume immediately
        """
        # Get config parameters
        dynamics = session_state.get("_platform_dynamics", {})
        action_params = dynamics.get("actions", {}).get("creator_incentive", {})
        
        activity_boost_base = action_params.get("activity_boost_base", 0.20)
        diminishing_factor = action_params.get("diminishing_factor", 2.0)
        content_multiplier = action_params.get("content_multiplier", 50)
        
        creator_activity = session_state.get("creator_activity", 0.5)
        activity_boost_base = activity_boost_base * math.pow(1 - creator_activity, diminishing_factor)
        
        activity_boost = self._apply_action_noise(activity_boost_base, "creator_incentive", session_state)
        
        session_state["creator_activity"] = min(1.0, creator_activity + activity_boost)
        
        content_volume = session_state.get("content_volume", 100)
        new_content = int(content_multiplier * session_state["creator_activity"])
        session_state["content_volume"] = content_volume + new_content
        
        history = session_state.setdefault("action_history", [])
        history.append({
            "day": session_state.get("day", 0),
            "action": "creator_incentive",
            "creator_activity_gain": activity_boost,
            "new_content": new_content
        })
        
        return json.dumps({
            "status": "success",
            "action": "creator_incentive",
            "creator_activity": round(session_state["creator_activity"], 3),
            "activity_boost_applied": round(activity_boost, 4),
            "content_added": new_content,
            "total_content": session_state["content_volume"]
        }, ensure_ascii=False, indent=2)

    def moderation_tighten(self, session_state: Dict[str, Any]) -> str:
        """Strengthen content moderation to improve overall content quality.
        
        Returns:
            A JSON string with action results:
            {
                "status": "success",
                "action": "moderation_tighten",
                "content_quality": 0.82,
                "content_removed": 15,
                "remaining_content": 85,
                "creator_activity": 0.55
            }
        
        Note:
            - Effect: Increases quality by removing low-quality content, may reduce creator activity
        """
        # Get config parameters
        dynamics = session_state.get("_platform_dynamics", {})
        action_params = dynamics.get("actions", {}).get("moderation_tighten", {})
        
        quality = session_state.get("content_quality", 0.5)
        
        quality_boost_base = action_params.get("quality_boost_base", 0.25)
        diminishing_factor = action_params.get("diminishing_factor", 2.0)
        
        quality_boost_base_calc = quality_boost_base * math.pow(1 - quality, diminishing_factor)
        
        quality_boost = self._apply_action_noise(quality_boost_base_calc, "moderation_tighten", session_state)
        
        creator_penalty_base = action_params.get("creator_penalty_base", 0.10)
        penalty_amplifier = action_params.get("penalty_amplifier", 1.5)
        
        creator_penalty = creator_penalty_base * (1 + penalty_amplifier * quality)
        
        content_removal_rate = action_params.get("content_removal_rate", 0.15)
        
        session_state["content_quality"] = min(1.0, quality + quality_boost)
        
        content_volume = session_state.get("content_volume", 100)
        removed = int(content_volume * content_removal_rate)
        session_state["content_volume"] = max(0, content_volume - removed)
        
        creator_activity = session_state.get("creator_activity", 0.5)
        session_state["creator_activity"] = max(0.0, creator_activity - creator_penalty)
        
        history = session_state.setdefault("action_history", [])
        history.append({
            "day": session_state.get("day", 0),
            "action": "moderation_tighten",
            "quality_gain": quality_boost,
            "content_removed": removed,
            "creator_penalty": creator_penalty
        })
        
        return json.dumps({
            "status": "success",
            "action": "moderation_tighten",
            "content_quality": round(session_state["content_quality"], 3),
            "quality_boost_applied": round(quality_boost, 4),
            "content_removed": removed,
            "remaining_content": session_state["content_volume"],
            "creator_activity": round(session_state["creator_activity"], 3),
            "creator_penalty_applied": round(creator_penalty, 4)
        }, ensure_ascii=False, indent=2)

    def get_platform_metrics(self, session_state: Dict[str, Any]) -> str:
        """View current platform status and key metrics.
        
        Returns:
            A JSON string with current platform metrics:
            {
                "day": 5,
                "core_metrics": {"DAU": 1200},
                "content_ecosystem": {...},
                "user_behavior": {...},
                "forecast": {
                    "tomorrow_retained_users": 906,
                    "tomorrow_natural_new": 21,
                    "tomorrow_estimated_dau": 927
                }
            }
        
        Note:
            - Effect: Read-only metrics query, no state changes
            - Provides forecast for next day based on current state
        """
        day = session_state.get("day", 0)
        dau = session_state.get("dau", 0)
        content_volume = session_state.get("content_volume", 0)
        content_quality = session_state.get("content_quality", 0.5)
        creator_activity = session_state.get("creator_activity", 0.5)
        engagement_level = session_state.get("engagement_level", 0)
        
        # Calculate retention rate
        retention_rate = _calculate_retention_rate(
            content_volume, 
            content_quality, 
            engagement_level,
            session_state
        )
        
        # Calculate expected DAU for next day (for reference)
        estimated_retained = int(dau * retention_rate)
        natural_new = _calculate_natural_growth(content_quality, creator_activity, session_state)
        
        return json.dumps({
            "day": day,
            "core_metrics": {
                "DAU": dau
            },
            "content_ecosystem": {
                "content_volume": content_volume,
                "content_quality": round(content_quality, 3),
                "creator_activity": round(creator_activity, 3)
            },
            "user_behavior": {
                "engagement_level": round(engagement_level, 3),
                "estimated_retention_rate": f"{retention_rate*100:.1f}%"
            },
            "forecast": {
                "tomorrow_retained_users": estimated_retained,
                "tomorrow_natural_new": natural_new,
                "tomorrow_estimated_dau": estimated_retained + natural_new
            }
        }, ensure_ascii=False, indent=2)


# ==================== System Dynamics Core Functions ====================

def simulate_platform_day(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate one day of platform dynamics evolution
    
    Core Logic:
    1. Calculate retention rate (based on content, quality, algorithm)
    2. Calculate new users (natural growth + quality word-of-mouth)
    3. Calculate content evolution (creator production + natural decay)
    4. Calculate quality evolution (natural decay + algorithm impact)
    5. Update DAU
    
    Args:
        session_state: Platform state dictionary
        
    Returns:
        Dictionary containing detailed changes for the day
    """
    # Get current state
    dau = session_state.get("dau", 1000)
    content_volume = session_state.get("content_volume", 100)
    content_quality = session_state.get("content_quality", 0.5)
    creator_activity = session_state.get("creator_activity", 0.5)
    engagement_level = session_state.get("engagement_level", 0)
    
    dynamics = session_state.get("_platform_dynamics", {})
    
    noise_config = dynamics.get("noise_config", {})
    seed = noise_config.get("seed", None)
    if seed is None:
        current_day = session_state.get("day", 0)
        rng = random.Random(current_day)
    else:
        rng = random.Random(seed)
    
    decay_params = dynamics.get("decay", {})
    ecosystem_params = dynamics.get("content_ecosystem", {})
    
    quality_decay_strength = decay_params.get("quality_decay", 0.05)
    quality_equilibrium = decay_params.get("quality_equilibrium", 0.35)
    creator_decay_strength = decay_params.get("creator_decay", 0.05)
    creator_equilibrium = decay_params.get("creator_equilibrium", 0.30)
    content_decay_rate = decay_params.get("content_decay_rate", 0.05)
    engagement_decay_rate = decay_params.get("engagement_decay", 0.1)
    
    content_creation_multiplier = ecosystem_params.get("content_creation_multiplier", 30)
    content_quality_bonus = ecosystem_params.get("content_quality_bonus", 0.5)
    
    retention_rate = _calculate_retention_rate(content_volume, content_quality, engagement_level, session_state, rng)
    retained_users = int(dau * retention_rate)
    churned_users = dau - retained_users
    
    natural_new_users = _calculate_natural_growth(content_quality, creator_activity, session_state, rng)
    
    new_dau = retained_users + natural_new_users
    session_state["dau"] = new_dau
    
    new_content_created = int(content_creation_multiplier * creator_activity * (1 + content_quality_bonus * content_quality))
    content_decay = int(content_volume * content_decay_rate)
    session_state["content_volume"] = max(10, content_volume + new_content_created - content_decay)
    
    engagement_penalty = engagement_level * 0.03
    quality_decay = (content_quality - quality_equilibrium) * quality_decay_strength
    session_state["content_quality"] = max(0.1, content_quality - quality_decay - engagement_penalty)
    
    session_state["engagement_level"] = max(0.0, engagement_level - engagement_decay_rate)
    
    creator_decay = (creator_activity - creator_equilibrium) * creator_decay_strength
    session_state["creator_activity"] = max(0.1, creator_activity - creator_decay)
    
    history = session_state.setdefault("dau_history", [])
    history.append({
        "day": session_state.get("day", 0),
        "dau": new_dau,
        "retained": retained_users,
        "new": natural_new_users,
        "churned": churned_users,
        "content_volume": session_state["content_volume"],
        "content_quality": round(session_state["content_quality"], 3),
        "retention_rate": round(retention_rate, 3)
    })
    
    return {
        "dau": new_dau,
        "dau_change": new_dau - dau,
        "retained_users": retained_users,
        "new_users": natural_new_users,
        "churned_users": churned_users,
        "retention_rate": round(retention_rate, 3),
        "content_created": new_content_created,
        "content_decayed": content_decay
    }


def _calculate_retention_rate(
    content_volume: int,
    content_quality: float,
    engagement_level: float,
    session_state: Dict[str, Any],
    rng: random.Random
) -> float:
    """Calculate user retention rate
    
    Retention Rate = f(Content Volume, Content Quality, Engagement Level)
    
    - More content → higher retention (users have things to consume)
    - Higher quality → higher retention (users want to stay)
    - Algorithm stimulation temporarily boosts retention, but quality decline backlashes
    
    Args:
        content_volume: Amount of available content
        content_quality: Content quality [0, 1]
        engagement_level: Algorithm stimulation level [0, 1]
        session_state: State dictionary to get config parameters
        
    Returns:
        Retention rate [0, 1]
    """
    # Get config parameters
    dynamics = session_state.get("_platform_dynamics", {})
    retention_params = dynamics.get("retention", {})
    
    base_retention = retention_params.get("base_retention", 0.6)
    content_factor_weight = retention_params.get("content_factor_weight", 0.15)
    quality_factor_weight = retention_params.get("quality_factor_weight", 0.20)
    engagement_factor_weight = retention_params.get("engagement_factor_weight", 0.1)
    max_retention = retention_params.get("max_retention", 0.95)
    min_retention = retention_params.get("min_retention", 0.3)
    
    content_factor = content_factor_weight * math.log(max(10, content_volume) / 10) / math.log(10)
    
    quality_factor = quality_factor_weight * content_quality
    
    engagement_factor = engagement_factor_weight * engagement_level
    
    quality_penalty = 0 if content_quality > 0.3 else (0.3 - content_quality) * 0.5
    
    retention = base_retention + content_factor + quality_factor + engagement_factor - quality_penalty
    
    dynamics = session_state.get("_platform_dynamics", {})
    noise_config = dynamics.get("noise_config", {})
    if noise_config.get("enabled", False):
        retention_noise_config = noise_config.get("retention_noise", {})
        fluctuation = retention_noise_config.get("fluctuation", 0.0)
        if fluctuation > 0:
            noise = rng.uniform(-fluctuation, fluctuation)
            retention += noise
    
    return max(min_retention, min(max_retention, retention))


def _calculate_natural_growth(
    content_quality: float,
    creator_activity: float,
    session_state: Dict[str, Any],
    rng: random.Random
) -> int:
    """Calculate natural new user growth (word-of-mouth)
    
    High-quality content + Active creators = Natural growth
    
    Args:
        content_quality: Content quality [0, 1]
        creator_activity: Creator activity level [0, 1]
        session_state: State dictionary to get config parameters
        
    Returns:
        Number of new users
    """
    # Get config parameters
    dynamics = session_state.get("_platform_dynamics", {})
    growth_params = dynamics.get("natural_growth", {})
    
    base_growth = growth_params.get("base_growth", 10)
    quality_multiplier = growth_params.get("quality_multiplier", 30)
    creator_multiplier = growth_params.get("creator_multiplier", 20)
    
    quality_bonus = int(quality_multiplier * content_quality)
    
    creator_bonus = int(creator_multiplier * creator_activity)
    
    natural_growth = base_growth + quality_bonus + creator_bonus
    
    dynamics = session_state.get("_platform_dynamics", {})
    noise_config = dynamics.get("noise_config", {})
    if noise_config.get("enabled", False):
        growth_noise_config = noise_config.get("growth_noise", {})
        fluctuation = growth_noise_config.get("fluctuation", 0.0)
        if fluctuation > 0:
            noise_multiplier = 1.0 + rng.uniform(-fluctuation, fluctuation)
            natural_growth = int(natural_growth * noise_multiplier)
    
    return max(0, natural_growth)

