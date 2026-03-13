# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 OPPO. All rights reserved.
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
from typing import Dict, Any, Optional

def operation_bench_is_finished(
    state: Dict[str, Any],
    max_days: Optional[int] = None,
    min_dau_threshold: int = 100,
) -> bool:
    """
    Determine whether the operation simulation should terminate
    
    Termination conditions:
    1. Reached maximum days limit
    2. DAU falls below minimum threshold (platform collapse)
    
    Args:
        state: Current state dictionary
        max_days: Maximum days limit, None means unlimited
        min_dau_threshold: Minimum DAU threshold, below this value indicates platform failure
    
    Returns:
        True means should terminate, False means continue running
    """
    # 1) Absolute days limit
    day = state.get("day", 0)
    if max_days is not None and day >= max_days:
        print(f"[Termination Condition Met] Reached maximum days limit: day={day}, max_days={max_days}")
        return True
    
    # 2) DAU collapse detection
    dau = state.get("dau", 0)
    if dau < min_dau_threshold:
        print(f"[Termination Condition Met] DAU below minimum threshold: dau={dau}, threshold={min_dau_threshold}")
        return True
    
    return False


def operation_bench_cal_metric(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate and print key metrics
    
    Args:
        state: Current state dictionary
    
    Returns:
        Dictionary containing calculated metrics
    """
    day = state.get("day", 0)
    dau = state.get("dau", 0)
    content_volume = state.get("content_volume", 0)
    content_quality = state.get("content_quality", 0.5)
    creator_activity = state.get("creator_activity", 0.5)
    engagement_level = state.get("engagement_level", 0.0)
    
    # Calculate historical statistics
    dau_history = state.get("dau_history", [])
    if dau_history:
        avg_dau = sum(h.get("dau", 0) for h in dau_history) / len(dau_history)
        max_dau = max(h.get("dau", 0) for h in dau_history)
        avg_retention = sum(h.get("retention_rate", 0) for h in dau_history) / len(dau_history)
    else:
        avg_dau = dau
        max_dau = dau
        avg_retention = 0.0
    
    print(f"[Metrics] Day {day} | DAU: {dau} | "
          f"Content: {content_volume} | Quality: {content_quality:.2f} | "
          f"Creator Activity: {creator_activity:.2f} | Engagement: {engagement_level:.2f}")
    print(f"[History] Avg DAU: {avg_dau:.0f} | Peak DAU: {max_dau} | "
          f"Avg Retention: {avg_retention:.1%}")
    
    return {
        "final_day": day,
        "final_dau": dau,
        "avg_dau": round(avg_dau, 2),
        "max_dau": max_dau,
        "avg_retention": round(avg_retention, 4),
        "final_content_volume": content_volume,
        "final_content_quality": round(content_quality, 3),
        "final_creator_activity": round(creator_activity, 3),
        "final_engagement_level": round(engagement_level, 3),
    }
