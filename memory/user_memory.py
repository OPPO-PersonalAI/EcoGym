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

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

@dataclass
class MemoryItem:
    role: str = "user"
    content: str = ""
    
    created_at: int = 0
    ttl: Optional[int] = None
    
    relevance_score: float = 0.0
    source_module: str = "unknown"
    
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        prefix = f"[{self.role.upper()}]"
        text = str(self.content)
        if self.tool_calls:
            text += f"\n[Tools]: {json.dumps(self.tool_calls, ensure_ascii=False)}"
        return f"{prefix} {text}"

def messages2items(messages, step_index: int) -> List[MemoryItem]:
    items = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            tool_calls = msg.get('tool_calls', None)
        else:
            role = getattr(msg, 'role', 'user')
            content = getattr(msg, 'content', '')
            tool_calls = getattr(msg, 'tool_calls', None)

        if content is None: content = ""
            
        items.append(MemoryItem(
            role=role,
            content=str(content),
            tool_calls=tool_calls,
            created_at=step_index
        ))
    return items