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

from collections import deque
from typing import List, Set
from memory.user_memory import MemoryItem

STOP_WORDS = {
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 
    'of', 'to', 'in', 'for', 'with', 'by', 'from', 'up', 'about', 
    'into', 'over', 'after', 'user', 'assistant', 'system', 'message'
}

class RollingWindow:
    def __init__(self, 
        window_size: int = 20, 
        score_threshold: float = 0.15, 
        retrieve_top_k: int = 5,
        **kwargs
    ):
        self.max_size = window_size
        self.threshold = score_threshold
        self.retrieve_top_k = retrieve_top_k
        self.buffer = deque(maxlen=window_size)
    
    def add(self, step_index: int, items: List[MemoryItem]) -> None:
        for item in items:
            item.source_module = "rolling_window"
            self.buffer.append(item)
    
    def _tokenize(self, text: str) -> Set[str]:
        if not text: return set()
        tokens = set(text.lower().split())
        return {t for t in tokens if t not in STOP_WORDS and len(t) > 1}

    def search(self, query: str, limit: int = None) -> List[MemoryItem]:
        if not self.buffer: return []
        
        final_limit = limit if limit is not None else self.retrieve_top_k
        
        immediate_limit = 2
        all_items = list(self.buffer)
        results = []
        
        if len(all_items) > 0:
            recents = all_items[-immediate_limit:]
            for item in recents:
                item.relevance_score = 1.0 
                results.append(item)
        
        history_pool = all_items[:-immediate_limit]
        if not history_pool or not query:
            return results

        query_tokens = self._tokenize(query)
        if not query_tokens: return results
            
        relevant_history = []
        for item in history_pool:
            content_tokens = self._tokenize(str(item.content))
            if not content_tokens: continue
            
            intersection = query_tokens.intersection(content_tokens)
            
            union = query_tokens.union(content_tokens)
            score = len(intersection) / len(union) if union else 0.0
            
            if score >= self.threshold:
                item.relevance_score = score
                relevant_history.append(item)
        
        relevant_history.sort(key=lambda x: x.relevance_score, reverse=True)
        
        final_results = relevant_history[:final_limit] + results
        
        return final_results

    def clear(self):
        self.buffer.clear()