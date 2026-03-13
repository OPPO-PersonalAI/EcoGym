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

import json
import logging
import re
from textwrap import dedent
from typing import Dict, List, Any, Optional, Set, Tuple
from tenacity import retry, stop_after_attempt, wait_fixed

from memory.user_memory import MemoryItem

STOP_WORDS = {
    'the', 'is', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 
    'by', 'value', 'variable', 'set', 'get', 'check', 'what', 'how', 'status',
    'current', 'updated'
}

class ScratchPad:
    def __init__(self, llm_client: Any, model_name: str, max_vars: int = 50, retrieve_top_k: int = 10, **kwargs):
        self.client = llm_client
        self.model_name = model_name
        self.storage: Dict[str, str] = {}
        self._update_order: List[str] = []
        
        self.max_vars = max_vars
        self.retrieve_top_k = retrieve_top_k
        self.logger = logging.getLogger("ScratchPad")
        
        self.extract_prompt = dedent("""
            You are a Memory Manager. Your job is to extract **State Variables**, **Plans**, **Goals**, or **Key Facts** from the provided text into a JSON object.
            Rules:
            1. Extract clearly defined variables (e.g., "Set price to 5" -> {"price": "5"}).
            2. Extract abstract goals (e.g., "Aiming for 100 DAU" -> {"dau_goal": "100"}).
            3. Extract boolean states.
            4. Ignore general chit-chat.
            5. Return ONLY a JSON object.
        """)

    def add(self, step_index: int, items: List[MemoryItem]) -> None:
        if not self.client: return

        text_batch = []
        for item in items:
            if item.content and len(str(item.content)) > 10:
                prefix = "Agent thought:" if item.role == "assistant" else "User input:"
                text_batch.append(f"{prefix} {item.content}")
        
        if not text_batch: return
        full_text = "\n".join(text_batch)
        
        try:
            new_vars = self._extract_with_retry(full_text)
            if new_vars:
                for k, v in new_vars.items():
                    if k in self.storage:
                        if k in self._update_order:
                            self._update_order.remove(k)
                    
                    self.storage[k] = str(v)
                    self._update_order.append(k)

                self.logger.info(f"[Memory] 🧠 ScratchPad extracted: {new_vars}")
                
                while len(self._update_order) > self.max_vars:
                    oldest_key = self._update_order.pop(0)
                    if oldest_key in self.storage:
                        del self.storage[oldest_key]
                        
        except Exception as e:
            self.logger.warning(f"[Memory] ScratchPad extraction failed: {e}")

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
    def _extract_with_retry(self, text: str) -> Dict[str, str]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.extract_prompt},
                {"role": "user", "content": f"Extract from this:\n{text}"}
            ],
        )
        content = response.choices[0].message.content
        return json.loads(content) if content else {}

    def _tokenize(self, text: str) -> Set[str]:
        if not text: return set()
        clean_text = text.replace('_', ' ').replace('-', ' ').lower()
        tokens = set(clean_text.split())
        return {t for t in tokens if t not in STOP_WORDS and len(t) > 1}

    def search(self, query: str, limit: int = None) -> List[MemoryItem]:
        if not self.storage: return []
        
        final_limit = limit if limit is not None else self.retrieve_top_k
        hits: List[Tuple[float, str, str]] = []
        
        query_tokens = self._tokenize(query)
        use_similarity = len(query_tokens) > 0
        
        if use_similarity:
            for key, value in self.storage.items():
                kv_text = f"{key} {value}"
                kv_tokens = self._tokenize(kv_text)
                
                if not kv_tokens: continue
                
                intersection = query_tokens.intersection(kv_tokens)
                union = query_tokens.union(kv_tokens)
                
                base_score = len(intersection) / len(union) if union else 0.0
                
                for qt in query_tokens:
                    if qt in key.lower():
                        base_score += 0.5
                
                if base_score > 0.05: hits.append((base_score, key, value))
            
            hits.sort(key=lambda x: x[0], reverse=True)
            
        if not hits:
            recent_keys = self._update_order[-final_limit:][::-1]
            for k in recent_keys:
                hits.append((0.1, k, self.storage[k]))
        
        final_hits_data = hits[:final_limit]
        
        result_items = []
        for score, k, v in final_hits_data:
            result_items.append(MemoryItem(
                role="system",
                content=f"{k} = {v}",
                source_module="scratch_pad",
                relevance_score=score + 1.0
            ))
            
        return result_items

    def clear(self):
        self.storage.clear()
        self._update_order.clear()