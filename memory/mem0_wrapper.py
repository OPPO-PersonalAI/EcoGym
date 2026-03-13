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

import os
from typing import List, Dict, Any
from mem0 import Memory
from memory.user_memory import MemoryItem

class Mem0Wrapper:
    def __init__(self, user_id: str = "default_user", model_name: str = "gpt-4o", retrieve_top_k: int = 5, **kwargs):
        self.user_id = user_id
        self.retrieve_top_k = retrieve_top_k
        
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": kwargs.get("collection_name", "mem0_collection"),
                    "path": kwargs.get("vector_store_path", "./chromadb_mem0")
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": model_name,
                    "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"),
                    "openai_base_url": os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_URL") or os.getenv("API_URL")
                }
            },
             "embedder": {
                "provider": "openai",
                "config": {
                    "model": kwargs.get("embedding_model", "text-embedding-3-small"),
                    "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"),
                    "openai_base_url": os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_URL") or os.getenv("API_URL")
                }
            }
        }
        
        try:
            self.client = Memory.from_config(config)
        except Exception as e:
            print(f"[Mem0Wrapper] Init failed: {e}")
            self.client = None

    def add(self, step_index: int, items: List[MemoryItem]) -> None:
        if not self.client: return
        
        mem0_msgs = []
        for item in items:
            mem0_msgs.append({
                "role": item.role, 
                "content": str(item.content)
            })
            
        try:
            self.client.add(mem0_msgs, user_id=self.user_id)
        except Exception as e:
            print(f"[Mem0Wrapper] Add failed: {e}")

    def search(self, query: str, limit: int = 5) -> List[MemoryItem]:
        if not self.client or not query: return []
        
        final_limit = limit if limit is not None else self.retrieve_top_k
        
        try:
            results = self.client.search(query, user_id=self.user_id, limit=final_limit)
            
            hits = results.get("results", []) if isinstance(results, dict) else results
            
            items = []
            for hit in hits:
                content = hit.get("memory", "")
                score = hit.get("score", 0.5)
                if content:
                    items.append(MemoryItem(
                        role="system",
                        content=f"Fact: {content}",
                        source_module="mem0_facts",
                        relevance_score=score
                    ))
            return items
        except Exception as e:
            print(f"[Mem0Wrapper] Search failed: {e}")
            return []
            
    def clear(self):
        if self.client:
            self.client.reset()