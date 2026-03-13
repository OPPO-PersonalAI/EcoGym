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
from typing import List, Dict, Any, Optional
from openai import OpenAI

from memory.user_memory import MemoryItem, messages2items
from memory.rolling_window import RollingWindow
from memory.scratch_pad import ScratchPad
from memory.vector_db import VectorMem
from memory.mem0_wrapper import Mem0Wrapper

class MemoryManager:
    def __init__(self, model_name: str, memory_config: Dict[str, Any], llm_client: Optional[OpenAI] = None):
        self.config = memory_config
        self.use_memory = memory_config.get('use_memory', False)
        self.user_id = memory_config.get('user_id', 'default_user')
        self.model_name = model_name
        
        if llm_client:
            self.client = llm_client
        else:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_URL") or os.getenv("API_URL")
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = None
                if self.use_memory:
                    print("[Memory] Warning: No OpenAI Client available (api_key missing?), LLM-based memory features will fail.")

        general_conf = memory_config.get('general_config', {})
        self.embedding_config = general_conf.get('embedding_config', {})
        self.total_limit = general_conf.get('total_prompt_limit', 15)
        
        self.modules = {}
        if not self.use_memory: return

        module_configs = memory_config.get('modules', {})
        
        if module_configs.get('rolling_window', {}).get('enabled', False):
            self.modules['rolling_window'] = RollingWindow(**module_configs['rolling_window'])
        
        if module_configs.get('scratch_pad', {}).get('enabled', False):
            self.modules['scratch_pad'] = ScratchPad(llm_client=self.client, model_name=self.model_name, **module_configs['scratch_pad'])
        
        if module_configs.get('vector_db', {}).get('enabled', False):
            self.modules['vector_db'] = VectorMem(embedding_config=self.embedding_config, **module_configs['vector_db'])
        
        if module_configs.get('mem0', {}).get('enabled', False):
            mem0_conf = module_configs['mem0']
            if 'embedding_model' not in mem0_conf:
                mem0_conf['embedding_model'] = self.embedding_config.get('model', 'text-embedding-3-small')
            
            self.modules['mem0'] = Mem0Wrapper(user_id=self.user_id, model_name=model_name, **mem0_conf)

        print(f"[UnifiedMemory] Initialized modules: {list(self.modules.keys())}")

    def add(self, messages: List[Dict[str, Any]], step_index: int = 0) -> None:
        if not self.use_memory: return
        items = messages2items(messages, step_index)
        for name, module in self.modules.items():
            try:
                module.add(step_index, items)
            except Exception as e:
                print(f"[UnifiedMemory] Error adding to {name}: {e}")

    def retrieve(self, query: str) -> str:
        if not self.use_memory or not query: return ""

        all_hits: List[MemoryItem] = []
        
        for name, module in self.modules.items():
            try:
                hits = module.search(query)
                all_hits.extend(hits)
            except Exception as e:
                print(f"[UnifiedMemory] Error searching {name}: {e}")

        if not all_hits: return ""

        all_hits.sort(key=lambda x: x.relevance_score, reverse=True)
        
        seen_content = set()
        unique_hits = []
        for item in all_hits:
            content_sig = str(item.content).strip().lower()
            if content_sig not in seen_content:
                seen_content.add(content_sig)
                unique_hits.append(item)
        
        final_hits = unique_hits[:self.total_limit]
        
        sections = {
            "Facts & Knowledge": [], 
            "Current Context": [], 
            "Variables": []
        }
        
        for item in final_hits:
            text = f"- {item.content}"
            if item.source_module in ['mem0_facts', 'vector_db_history']:
                sections["Facts & Knowledge"].append(text)
            elif item.source_module == 'scratch_pad':
                sections["Variables"].append(text)
            else:
                sections["Current Context"].append(f"{item.role}: {item.content}")

        prompt_parts = []
        if sections["Variables"]:
            prompt_parts.append("### Detected Variables\n" + "\n".join(sections["Variables"]))
        if sections["Facts & Knowledge"]:
            prompt_parts.append("### Relevant Memories\n" + "\n".join(sections["Facts & Knowledge"]))
        if sections["Current Context"]:
            prompt_parts.append("### Relevant History\n" + "\n".join(sections["Current Context"]))
            
        return "\n\n".join(prompt_parts)
    
    def clear(self):
        for module in self.modules.values():
            if hasattr(module, 'clear'): module.clear()