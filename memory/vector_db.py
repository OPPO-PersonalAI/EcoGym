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
import uuid
import logging
from typing import List, Any, Dict
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from memory.user_memory import MemoryItem

class VectorMem:
    def __init__(self, 
        collection_name: str = "vector_memories", 
        persist_directory: str = "./chromadb_memory",
        similarity_threshold: float = 0.3,
        embedding_config: Dict[str, str] = None, 
        retrieve_top_k: int = 3,
        **kwargs
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.similarity_threshold = similarity_threshold
        self.retrieve_top_k = retrieve_top_k
        
        self.ef = None
        if embedding_config and embedding_config.get("provider") == "openai":
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_URL") or os.getenv("API_URL")
            
            if not api_key:
                print("[VectorDB] ⚠️ API_KEY not found in env. OpenAI embedding will fail.")
            
            self.ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=base_url,
                model_name=embedding_config.get("model", "text-embedding-3-small")
            )
            print(f"[VectorDB Debug] Setup OpenAI Embedding: {embedding_config.get('model')}")
        else:
            print("[VectorDB] ⚠️ Using default Chroma embedding.")

        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory, settings=Settings(anonymized_telemetry=False))
            
            self.collection = self.client.get_or_create_collection(name=self.collection_name, embedding_function=self.ef)
            
            count = self.collection.count()
            print(f"[VectorDB Debug] Init Complete. DB Path: {self.persist_directory}, Count: {count}")
            
        except Exception as e:
            print(f"[VectorDB Debug] !!! Init CRASHED: {e}")
            raise e

    def add(self, step_index: int, items: List[MemoryItem]) -> None:
        documents = []
        metadatas = []
        ids = []
        
        for item in items:
            if not item.content or len(str(item.content)) < 4: 
                continue
            
            if item.role == "system": continue

            doc_id = f"{step_index}_{uuid.uuid4().hex[:8]}"
            documents.append(str(item.content))
            metadatas.append({
                "role": item.role,
                "step": step_index,
                "timestamp": item.created_at
            })
            ids.append(doc_id)
            
        if documents:
            try:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                print(f"[VectorDB] ✅ Saved {len(documents)} items at step {step_index}.")
            except Exception as e:
                print(f"[VectorDB] Add error: {e}")

    def search(self, query: str, limit: int = None) -> List[MemoryItem]:
        if not query: 
            print("[VectorDB Debug] Search skipped: Query is empty.")
            return []
        
        final_limit = limit if limit is not None else self.retrieve_top_k

        try:
            results = self.collection.query(query_texts=[query], n_results=final_limit)
        except Exception as e:
            print(f"[VectorDB] Query failed: {e}")
            return []

        items = []
        if results['documents']:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for i, doc in enumerate(docs):
                score = 1.0 - distances[i] 
                
                print(f"[VectorDB Debug] Search Score: {score:.3f} | Text: {doc[:20]}...")

                if score < self.similarity_threshold:
                    continue
                    
                meta = metas[i] if i < len(metas) else {}
                items.append(MemoryItem(
                    role=meta.get("role", "system"),
                    content=doc,
                    created_at=meta.get("step", 0),
                    relevance_score=score,
                    source_module="vector_db_history"
                ))
        else:
            print("[VectorDB Debug] No documents found in raw query.")
        
        return items
    
    def clear(self):
        try:
            self.client.delete_collection(self.collection_name)
            print("[VectorDB] Collection cleared.")
        except Exception as e:
            print(f"[VectorDB] Clear failed: {e}")