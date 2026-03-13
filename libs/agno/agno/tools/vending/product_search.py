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
import json
import numpy as np
import os
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle


class ProductDatabase:
    """Offline product database with semantic search capabilities."""
    
    def __init__(
        self, 
        jsonl_path: str = "data/vending/products.jsonl",
        use_embeddings: bool = True,
        embedding_model: str = "text-embedding-3-small",
        cache_embeddings: bool = True,
        model_pricing_config_path: Optional[str] = None
    ):
        """Initialize product database.
        
        Args:
            jsonl_path: Path to products JSONL file
            use_embeddings: Whether to use embedding-based search (vs simple text match)
            embedding_model: OpenAI embedding model to use
            cache_embeddings: Whether to cache embeddings to disk
            model_pricing_config_path: Optional path to model pricing config (for supplier routing)
        """
        self.jsonl_path = Path(jsonl_path)
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
        self.cache_embeddings = cache_embeddings
        self.model_pricing_config_path = model_pricing_config_path
        
        self.products: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_client = None
        
        self._load_products()
        
        if use_embeddings:
            self._initialize_embeddings()
    
    def _load_products(self):
        """Load products from JSONL file."""
        if not self.jsonl_path.exists():
            print(f"Warning: Product database not found at {self.jsonl_path}")
            print("Please run: python scripts/generate_product_database.py")
            self.products = []
            return
        
        self.products = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        product = json.loads(line)
                        self.products.append(product)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(self.products)} products from {self.jsonl_path}")
    
    def _get_embedding_supplier(self) -> str:
        """Get the supplier for the embedding model from pricing config."""
        if not self.model_pricing_config_path:
            return "mtu"
        
        try:
            config_path = Path(self.model_pricing_config_path)
            if not config_path.exists():
                print(f"Warning: Model pricing config not found at {config_path}, using default supplier 'mtu'")
                return "mtu"
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            token_pricing = config.get("token_pricing", {})
            if self.embedding_model in token_pricing:
                supplier = token_pricing[self.embedding_model].get("supplier", "mtu")
                print(f"Using supplier '{supplier}' for embedding model '{self.embedding_model}'")
                return supplier
            else:
                print(f"Warning: Embedding model '{self.embedding_model}' not found in pricing config, using default supplier 'mtu'")
                return "mtu"
        except Exception as e:
            print(f"Warning: Failed to load model pricing config: {e}, using default supplier 'mtu'")
            return "mtu"
    
    def _get_api_credentials(self, supplier: str) -> Dict[str, Optional[str]]:
        """Get API credentials for specified supplier."""
        supplier_upper = supplier.upper()
        
        api_key = os.getenv(f"{supplier_upper}_API_KEY")
        api_url = os.getenv(f"{supplier_upper}_API_URL")
        
        if not api_key or not api_url:
            print(f"Supplier-specific credentials ({supplier_upper}_API_KEY, {supplier_upper}_BASE_URL) not found, "
                  f"falling back to OPENAI_API_KEY and OPENAI_BASE_URL")
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            api_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_URL") or os.getenv("API_URL")
        
        return {"api_key": api_key, "base_url": api_url}
    
    def _get_embedding_cache_path(self) -> Path:
        """Get path for embedding cache file."""
        return self.jsonl_path.parent / f"{self.jsonl_path.stem}_embeddings.pkl"
    
    def _initialize_embeddings(self):
        """Initialize or load cached embeddings."""
        cache_path = self._get_embedding_cache_path()
        
        if self.cache_embeddings and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    if len(cache_data['embeddings']) == len(self.products):
                        self.embeddings = cache_data['embeddings']
                        print(f"Loaded embeddings from cache: {cache_path}")
                        return
            except Exception as e:
                print(f"Warning: Failed to load embedding cache: {e}")
        
        print("Computing embeddings for products...")
        self._compute_embeddings()
        
        if self.cache_embeddings:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({'embeddings': self.embeddings}, f)
                print(f"Saved embeddings to cache: {cache_path}")
            except Exception as e:
                print(f"Warning: Failed to save embedding cache: {e}")
    
    def _compute_embeddings(self):
        """Compute embeddings for all products."""
        if not self.products:
            self.embeddings = np.array([])
            return
        
        try:
            from openai import OpenAI
            
            supplier = self._get_embedding_supplier()
            credentials = self._get_api_credentials(supplier)
            
            api_key = credentials.get("api_key")
            api_url = credentials.get("base_url")
            
            if not api_key:
                print(f"Error: No API key found for supplier '{supplier}'. "
                      f"Please set {supplier.upper()}_API_KEY or API_KEY/OPENAI_API_KEY environment variable.")
                self.use_embeddings = False
                return
            
            client_kwargs = {"api_key": api_key}
            if api_url:
                client_kwargs["base_url"] = api_url
                print(f"Using API URL: {api_url}")
            
            self.embedding_client = OpenAI(**client_kwargs)
        except ImportError:
            print("Error: openai package not installed. Install with: pip install openai")
            self.use_embeddings = False
            return
        
        texts = []
        for p in self.products:
            text = f"{p.get('name', '')} {p.get('category', '')}"
            texts.append(text)
        
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.embedding_client.embeddings.create(
                    input=batch,
                    model=self.embedding_model
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error computing embeddings for batch {i}: {e}")
                all_embeddings.extend([[0.0] * 1536] * len(batch))
        
        self.embeddings = np.array(all_embeddings)
        print(f"Computed {len(self.embeddings)} embeddings")
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a search query."""
        if not self.embedding_client:
            try:
                from openai import OpenAI
                
                # Get supplier and credentials based on model pricing config
                supplier = self._get_embedding_supplier()
                credentials = self._get_api_credentials(supplier)
                
                api_key = credentials.get("api_key")
                api_url = credentials.get("base_url")
                
                if not api_key:
                    print(f"Error: No API key found for supplier '{supplier}'. "
                          f"Please set {supplier.upper()}_API_KEY or API_KEY/OPENAI_API_KEY environment variable.")
                    return np.array([])
                
                client_kwargs = {"api_key": api_key}
                if api_url:
                    client_kwargs["base_url"] = api_url
                
                self.embedding_client = OpenAI(**client_kwargs)
            except ImportError:
                return np.array([])
        
        try:
            response = self.embedding_client.embeddings.create(
                input=[query],
                model=self.embedding_model
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            return np.array([])
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for products matching the query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            category_filter: Optional category to filter by
            
        Returns:
            List of product dictionaries matching the query
        """
        if not self.products:
            return []
        
        candidates = self.products
        if category_filter:
            candidates = [p for p in self.products if p.get('category') == category_filter]
        
        if not candidates:
            return []
        
        if self.use_embeddings and self.embeddings is not None and len(self.embeddings) > 0:
            return self._embedding_search(query, candidates, top_k)
        
        return self._keyword_search(query, candidates, top_k)
    
    def _embedding_search(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform embedding-based semantic search."""
        query_emb = self._get_query_embedding(query)
        if len(query_emb) == 0:
            return self._keyword_search(query, candidates, top_k)
        
        candidate_indices = [self.products.index(c) for c in candidates]
        candidate_embeddings = self.embeddings[candidate_indices]
        
        query_emb = query_emb / np.linalg.norm(query_emb)
        candidate_embeddings = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        similarities = candidate_embeddings @ query_emb
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [candidates[i] for i in top_indices]
        
        return results
    
    def _keyword_search(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback keyword-based search."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored = []
        for product in candidates:
            name = product.get('name', '').lower()
            category = product.get('category', '').lower()
            text = f"{name} {category}"
            
            score = sum(1 for word in query_words if word in text)
            
            if score > 0:
                scored.append((score, product))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [p for _, p in scored[:top_k]]
    
    def get_all_categories(self) -> List[str]:
        """Get list of all product categories."""
        categories = set()
        for p in self.products:
            cat = p.get('category')
            if cat:
                categories.add(cat)
        return sorted(categories)
    
    def get_products_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all products in a specific category."""
        return [p for p in self.products if p.get('category') == category]
    
    def reload(self):
        """Reload products from disk."""
        self._load_products()
        if self.use_embeddings:
            self._initialize_embeddings()

