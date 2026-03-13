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
from typing import List, Dict, Optional, Any
import time
import json
from agno.tools.toolkit import Toolkit
from agno.tools.vending.product_search import ProductDatabase

class SupplierCommunicationTools(Toolkit):
    """
    Simplified Toolkit for product research and ordering.
    Implements tools for researching products with prices and placing orders directly.
    
    Uses offline product database with embedding-based semantic search.
    """

    def __init__(
        self,
        product_db_path: str = "data/vending/products.jsonl",
        use_embeddings: bool = True,
        embedding_model: str = "text-embedding-3-small",
        model_pricing_config_path: Optional[str] = None,
        add_instructions: bool = True,
        **kwargs: Any,
    ):
        tools = [
            self.products_research,
            self.order_place,
        ]
        
        super().__init__(
            name="supplier_communication_tools",
            tools=tools,
            add_instructions=add_instructions,
            auto_register=True,
            **kwargs,
        )
        
        self.product_db = ProductDatabase(
            jsonl_path=product_db_path,
            use_embeddings=use_embeddings,
            embedding_model=embedding_model,
            cache_embeddings=True,
            model_pricing_config_path=model_pricing_config_path
        )
        
        self.orders: List[Dict[str, Any]] = []
        self.delivery_days: int = 7

    def products_research(self, session_state: Dict[str, Any], query: str) -> str:
        """Search for retail products with wholesale pricing information.
        
        Args:
            query: Search query for products
        
        Returns:
            A JSON string with search results:
            {
                "query": "your search query",
                "products": [
                    {"name": "Product Name", "category": "category_name", "wholesale_price": 0.50},
                    ...
                ],
                "total_products_found": 3
            }
        
        Note:
            - Effect: Returns 3 products with wholesale prices, saves to state for ordering
            - Uses semantic search for flexible matching
            - If returned products are totally not related to the query, such products may be unavailable currently.
        """
        start_time = time.time()
        try:
            top_k = 3
            
            search_results = self.product_db.search(
                query=query,
                top_k=top_k,
                category_filter=None
            )
            
            products_list = []
            for product in search_results:
                products_list.append({
                    "name": product.get("name", "Unknown"),
                    "category": product.get("category", "unknown"),
                    "wholesale_price": product.get("wholesale_price", 0.50)
                })
            
            if session_state is None:
                session_state = {}
            
            history = session_state.setdefault("product_research_history", [])
            if len(history) >= 2:
                history.pop(0)
            history.append({
                "query": query,
                "results": products_list,
                "timestamp": time.time()
            })
            
            existing_products = session_state.setdefault("products", [])
            wholesale_prices = session_state.setdefault("wholesale_prices", {})
            
            for p in products_list:
                name = p.get("name", "")
                price = p.get("wholesale_price")
                
                if name and name not in existing_products:
                    existing_products.append(name)
                
                if name and isinstance(price, (int, float)) and price > 0:
                    wholesale_prices[name] = float(price)
            
            return json.dumps({
                "query": query,
                "products": products_list,
                "total_products_found": len(products_list)
            }, ensure_ascii=False, indent=2)
        finally:
            elapsed_time = time.time() - start_time
            print(f"[Timing] products_research (offline): {elapsed_time:.3f} seconds")

    def order_place(self, session_state: Dict[str, Any], items: List[Dict[str, Any]]) -> str:
        """Place a product order for retail inventory.
        
        Args:
            items: List of order items with "name" (str) and "quantity" (int)
                   Product names must exactly match results from products_research()
                   Maximum 1 product per order
                   Example: [{"name": "Product Name 1", "quantity": 10}]
        
        Returns:
            A JSON string with order confirmation:
            {
                "status": "success",
                "delivery_days": 7,
                "order": {
                    "items": [...],
                    "status": "processing",
                    "created_day": 0,
                    "delivery_day": 7,
                    "total_cost": 50.00
                }
            }
        
        Note:
            - Effect: Creates order with 7-day delivery time, deducts cost immediately
            - Inventory added on delivery day (via task_done)
        """
        start_time = time.time()
        try:
            if session_state is None:
                session_state = {}
            
            if isinstance(items, str):
                try:
                    items = json.loads(items)
                except json.JSONDecodeError:
                    items_preview = items[:100] + "..." if len(items) > 100 else items
                    return json.dumps({
                        "status": "error",
                        "error_type": "invalid_json",
                        "message": "Invalid JSON format for items parameter",
                        "items_preview": items_preview
                    }, ensure_ascii=False)
            
            if not isinstance(items, list):
                return json.dumps({
                    "status": "error",
                    "error_type": "invalid_type",
                    "message": f"items must be a list, got {type(items).__name__}"
                }, ensure_ascii=False)
            
            if len(items) == 0:
                return json.dumps({
                    "status": "error",
                    "error_type": "empty_items",
                    "message": "items list cannot be empty. You must provide at least one item with 'name' and 'quantity'."
                }, ensure_ascii=False)
            
            max_items_per_order = 1
            if len(items) > max_items_per_order:
                return json.dumps({
                    "status": "error",
                    "error_type": "too_many_items",
                    "message": f"Cannot order more than {max_items_per_order} different product in a single order. You provided {len(items)} products.",
                    "max_allowed": max_items_per_order,
                    "items_provided": len(items)
                }, ensure_ascii=False)
            
            current_day = session_state.get("day", 0)
            wholesale_prices = session_state.get("wholesale_prices", {})
            
            missing_products = []
            total_cost = 0.0
            
            for item in items:
                if not isinstance(item, dict):
                    item_str = str(item)
                    item_preview = item_str[:100] + "..." if len(item_str) > 100 else item_str
                    return json.dumps({
                        "status": "error",
                        "error_type": "invalid_item_type",
                        "message": f"Each item must be a dictionary, got {type(item).__name__}",
                        "item_preview": item_preview
                    }, ensure_ascii=False)
                
                product_name = item.get("name", "")
                quantity = item.get("quantity", 0)
                
                if not product_name:
                    return json.dumps({
                        "status": "error",
                        "error_type": "missing_name",
                        "message": "Each item must have a 'name' field",
                        "item_info": {"name": product_name, "quantity": quantity}
                    }, ensure_ascii=False)
                if not isinstance(quantity, int) or quantity <= 0:
                    return json.dumps({
                        "status": "error",
                        "error_type": "invalid_quantity",
                        "message": "Each item must have a 'quantity' field (positive integer)",
                        "item_info": {"name": product_name, "quantity": quantity}
                    }, ensure_ascii=False)
                
                if product_name not in wholesale_prices:
                    missing_products.append(product_name)
                else:
                    price = wholesale_prices[product_name]
                    total_cost += price * quantity
            
            if missing_products:
                all_available = list(wholesale_prices.keys())
                available_products = all_available[:5]
                return json.dumps({
                    "status": "error",
                    "error_type": "missing_prices",
                    "missing_products": missing_products,
                    "available_products_sample": available_products,
                    "total_available": len(all_available),
                    "message": "Product names must match EXACTLY (case-sensitive) with names from research_products() results."
                }, ensure_ascii=False, indent=2)
            
            current_money = session_state.get("money", 0.0)
            
            if current_money < total_cost:
                return json.dumps({
                    "status": "error",
                    "error_type": "insufficient_funds",
                    "message": f"Insufficient funds. Order cost: ${total_cost:.2f}, Available: ${current_money:.2f}",
                    "order_cost": round(total_cost, 2),
                    "available_money": round(current_money, 2),
                    "shortfall": round(total_cost - current_money, 2)
                }, ensure_ascii=False, indent=2)
            
            session_state["money"] = current_money - total_cost
            
            delivery_day = current_day + self.delivery_days
            
            order_details = {
                "items": items,
                "status": "processing",
                "created_day": current_day,
                "delivery_day": delivery_day,
                "delivery_days": self.delivery_days,
                "total_cost": round(total_cost, 2),
            }
            self.orders.append(order_details)
            session_state.setdefault("orders", []).append(order_details)
            
            return json.dumps({
                "status": "success",
                "delivery_days": self.delivery_days,
                "order": order_details
            }, ensure_ascii=False, indent=2)
        finally:
            elapsed_time = time.time() - start_time
            print(f"[Timing Statistics] order_place: {elapsed_time:.3f} seconds")