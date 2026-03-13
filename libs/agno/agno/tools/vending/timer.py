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
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import time
import json

from agno.tools.toolkit import Toolkit

if TYPE_CHECKING:
    from agno.tools.seller import SalesTools
    from agno.tools.supplier import SupplierCommunicationTools


class TimerTools(Toolkit):
    """
    A simple toolkit to control simulated time in the Agent session state.

    - task_done(): Signal that all actions for the current day are complete,
      which will advance to the next day and automatically process all time-based events.
    """

    def __init__(
        self,
        sales_tools: Optional["SalesTools"] = None,
        supplier_tools: Optional["SupplierCommunicationTools"] = None,
        add_instructions: bool = True,
        **kwargs: Any,
    ):
        tools = [self.task_done]
        
        self.sales_tools = sales_tools
        self.supplier_tools = supplier_tools

        super().__init__(
            name="timer_tools",
            tools=tools,
            add_instructions=add_instructions,
            auto_register=True,
            show_result_tools=["task_done"],
            **kwargs,
        )

    def task_done(self, session_state: Dict[str, Any]) -> str:
        """Complete current day operations, simulate daily events, and advance to next day.
        
        Returns:
            A JSON string with day transition summary:
            {
                "status": "success",
                "current_day": 2,
                "events": {
                    "sales": {
                        "items_sold": 15,
                        "revenue": 45.00,
                        "details": {...}
                    },
                    "deliveries": [...],
                    "replies": 0,
                    "errors": [...]
                }
            }
        
        Note:
            - Effect: Advances day, processes sales and deliveries
            - Automatically updates inventory based on daily events
        """
        if session_state is None:
            session_state = {}

        if "day" not in session_state or not isinstance(session_state["day"], int):
            session_state["day"] = 0

        current_day = int(session_state["day"])
        events = {
            "sales": {},
            "deliveries": [],
            "replies": 0,
            "errors": []
        }
        
        if self.sales_tools is not None:
            try:
                sold_json = self.sales_tools.simulate_day(session_state)
                sold = json.loads(sold_json) if isinstance(sold_json, str) else sold_json
                total_sold = sum(sold.values())
                if total_sold > 0:
                    revenue = sum(
                        qty * session_state.get("product_prices", {}).get(product, 0)
                        for product, qty in sold.items()
                    )
                    events["sales"] = {
                        "items_sold": total_sold,
                        "revenue": round(revenue, 2),
                        "details": sold
                    }
            except Exception as e:
                events["errors"].append(f"Sales simulation failed: {str(e)}")

        new_day = current_day + 1
        session_state["day"] = new_day

        day_history: List[int] = session_state.setdefault("day_history", [])
        day_history.append(new_day)

        day_events = self._process_day_events(session_state, new_day)
        if day_events:
            events.update(day_events)
        
        current_state = self._get_current_state_summary(session_state)
        
        total_revenue = events.get("sales", {}).get("revenue", 0)
        items_sold = events.get("sales", {}).get("items_sold", 0)
        deliveries = len(events.get("deliveries", []))
        
        summary_parts = [f"Day {new_day}"]
        if items_sold > 0:
            summary_parts.append(f"Sales: {items_sold} items, ${total_revenue:.2f} revenue")
        if deliveries > 0:
            summary_parts.append(f"Deliveries: {deliveries} orders received")
        summary = " | ".join(summary_parts)
        
        return json.dumps({
            "status": "success",
            "current_day": new_day,
            "events": events,
            "current_state": current_state,
            "summary": summary
        }, ensure_ascii=False, indent=2)

    def _process_day_events(self, session_state: Dict[str, Any], new_day: int) -> dict:
        """Process all events that should occur on the new day.
        
        Returns a dict of events that happened.
        """
        events = {
            "deliveries": [],
            "replies": 0
        }
        
        orders = session_state.get("orders", [])
        delivered_orders = []
        total_cost = 0.0
        
        for order in orders:
            if order.get("status") == "processing" and order.get("delivery_day") == new_day:
                items = order.get("items", [])
                product_quantities = session_state.setdefault("product_quantities", {})
                
                for item in items:
                    product_name = item.get("name", "")
                    quantity = item.get("quantity", 0)
                    
                    if product_name and quantity > 0:
                        product_quantities[product_name] = product_quantities.get(product_name, 0) + quantity
                
                total_cost += order.get("total_cost", 0.0)
                
                order["status"] = "delivered"
                order["delivered_day"] = new_day
                delivered_orders.append(order)
        
        if delivered_orders:
            events["deliveries"] = [{
                "order_count": len(delivered_orders),
                "total_cost": round(total_cost, 2),
                "orders": delivered_orders
            }]
        
        if self.supplier_tools is not None:
            events["replies"] = 0
        
        return events

    def _get_current_state_summary(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get current business state summary for agent visibility.
        
        Returns a dict with key business metrics.
        """
        money = session_state.get("money", 0.0)
        product_quantities = session_state.get("product_quantities", {})
        total_inventory = sum(product_quantities.values())
        
        orders = session_state.get("orders", [])
        pending_orders = sum(1 for order in orders if order.get("status") == "processing")
        
        product_count = len(product_quantities)
        
        return {
            "money": round(money, 2),
            "total_inventory_items": total_inventory,
            "product_types_count": product_count,
            "pending_orders": pending_orders
        }

