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

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def vending_bench_is_finished(
    state: Dict[str, Any],
    no_sales_days_threshold: int = 5,
    max_days: Optional[int] = None,
) -> bool:
    """
    Determine whether the simulation should terminate

    Termination condition: Money is less than or equal to zero AND no sales for N consecutive days

    Args:
        state: Current state dictionary, including money and sales_history
        no_sales_days_threshold: Number of consecutive days with no sales to trigger termination, default is 5 days

    Returns:
        True means the simulation should terminate, False means continue running
    """
    day = state.get("day", 0)
    if max_days is not None and day >= max_days + 1:
        logger.info(f"[Termination Condition Met] Maximum days limit exceeded: day={day}, max_days={max_days}")
        return True
    money = state.get("money", 0)
    if money > 0:
        return False
    
    sales_history = state.get("sales_history", [])
    
    if len(sales_history) < no_sales_days_threshold:
        return False
    
    recent_days = sales_history[-no_sales_days_threshold:]
    
    for day_record in recent_days:
        revenue = day_record.get("revenue", 0)
        sold = day_record.get("sold", {})
        
        if revenue > 0 or (sold and any(qty > 0 for qty in sold.values())):
            return False
    
    logger.info(f"[Termination Condition Met] Funds: ${money:.2f}, No sales for {no_sales_days_threshold} consecutive days")
    return True

def vending_bench_cal_metric(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate and print key metrics

    Args:
        state: Current state dictionary

    Returns:
        Dictionary containing key metrics
    """
    money = state.get("money", 0)
    day = state.get("day", 0)
    sales_history = state.get("sales_history", [])
    

    total_revenue = sum(record.get("revenue", 0) for record in sales_history)
    

    avg_daily_revenue = total_revenue / len(sales_history) if sales_history else 0

    product_quantities = state.get("product_quantities", {})
    total_inventory = sum(product_quantities.values())
    
    wholesale_prices = state.get("wholesale_prices", {})
    inventory_value = sum(
        product_quantities.get(product, 0) * wholesale_prices.get(product, 0.0)
        for product in product_quantities.keys()
    )
    

    orders = state.get("orders", [])
    pending_orders_value = sum(
        order.get("total_cost", 0.0)
        for order in orders
        if order.get("status") == "processing"
    )
    
  
    net_worth = money + inventory_value + pending_orders_value
    
    last_sales_day = 0
    for record in reversed(sales_history):
        revenue = record.get("revenue", 0)
        sold = record.get("sold", {})

        if revenue > 0 or (sold and any(qty > 0 for qty in sold.values())):
            last_sales_day = record.get("day", 0)
            break
    
    days_before_sales_stopped = last_sales_day
    
    print(f"[Metric] Day {day} | Cash: ${money:.2f} | Net Worth: ${net_worth:.2f} | Total Revenue: ${total_revenue:.2f} | Average Daily Revenue: ${avg_daily_revenue:.2f} | Inventory: {total_inventory} units | Last Sale: Day {last_sales_day}")
    
    return {
        "final_day": day,
        "final_money": round(money, 2),
        "final_net_worth": round(net_worth, 2),
        "final_inventory_value": round(inventory_value, 2),
        "final_pending_orders_value": round(pending_orders_value, 2),
        "total_revenue": round(total_revenue, 2),
        "avg_daily_revenue": round(avg_daily_revenue, 2),
        "total_inventory": total_inventory,
        "total_sales_days": len(sales_history),
        "days_before_sales_stopped": days_before_sales_stopped,
        "last_sales_day": last_sales_day,
    }