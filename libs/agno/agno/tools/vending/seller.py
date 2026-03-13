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
from __future__ import annotations

"""
Sales (Demand) Module
---------------------

This module simulates customer demand and provides pricing recommendations for a
retail business environment. The current design focuses on:

- Group-based demand using an external demand_structure.json
- Seasonal demand per group via sinusoidal curves
- Intra-group competition via price-based share allocation
- Optional per-product demand parameters for price recommendation
- Deterministic RNG via an optional seed

Public interface (see `SalesModel` and `SalesTools`):
- get_price(session_state, product_name)
- set_price(session_state, product_name, price)
- simulate_day(session_state)  -> updates inventory, money, sales_history
- (internal helpers) recommend_price / recommend_prices for analysis or tooling
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Union, Any
import json
import math
import random
import os

Weather = Literal["sunny", "rainy", "cloudy", "hot", "cold"]


class PricingObjective(str, Enum):
    PROFIT = "profit"
    REVENUE = "revenue"
    SELL_THROUGH = "sell_through"


@dataclass
class DemandParams:
    """Price sensitivity triple for a product.

    eps (elasticity) is negative; magnitude indicates sensitivity.
    p_ref is the reference price perceived by customers.
    b_base is baseline expected daily demand near p_ref (before exogenous factors).
    """

    eps: float
    p_ref: float
    b_base: float
    generated_by: Literal["rule", "llm"] = "rule"
    cached_at: Optional[str] = None


@dataclass
class DayInfo:
    """Exogenous factors for a day."""

    date: str
    dow: int
    month: int
    weather: Weather
    location_factor: float = 1.0


@dataclass
class MachineState:
    """Readonly snapshot of machine state used by the sales model."""

    product_prices: Dict[str, float]
    product_quantities: Dict[str, int]
    product_categories: Dict[str, str]


@dataclass
class SalesObservation:
    """Observed sales, used for online updates (optional)."""

    product_name: str
    price: float
    sold_qty: int
    day_info: DayInfo


@dataclass
class DemandConfig:
    """Config knobs for price recommendation and EMA updates.

    Note:
        These parameters are used only for:
        - Per-product DemandParams initialization
        - `recommend_price` / `recommend_prices`
        - `update_demand_params` (EMA on b_base / eps)

        The main environment demand (used by `simulate_day`) is fully driven by
        the group-based model and demand_structure.json, and does NOT depend
        on these fields.
    """

    dow_factor: List[float] = field(
        default_factory=lambda: [1.00, 1.00, 1.00, 1.00, 1.05, 1.20, 1.25]
    )
    month_factor: List[float] = field(
        default_factory=lambda: [0.95, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.20, 1.10, 1.00, 0.98, 0.96]
    )
    weather_factor: Dict[Weather, float] = field(
        default_factory=lambda: {
            "sunny": 1.05,
            "hot": 1.10,
            "rainy": 1.08,
            "cold": 1.05,
            "cloudy": 1.00,
        }
    )

    choice_max_reduction: float = 0.50
    choice_optimal_options: int = 8
    choice_alpha: float = 0.05

    noise_std: float = 0.20

    grid_start_ratio: float = 0.80
    grid_stop_ratio: float = 1.50
    grid_step_ratio: float = 0.05

    ema_alpha: float = 0.05
    adjust_eps: bool = False
    eps_alpha: float = 0.01
    eps_bounds: tuple[float, float] = (-2.5, -0.4)

    seed: Optional[int] = 2025


class SalesModel:
    """Group-based demand simulator + pricing recommendation engine.

    The primary demand model is group-based and driven by an external
    demand_structure.json plus the offline products catalog:

    - Each product is assigned to a demand group
    - Each group has a seasonal demand curve (Base, Amp, T, Phi)
    - Group-level competition / complement relations adjust effective demand
    - Within a group, products share group demand according to price-based utilities

    The legacy per-product DemandParams and EMA updater are kept only for
    price recommendation and possible future extensions.
    """

    def __init__(
        self,
        config: Optional[DemandConfig] = None,
        demand_structure_path: Optional[str] = None,
        product_catalog_path: Optional[str] = None,
    ):
        self.cfg = config or DemandConfig()
        self._params: Dict[str, DemandParams] = {}
        self._rng = random.Random(self.cfg.seed)
        self.demand_structure = self._load_demand_structure(demand_structure_path)
        self._product_meta: Dict[str, str] = self._load_product_catalog(product_catalog_path)
        self._product_group_cache: Dict[str, str] = {}


    def get_price(self, product_name: str, machine_state: MachineState) -> float:
        return float(machine_state.product_prices[product_name])

    def get_demand_params(self, product_name: str) -> DemandParams:
        return self._ensure_params(product_name)

    def recommend_price(
        self,
        product_name: str,
        machine_state: MachineState,
        objective: PricingObjective = PricingObjective.PROFIT,
        wholesale_cost: Optional[float] = None,
        day_info: Optional[DayInfo] = None,
    ) -> float:
        """Grid-search around reference price under a given `objective`.

        If `day_info` is None, an average sunny weekend day is assumed.
        """
        params = self._ensure_params(product_name)
        inv = int(machine_state.product_quantities.get(product_name, 0))
        if inv <= 0:
            return round(max(0.01, params.p_ref), 2)

        if day_info is None:
            day_info = DayInfo(date="1970-07-05", dow=5, month=7, weather="sunny")

        start = params.p_ref * self.cfg.grid_start_ratio
        stop = params.p_ref * self.cfg.grid_stop_ratio
        step = max(0.01, params.p_ref * self.cfg.grid_step_ratio)

        best_price, best_score = params.p_ref, -math.inf
        cat = machine_state.product_categories.get(product_name, "misc")
        cat_count = self._category_active_count(cat, machine_state)

        for p in self._frange(start, stop, step):
            q = self._expected_qty(product_name, p, inv, cat_count, day_info)
            if objective == PricingObjective.REVENUE:
                score = p * q
            elif objective == PricingObjective.SELL_THROUGH:
                score = q + 0.05 * p
            else:  # PROFIT
                c = wholesale_cost or 0.0
                score = (p - c) * q

            if score > best_score:
                best_score, best_price = score, p

        return round(max(0.01, best_price), 2)

    def recommend_prices(
        self,
        product_names: List[str],
        machine_state: MachineState,
        objective: PricingObjective = PricingObjective.PROFIT,
        wholesale_costs: Optional[Dict[str, float]] = None,
        day_info: Optional[DayInfo] = None,
    ) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for name in product_names:
            result[name] = self.recommend_price(
                product_name=name,
                machine_state=machine_state,
                objective=objective,
                wholesale_cost=(wholesale_costs or {}).get(name),
                day_info=day_info,
            )
        return result

    def simulate_day(
        self,
        machine_state: MachineState,
        day_info: DayInfo,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Simulate sales for each product using the group-based demand model."""
        t: Optional[int] = None
        if session_state is not None:
            try:
                t = int(session_state.get("day", 0))
            except Exception:
                t = None

        return self._simulate_day_group_based(machine_state, day_info, t=t, session_state=session_state)

    def update_demand_params(self, obs: SalesObservation) -> None:
        """Lightweight EMA update on b_base; optional safeguarded tweak on eps.

        This updater belongs to the legacy per-product demand model and is used
        only to refine parameters consumed by `recommend_price` /
        `recommend_prices`. The group-based demand model (used in `simulate_day`)
        does not depend on these updates.
        """
        params = self._ensure_params(obs.product_name)

        denom = max(
            self._dow_factor(obs.day_info.dow)
            * self._month_factor(obs.day_info.month)
            * self._weather_factor(obs.day_info.weather)
            * float(max(0.0, obs.day_info.location_factor)),
            1e-6,
        )
        q0 = obs.sold_qty / denom

        alpha = self.cfg.ema_alpha
        params.b_base = (1 - alpha) * params.b_base + alpha * q0

        if self.cfg.adjust_eps:
            delta = (obs.price - params.p_ref) / max(1e-6, params.p_ref)
            direction = -1.0 if (delta > 0 and obs.sold_qty > params.b_base) else 1.0
            new_eps = params.eps + self.cfg.eps_alpha * direction
            lo, hi = self.cfg.eps_bounds
            params.eps = float(min(hi, max(lo, new_eps)))

        self._params[obs.product_name] = params

    def _ensure_params(self, product_name: str) -> DemandParams:
        if product_name not in self._params:
            eps = self._rng.uniform(-1.6, -0.8)
            p_ref = round(self._rng.uniform(1.2, 2.5), 2)
            b_base = self._rng.uniform(3.0, 10.0)
            self._params[product_name] = DemandParams(eps=eps, p_ref=p_ref, b_base=b_base, generated_by="rule")
        return self._params[product_name]

    def _frange(self, start: float, stop: float, step: float):
        x = float(start)
        while x <= stop + 1e-12:
            yield round(x, 4)
            x += step

    def _dow_factor(self, dow: int) -> float:
        return float(self.cfg.dow_factor[dow % 7])

    def _month_factor(self, month: int) -> float:
        return float(self.cfg.month_factor[(month - 1) % 12])

    def _weather_factor(self, weather: Weather) -> float:
        return float(self.cfg.weather_factor.get(weather, 1.0))

    def _choice_multiplier(self, k_options: int) -> float:
        """Category diversity penalty.

        Smoothly reduces demand when the number of options exceeds an optimal
        threshold. Clamped by `choice_max_reduction`.
        """
        if k_options <= self.cfg.choice_optimal_options:
            return 1.0
        over = max(0, k_options - self.cfg.choice_optimal_options)
        mult = 1.0 - self.cfg.choice_alpha * over
        floor = 1.0 - self.cfg.choice_max_reduction
        return max(floor, mult)

    def _category_active_count(self, category: str, machine_state: MachineState) -> int:
        return sum(
            1
            for name, q in machine_state.product_quantities.items()
            if q > 0 and machine_state.product_categories.get(name, "misc") == category
        )

    def _expected_qty(
        self,
        product_name: str,
        price: float,
        stock_cap: int,
        cat_active_count: int,
        day_info: DayInfo,
    ) -> float:
        params = self._ensure_params(product_name)
        delta = (price - params.p_ref) / max(1e-8, params.p_ref)
        impact = params.eps * delta
        q0 = max(0.0, params.b_base * (1.0 + impact))

        q1 = q0 * self._dow_factor(day_info.dow) * self._month_factor(day_info.month) * self._weather_factor(day_info.weather)
        q1 *= float(max(0.0, day_info.location_factor))

        q2 = q1 * self._choice_multiplier(cat_active_count)

        return min(q2, float(stock_cap))

    def _apply_noise(self, x: float) -> float:
        sigma = abs(x) * self.cfg.noise_std
        return max(0.0, self._rng.gauss(x, sigma))


    def _load_product_catalog(self, path: Optional[str] = None) -> Dict[str, str]:
        """Load offline product catalog (name -> category) from products.jsonl.

        This is used to assign SKUs to groups even if session_state did not
        populate product_categories for newly delivered items.
        """
        if path is None:
            default_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "..",
                "data",
                "products.jsonl",
            )
            path = default_path

        meta: Dict[str, str] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = obj.get("name")
                    category = obj.get("category")
                    if isinstance(name, str) and isinstance(category, str):
                        meta[name] = category
        except Exception:
            return {}
        return meta

    def _load_demand_structure(self, path: Optional[str]):
        """Load demand_structure.json if present and normalize common shorthand forms.

        Normalization handles:
        - groups[].members provided as a string or list of strings -> convert to
          [{"match": "category", "value": <cat>}]
        - relations with competition given as from/to -> converted to between
        - ensures ids are strings (seller uses them as keys)
        """
        if not path:
            default_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "..",
                "data",
                "demand_structure.json",
            )
            path = default_path
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        groups = data.get("groups", [])
        for g in groups:
            if "id" in g:
                g["id"] = str(g["id"])
            members = g.get("members", [])
            if isinstance(members, str):
                g["members"] = [{"match": "category", "value": members}]
            elif isinstance(members, list) and all(isinstance(m, str) for m in members):
                g["members"] = [{"match": "category", "value": m} for m in members]

        rels = data.get("relations", [])
        for r in rels:
            rtype = r.get("type")
            if "from" in r:
                r["from"] = str(r["from"])
            if "to" in r:
                r["to"] = str(r["to"])
            if "between" in r:
                r["between"] = [str(x) for x in r.get("between", [])]
            if rtype == "competition" and "between" not in r:
                if r.get("from") and r.get("to"):
                    r["between"] = [str(r["from"]), str(r["to"])]
        data["groups"] = groups
        data["relations"] = rels
        return data

    def _match_group(self, product_name: str, category: str) -> Optional[str]:
        """Map product_name to group_id using demand_structure members matching rules."""
        if product_name in self._product_group_cache:
            return self._product_group_cache[product_name]
        if not self.demand_structure:
            return None
        groups = self.demand_structure.get("groups", [])
        for g in groups:
            for member in g.get("members", []):
                if member.get("match") == "category" and member.get("value") == category:
                    gid = g.get("id")
                    if gid:
                        self._product_group_cache[product_name] = gid
                        return gid
        return None

    def _group_demand_curve(
        self, 
        base: float, 
        seasonality: dict, 
        t: int,
        avg_price: float = 0.0,
        p_ref: float = 0.0,
        epsilon: float = 0.0
    ) -> float:
        """Seasonal demand with group-level price elasticity.

        Formula:
            base * (1 + amp * sin(2π/T * t + phi)) * (avg_price / p_ref)^epsilon
        
        Note: 
        - amp is normalized to be a relative multiplier (0-1 range).
        - epsilon < 0 means demand decreases as price increases (normal goods)
        - If avg_price or p_ref is 0, price adjustment is skipped
        """
        T = max(1.0, float(seasonality.get("T", 30)))
        phi = float(seasonality.get("phi", 0.0))
        amp_raw = float(seasonality.get("amp", 0.0))
        amp = amp_raw / 100.0 if amp_raw > 1.0 else amp_raw
        amp = max(0.0, min(1.0, amp))
        
        seasonal_val = base * (1.0 + amp * math.sin(2 * math.pi / T * t + phi))
        
        if avg_price > 0 and p_ref > 0 and epsilon != 0.0:
            price_ratio = avg_price / p_ref
            price_adjustment = math.pow(price_ratio, epsilon)
            price_adjustment = max(0.01, min(100.0, price_adjustment))
        else:
            price_adjustment = 1.0
        
        final_demand = seasonal_val * price_adjustment
        return max(0.0, final_demand)

    def _apply_relations_caps(self, group_demand: Dict[str, float]) -> Dict[str, float]:
        """Apply complement/competition relations at group level."""
        if not self.demand_structure:
            return group_demand
        relations = self.demand_structure.get("relations", [])
        adjusted = dict(group_demand)
        for rel in relations:
            rtype = rel.get("type")
            if rtype == "complement":
                src = rel.get("from")
                tgt = rel.get("to")
                strength = float(rel.get("strength", 0.0))
                if src in adjusted and tgt in adjusted:
                    cap = max(0.0, adjusted[src] * strength)
                    adjusted[tgt] = min(adjusted[tgt], cap)
            elif rtype == "competition":
                between = rel.get("between", [])
                if len(between) == 2:
                    g1, g2 = between
                    strength = float(rel.get("strength", 0.0))
                    if g1 in adjusted and g2 in adjusted:
                        factor = max(0.0, 1.0 - 0.5 * strength)
                        adjusted[g1] *= factor
                        adjusted[g2] *= factor
        return adjusted

    def _simulate_day_group_based(
        self,
        machine_state: MachineState,
        day_info: DayInfo,
        t: Optional[int] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Group-based demand allocation with intra-group share and inter-group caps.

        Args:
            machine_state: Snapshot of product prices/quantities/categories.
            day_info: Exogenous day information (still used for backwards compatibility).
            t: Optional global time index (e.g. session_state["day"]). If not
               provided, a proxy is derived from day_info.
        """
        sold: Dict[str, int] = {}
        if not self.demand_structure:
            return sold

        groups = self.demand_structure.get("groups", [])

        product_group: Dict[str, str] = {}
        for name in machine_state.product_prices.keys():
            cat = machine_state.product_categories.get(name) or self._product_meta.get(name, "misc")
            gid = self._match_group(name, cat)
            if gid:
                product_group[name] = gid

        wholesale_prices = {}
        if session_state is not None:
            wholesale_prices = session_state.get("wholesale_prices", {})
        
        group_demand: Dict[str, float] = {}
        if t is not None:
            t_val = t
        else:
            t_val = day_info.dow + day_info.month * 30

        for g in groups:
            gid = g.get("id")
            base = float(g.get("base_demand", 0.0))
            season = g.get("seasonality", {})
            
            price_sensitivity = g.get("price_sensitivity", {})
            if not price_sensitivity:
                raise ValueError(
                    f"Missing 'price_sensitivity' config for group '{gid}'. "
                    f"Please add price_sensitivity with beta, epsilon, and reference_markup "
                    f"to demand_structure.json for this group."
                )
            
            epsilon = float(price_sensitivity.get("epsilon", 0.0))
            
            if "reference_markup" not in price_sensitivity:
                raise ValueError(
                    f"Missing 'reference_markup' in price_sensitivity for group '{gid}'. "
                    f"Please add 'reference_markup' (e.g., 1.1 for 10% markup, 2.5 for 150% markup) "
                    f"to the price_sensitivity config in demand_structure.json."
                )
            reference_markup = float(price_sensitivity["reference_markup"])
            
            avg_price = 0.0
            p_ref = 0.0
            group_items_for_price = [name for name, gid_temp in product_group.items() if gid_temp == gid]
            
            if group_items_for_price and epsilon != 0.0:
                total_inventory = 0
                weighted_price_sum = 0.0
                wholesale_sum = 0.0
                wholesale_count = 0
                
                for name in group_items_for_price:
                    if name in wholesale_prices:
                        wholesale = float(wholesale_prices[name])
                        if wholesale > 0:
                            wholesale_sum += wholesale
                            wholesale_count += 1
                
                for name in group_items_for_price:
                    price = float(machine_state.product_prices.get(name, 0.0))
                    inv = int(machine_state.product_quantities.get(name, 0))
                    if inv > 0 and price > 0:
                        weighted_price_sum += price * inv
                        total_inventory += inv
                
                if total_inventory > 0:
                    avg_price = weighted_price_sum / total_inventory
                    
                    if wholesale_count > 0:
                        avg_wholesale = wholesale_sum / wholesale_count
                        p_ref = avg_wholesale * reference_markup
                        p_ref = max(0.5, p_ref)
                    else:
                        p_ref = avg_price
                elif wholesale_count > 0:
                    avg_wholesale = wholesale_sum / wholesale_count
                    avg_price = avg_wholesale * reference_markup
                    p_ref = avg_price
            
            demand_val = self._group_demand_curve(base, season, t_val, avg_price, p_ref, epsilon)
            group_demand[gid] = demand_val

        group_demand = self._apply_relations_caps(group_demand)

        group_items: Dict[str, List[str]] = {}
        for name, gid in product_group.items():
            group_items.setdefault(gid, []).append(name)

        for gid, items in group_items.items():
            demand_cap = group_demand.get(gid, 0.0)
            if demand_cap <= 0:
                for sku in items:
                    sold[sku] = 0
                continue

            beta = 1.0
            gmeta = next((g for g in groups if g.get("id") == gid), None)
            if gmeta:
                beta = float(gmeta.get("price_sensitivity", {}).get("beta", 1.0))

            wholesale_prices = {}
            if session_state is not None:
                wholesale_prices = session_state.get("wholesale_prices", {})

            if not gmeta:
                raise ValueError(f"Group metadata not found for group {gid}. This should not happen.")
            
            price_sensitivity = gmeta.get("price_sensitivity", {})
            if "reference_markup" not in price_sensitivity:
                raise ValueError(
                    f"Missing 'reference_markup' in price_sensitivity for group '{gid}'. "
                    f"Please add 'reference_markup' (e.g., 1.1 for 10% markup, 2.5 for 150% markup) "
                    f"to the price_sensitivity config in demand_structure.json for this group."
                )
            
            reference_markup = float(price_sensitivity["reference_markup"])
            
            available_prices = []
            available_wholesale_prices = []
            for name in items:
                price = float(machine_state.product_prices.get(name, 0.0))
                inv = int(machine_state.product_quantities.get(name, 0))
                if inv > 0 and price > 0:
                    available_prices.append(price)
                    if name in wholesale_prices:
                        wholesale = float(wholesale_prices[name])
                        if wholesale > 0:
                            available_wholesale_prices.append(wholesale)
            
            if available_wholesale_prices:
                avg_wholesale = sum(available_wholesale_prices) / len(available_wholesale_prices)
                p_ref = avg_wholesale * reference_markup
                if p_ref < 0.5:
                    p_ref = 0.5
            elif available_prices:
                p_ref = sum(available_prices) / len(available_prices)
            else:
                all_prices = [float(machine_state.product_prices.get(name, 0.0)) for name in items]
                p_ref = min(p for p in all_prices if p > 0) if any(p > 0 for p in all_prices) else 1.0

            utils = []
            for name in items:
                price = float(machine_state.product_prices.get(name, 0.0))
                inv = int(machine_state.product_quantities.get(name, 0))
                if inv <= 0:
                    utils.append((name, None, inv))
                else:
                    if p_ref > 0:
                        u = -beta * (price / p_ref)
                    else:
                        u = -beta * price
                    
                    utils.append((name, u, inv))

            exp_sum = sum(math.exp(u) for _, u, inv in utils if u is not None and inv > 0)
            for name, u, inv in utils:
                if inv <= 0 or u is None or exp_sum <= 0:
                    sold[name] = 0
                    continue
                share = math.exp(u) / exp_sum
                exp_qty = share * demand_cap
                qty_noise = self._apply_noise(exp_qty)
                sold[name] = max(0, min(int(round(qty_noise)), inv))

        for name in machine_state.product_prices.keys():
            if name not in sold:
                sold[name] = 0
        return sold



"""
Agno-compatible wrapper for the SalesModel so it can be attached to an
`Agent(tools=[...])` or provided as part of a Team/Workflow.

This adapter exposes SalesModel methods as Agno tools. All inputs/outputs are
JSON-serializable so LLM agents can call them directly.
"""

from typing import Any
from agno.tools import Toolkit


def _machine_state_from_dict(d: dict) -> MachineState:
    return MachineState(
        product_prices={k: float(v) for k, v in d.get("product_prices", {}).items()},
        product_quantities={k: int(v) for k, v in d.get("product_quantities", {}).items()},
        product_categories={k: str(v) for k, v in d.get("product_categories", {}).items()},
    )


def _day_info_from_dict(d: dict) -> DayInfo:
    """Parse a dict into DayInfo with basic validation and clearer errors.

    Required keys: date (str), dow (0-6), month (1-12), weather (str)
    Optional: location_factor (float, default 1.0, clipped to ≥0)
    """
    if d is None:
        raise ValueError("day_info is required")

    missing = [k for k in ("date", "dow", "month", "weather") if k not in d]
    if missing:
        raise ValueError(f"day_info missing required keys: {missing}")

    date = str(d.get("date"))
    dow = int(d.get("dow"))
    month = int(d.get("month"))
    weather = str(d.get("weather"))
    location_factor = float(d.get("location_factor", 1.0))

    if not (0 <= dow <= 6):
        raise ValueError(f"day_info.dow must be in 0..6, got {dow}")
    if not (1 <= month <= 12):
        raise ValueError(f"day_info.month must be in 1..12, got {month}")
    if weather not in ("sunny", "rainy", "cloudy", "hot", "cold"):
        raise ValueError(f"day_info.weather unsupported: {weather}")

    return DayInfo(
        date=date,
        dow=dow,
        month=month,
        weather=weather,  # type: ignore[arg-type]
        location_factor=max(0.0, location_factor),
    )


class SalesTools(Toolkit):
    """Agno Toolkit exposing SalesModel capabilities as tools.

    Register this toolkit on an Agent: `Agent(tools=[SalesTools(SalesModel())])`.
    """

    def __init__(self, sales_model: SalesModel, *, name: str = "sales_tools", show_result: bool = True):
        self.sales = sales_model
        tool_funcs = [
            self.price_query,
            self.price_set,
        ]
        show_result_tools = [f.__name__ for f in tool_funcs] if show_result else []
        super().__init__(
            name=name,
            tools=tool_funcs,
            show_result_tools=show_result_tools,
        )

    def price_query(self, session_state: Dict[str, Any], product_name: str) -> str:
        """Retrieve the current retail price for a product.
        
        Args:
            product_name: Product name key (case-sensitive)
        
        Returns:
            A JSON string with price information:
            {
                "product_name": "Product Name",
                "price": 1.50
            }
        
        Note:
            - Effect: Read-only price query, no state changes
            - Returns error with available products if product not found
        """
        product_prices = session_state.get("product_prices", {})
        if product_name not in product_prices:
            available_products = list(product_prices.keys()) if product_prices else "none"
            product_quantities = session_state.get("product_quantities", {})
            available_inventory_products = list(product_quantities.keys()) if product_quantities else "none"
            
            error_msg = (
                f"Product '{product_name}' not found in product_prices. "
                f"Available products with prices: {available_products}. "
            )
            
            if available_inventory_products != "none" and available_inventory_products != available_products:
                error_msg += (
                    f"Note: You have inventory for products {available_inventory_products}, "
                    f"but prices haven't been set yet. "
                    f"Please set prices for these product keys first, then call price_query(). "
                )
            else:
                error_msg += (
                    f"Ensure you use the exact product name key that exists in product_prices/product_quantities. "
                    f"If you haven't placed any orders yet, you need to: "
                    f"1) Place an order using order_place(), "
                    f"2) Wait for delivery (task_done()), "
                    f"3) Set prices for the delivered products, "
                    f"4) Then call price_query()."
                )
            
            return error_msg
        return json.dumps({"product_name": product_name, "price": float(product_prices[product_name])}, ensure_ascii=False)

    def price_set(self, session_state: Dict[str, Any], product_name: str, price: float) -> str:
        """Set the retail price for a product.
        
        Args:
            product_name: Product name key matching inventory (case-sensitive)
            price: Retail price per unit in dollars (must be positive)
        
        Returns:
            A JSON string confirming price update:
            {
                "status": "success",
                "product_name": "Product Name",
                "price": 1.50
            }
        
        Note:
            - Effect: Updates product_prices for the specified product
        """
        if session_state is None:
            session_state = {}

        product_prices = session_state.setdefault("product_prices", {})
        wholesale_prices = session_state.get("wholesale_prices", {})

        price = float(price)
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        product_prices[product_name] = price
        return json.dumps({
            "status": "success",
            "product_name": product_name,
            "price": price
        }, ensure_ascii=False)

    def get_demand_params(self, session_state: Dict[str, Any], product_name: str) -> dict:
        """Get inferred demand parameters for a product.

        This helper exposes the internal DemandParams used by the legacy
        price recommendation logic (elasticity, reference price, baseline demand).
        
        Note:
            This does NOT affect the main group-based demand model used by
            `simulate_day`. It is intended for analysis or offline experiments
            around price recommendation only.
        """
        p = self.sales.get_demand_params(product_name)
        return {"eps": p.eps, "p_ref": p.p_ref, "b_base": p.b_base, "generated_by": p.generated_by, "cached_at": p.cached_at}

    def recommend_price(
        self,
        session_state: Dict[str, Any],
        product_name: str,
        objective: str = "profit",
        wholesale_cost: float | None = None,
    ) -> float:
        """Get an internal price recommendation for a product.

        This uses the legacy per-product DemandParams model (with weekday /
        month / weather multipliers) to perform a grid search over price and
        optimize a given objective.

        Note:
            This is NOT used by the main environment workflow. Agents set
            prices directly via `set_price`. This helper is provided only
            for analysis or manual benchmarking of pricing strategies.
        """
        ms = _machine_state_from_dict({
            "product_prices": session_state.get("product_prices", {}),
            "product_quantities": session_state.get("product_quantities", {}),
            "product_categories": session_state.get("product_categories", {}),
        })
        
        day = session_state.get("day", 0)
        di = DayInfo(
            date=f"2025-10-{day + 1:02d}",
            dow=day % 7,
            month=10,
            weather="sunny",
            location_factor=1.0,
        )
        
        if wholesale_cost is None:
            wholesale_prices = session_state.get("wholesale_prices", {})
            wholesale_cost = wholesale_prices.get(product_name)
        
        obj = {
            "profit": PricingObjective.PROFIT,
            "revenue": PricingObjective.REVENUE,
            "sell_through": PricingObjective.SELL_THROUGH,
        }.get(objective.lower(), PricingObjective.PROFIT)
        
        return self.sales.recommend_price(
            product_name=product_name,
            machine_state=ms,
            objective=obj,
            wholesale_cost=wholesale_cost,
            day_info=di,
        )

    def recommend_prices(
        self,
        session_state: Dict[str, Any],
        product_names: list[str],
        objective: str = "profit",
    ) -> dict:
        """Get internal price recommendations for multiple products.

        See `recommend_price` for details. This is only used for analysis /
        experiments and does not affect the group-based demand model.
        """
        ms = _machine_state_from_dict({
            "product_prices": session_state.get("product_prices", {}),
            "product_quantities": session_state.get("product_quantities", {}),
            "product_categories": session_state.get("product_categories", {}),
        })
        
        day = session_state.get("day", 0)
        di = DayInfo(
            date=f"2025-10-{day + 1:02d}",
            dow=day % 7,
            month=10,
            weather="sunny",
            location_factor=1.0,
        )
        
        wholesale_prices = session_state.get("wholesale_prices", {})
        
        obj = {
            "profit": PricingObjective.PROFIT,
            "revenue": PricingObjective.REVENUE,
            "sell_through": PricingObjective.SELL_THROUGH,
        }.get(objective.lower(), PricingObjective.PROFIT)
        
        return self.sales.recommend_prices(
            product_names=product_names,
            machine_state=ms,
            objective=obj,
            wholesale_costs=wholesale_prices,
            day_info=di,
        )

    def simulate_day(self, session_state: Dict[str, Any]) -> str:
        """Simulate daily sales and update inventory, money, and sales history.
        
        Returns:
            A JSON string with sales results:
            {
                "product_name": quantity_sold,
                ...
            }
        
        Note:
            - Effect: Updates inventory, adds revenue to money, appends to sales_history
            - Uses group-based demand model from demand_structure.json
        """
        ms = _machine_state_from_dict({
            "product_prices": session_state.get("product_prices", {}),
            "product_quantities": session_state.get("product_quantities", {}),
            "product_categories": session_state.get("product_categories", {}),
        })
        
        day = session_state.get("day", 0)
        di = DayInfo(
            date=f"2025-10-{day + 1:02d}",
            dow=day % 7,
            month=10,
            weather="sunny",
            location_factor=1.0,
        )
        
        sold = self.sales.simulate_day(ms, di, session_state=session_state)
        
        revenue = 0.0
        for product_name, qty_sold in sold.items():
            if product_name in session_state.get("product_quantities", {}):
                session_state["product_quantities"][product_name] -= qty_sold
            
            if product_name in session_state.get("product_prices", {}):
                revenue += qty_sold * session_state["product_prices"][product_name]
            
            if qty_sold > 0:
                try:
                    price = session_state.get("product_prices", {}).get(product_name)
                    if price is not None:
                        obs = SalesObservation(
                            product_name=product_name,
                            price=float(price),
                            sold_qty=int(qty_sold),
                            day_info=di,
                        )
                        self.sales.update_demand_params(obs)
                except Exception:
                    pass
        
        session_state["money"] = session_state.get("money", 0) + revenue
        
        sales_history = session_state.setdefault("sales_history", [])
        sales_history.append({
            "day": day,
            "sold": sold,
            "revenue": revenue,
        })
        
        return json.dumps(sold, ensure_ascii=False, indent=2)

    def update_demand_params(self, session_state: Dict[str, Any], product_name: str, sold_qty: int) -> dict:
        """Update demand model based on observed sales data.
        
        Automatically reads price and day info.
        """
        price = session_state.get("product_prices", {}).get(product_name)
        if price is None:
            raise ValueError(f"Price for product '{product_name}' not found in session_state")
        
        day = session_state.get("day", 0)
        di = DayInfo(
            date=f"2025-10-{day + 1:02d}",
            dow=day % 7,
            month=10,
            weather="sunny",
            location_factor=1.0,
        )
        
        obs = SalesObservation(
            product_name=product_name,
            price=float(price),
            sold_qty=int(sold_qty),
            day_info=di,
        )
        
        self.sales.update_demand_params(obs)
        
        p = self.sales.get_demand_params(product_name)
        return {"product_name": product_name, "eps": p.eps, "p_ref": p.p_ref, "b_base": p.b_base}

