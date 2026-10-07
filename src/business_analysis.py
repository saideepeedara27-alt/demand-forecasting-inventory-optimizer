from __future__ import annotations

import numpy as np
import pandas as pd


def build_sku_abc_analysis(data: pd.DataFrame) -> pd.DataFrame:
    sku_summary = (
        data.assign(revenue=data["units_sold"] * data["price"])
        .groupby(["product_id", "category"], as_index=False)
        .agg(
            total_units=("units_sold", "sum"),
            total_revenue=("revenue", "sum"),
            avg_price=("price", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    sku_summary["revenue_share"] = sku_summary["total_revenue"] / sku_summary["total_revenue"].sum()
    sku_summary["cumulative_revenue_share"] = sku_summary["revenue_share"].cumsum()
    sku_summary["abc_class"] = np.select(
        [
            sku_summary["cumulative_revenue_share"] <= 0.70,
            sku_summary["cumulative_revenue_share"] <= 0.90,
        ],
        ["A", "B"],
        default="C",
    )
    sku_summary[["total_revenue", "avg_price", "revenue_share", "cumulative_revenue_share"]] = (
        sku_summary[["total_revenue", "avg_price", "revenue_share", "cumulative_revenue_share"]].round(3)
    )
    return sku_summary


def add_business_priority(recommendations: pd.DataFrame, sku_analysis: pd.DataFrame) -> pd.DataFrame:
    abc_weight = {"A": 3, "B": 2, "C": 1}
    risk_weight = {"High": 3, "Medium": 2, "Low": 1}

    enriched = recommendations.merge(
        sku_analysis[["product_id", "abc_class", "avg_price"]],
        on="product_id",
        how="left",
    )
    enriched["lost_sales_risk_units"] = (
        enriched["lead_time_demand"] - enriched["inventory_on_hand"]
    ).clip(lower=0)
    enriched["lost_sales_risk_value"] = enriched["lost_sales_risk_units"] * enriched["avg_price"]
    max_risk_value = max(float(enriched["lost_sales_risk_value"].max()), 1.0)
    enriched["priority_score"] = (
        enriched["abc_class"].map(abc_weight).fillna(1)
        * enriched["stockout_risk"].map(risk_weight).fillna(1)
        * (1 + enriched["lost_sales_risk_value"] / max_risk_value)
    )
    enriched["priority_score"] = enriched["priority_score"].round(2)
    enriched["lost_sales_risk_units"] = enriched["lost_sales_risk_units"].round(2)
    enriched["lost_sales_risk_value"] = enriched["lost_sales_risk_value"].round(2)
    return enriched.sort_values("priority_score", ascending=False)
