from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LEAD_TIME_DAYS, SERVICE_LEVEL_Z


def build_inventory_recommendations(
    forecast: pd.DataFrame,
    latest_inventory: pd.DataFrame,
    lead_time_days: int = LEAD_TIME_DAYS,
    service_level_z: float = SERVICE_LEVEL_Z,
) -> pd.DataFrame:
    summary = (
        forecast.groupby(["store_id", "product_id"], as_index=False)
        .agg(
            forecast_horizon_demand=("predicted_units_sold", "sum"),
            avg_daily_forecast=("predicted_units_sold", "mean"),
            forecast_std=("predicted_units_sold", "std"),
        )
        .fillna({"forecast_std": 0})
    )

    summary["lead_time_demand"] = summary["avg_daily_forecast"] * lead_time_days
    summary["safety_stock"] = service_level_z * summary["forecast_std"] * np.sqrt(lead_time_days)
    summary["reorder_point"] = summary["lead_time_demand"] + summary["safety_stock"]

    recommendations = summary.merge(latest_inventory, on=["store_id", "product_id"], how="left")
    recommendations["reorder_quantity"] = (
        recommendations["reorder_point"] - recommendations["inventory_on_hand"]
    ).clip(lower=0)

    recommendations["stockout_risk"] = np.select(
        [
            recommendations["inventory_on_hand"] < recommendations["lead_time_demand"],
            recommendations["inventory_on_hand"] < recommendations["reorder_point"],
        ],
        ["High", "Medium"],
        default="Low",
    )

    numeric_columns = [
        "forecast_horizon_demand",
        "avg_daily_forecast",
        "forecast_std",
        "lead_time_demand",
        "safety_stock",
        "reorder_point",
        "reorder_quantity",
    ]
    recommendations[numeric_columns] = recommendations[numeric_columns].round(2)
    return recommendations.sort_values(["stockout_risk", "reorder_quantity"], ascending=[True, False])
