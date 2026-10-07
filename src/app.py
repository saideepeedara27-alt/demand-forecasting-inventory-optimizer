from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    FEATURE_IMPORTANCE_PATH,
    FORECAST_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    RECOMMENDATIONS_PATH,
    SKU_ANALYSIS_PATH,
)


st.set_page_config(
    page_title="Demand Forecasting and Inventory Optimization",
    layout="wide",
)


@st.cache_data
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    recommendations = pd.read_csv(RECOMMENDATIONS_PATH)
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    sku_analysis = pd.read_csv(SKU_ANALYSIS_PATH)
    model_comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    return forecast, recommendations, feature_importance, sku_analysis, model_comparison, metrics


forecast_df, recommendations_df, feature_importance_df, sku_analysis_df, model_comparison_df, metrics = load_outputs()

st.title("AI Demand Forecasting and Inventory Optimization")

st.subheader("Project Snapshot")
snapshot_cols = st.columns(4)
snapshot_cols[0].metric("Dataset", metrics.get("dataset_label", "unknown"))
snapshot_cols[1].metric("Validation", "Time-based")
snapshot_cols[2].metric("Primary Model", metrics.get("model", "Random Forest"))
snapshot_cols[3].metric("Inventory Data", metrics.get("inventory_source", "unknown"))

overview_cols = st.columns(3)
with overview_cols[0]:
    st.markdown("**Business Problem**")
    st.write("Forecast demand early enough to reduce stockouts and overstocking.")
with overview_cols[1]:
    st.markdown("**ML Approach**")
    st.write("Lag and rolling demand features with Random Forest regression.")
with overview_cols[2]:
    st.markdown("**Business Output**")
    st.write("Safety stock, reorder point, reorder quantity, risk, and SKU priority.")

metric_cols = st.columns(4)
metric_cols[0].metric("MAE", metrics["mae"])
metric_cols[1].metric("RMSE", metrics["rmse"])
metric_cols[2].metric("MAPE", f"{metrics['mape']}%")
metric_cols[3].metric(
    "MAE Lift vs Baseline",
    f"{metrics['mae_improvement_vs_baseline_pct']}%",
)

st.subheader("Model Comparison")
st.dataframe(model_comparison_df, width="stretch", hide_index=True)

comparison_chart = px.bar(
    model_comparison_df.sort_values("mae"),
    x="model",
    y="mae",
    color="model",
    labels={"model": "Model", "mae": "MAE"},
)
st.plotly_chart(comparison_chart, width="stretch")

store_options = sorted(forecast_df["store_id"].unique())
product_options = sorted(forecast_df["product_id"].unique())

filter_cols = st.columns(2)
selected_store = filter_cols[0].selectbox("Store", store_options)
selected_product = filter_cols[1].selectbox("Product", product_options)

filtered_forecast = forecast_df[
    (forecast_df["store_id"] == selected_store)
    & (forecast_df["product_id"] == selected_product)
].copy()

st.subheader("Actual Demand vs Forecasted Demand")
line_data = filtered_forecast.melt(
    id_vars=["date"],
    value_vars=["units_sold", "predicted_units_sold"],
    var_name="series",
    value_name="units",
)
fig = px.line(
    line_data,
    x="date",
    y="units",
    color="series",
    markers=True,
    labels={"date": "Date", "units": "Units", "series": "Series"},
)
st.plotly_chart(fig, width="stretch")

with st.expander("Data preprocessing and validation design"):
    st.markdown(
        """
- Data is sorted by `store_id`, `product_id`, and `date`.
- Validation is time-based: earlier dates train the model, the latest forecast horizon is test data.
- Lag features use previous demand values such as `lag_7`.
- Rolling averages use `shift(1)` so the current day's demand is not leaked into features.
- The weekly baseline predicts demand using the same item-store demand from the previous week.
- `inventory_on_hand` is used only for inventory recommendation logic, not as a model feature.
        """
    )

st.subheader("Inventory Recommendations")
risk_order = ["High", "Medium", "Low"]
selected_risk = st.multiselect("Stockout Risk", risk_order, default=risk_order)
filtered_recommendations = recommendations_df[
    recommendations_df["stockout_risk"].isin(selected_risk)
].sort_values("reorder_quantity", ascending=False)

st.dataframe(
    filtered_recommendations,
    width="stretch",
    hide_index=True,
)

st.subheader("Priority View")
priority_cols = [
    "store_id",
    "product_id",
    "abc_class",
    "stockout_risk",
    "inventory_on_hand",
    "reorder_point",
    "reorder_quantity",
    "lost_sales_risk_value",
    "priority_score",
]
st.dataframe(
    recommendations_df[priority_cols].sort_values("priority_score", ascending=False).head(15),
    width="stretch",
    hide_index=True,
)

st.subheader("Stockout Risk Summary")
risk_summary = (
    recommendations_df["stockout_risk"]
    .value_counts()
    .rename_axis("stockout_risk")
    .reset_index(name="sku_store_count")
)
bar = px.bar(
    risk_summary,
    x="stockout_risk",
    y="sku_store_count",
    color="stockout_risk",
    category_orders={"stockout_risk": risk_order},
    labels={"stockout_risk": "Stockout Risk", "sku_store_count": "Product-Store Count"},
)
st.plotly_chart(bar, width="stretch")

left, right = st.columns(2)

with left:
    st.subheader("ABC Product Classification")
    abc_chart = px.bar(
        sku_analysis_df,
        x="product_id",
        y="total_revenue",
        color="abc_class",
        labels={"product_id": "Product", "total_revenue": "Revenue", "abc_class": "ABC Class"},
    )
    st.plotly_chart(abc_chart, width="stretch")

with right:
    st.subheader("Top Forecast Drivers")
    importance_chart = px.bar(
        feature_importance_df.head(10).sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    st.plotly_chart(importance_chart, width="stretch")

st.subheader("Promotion and Price Scenario")
scenario_cols = st.columns(3)
price_change = scenario_cols[0].slider("Price Change", -20, 20, 0, format="%d%%")
promotion_lift = scenario_cols[1].slider("Promotion Lift", 0, 60, 25, format="%d%%")
run_promotion = scenario_cols[2].toggle("Promotion Active")

base_forecast = filtered_forecast["predicted_units_sold"].sum()
price_elasticity = -0.8
price_effect = 1 + (price_elasticity * price_change / 100)
promo_effect = 1 + (promotion_lift / 100 if run_promotion else 0)
scenario_demand = max(0, base_forecast * price_effect * promo_effect)

scenario_metric_cols = st.columns(3)
scenario_metric_cols[0].metric("Base Forecast Demand", f"{base_forecast:.0f} units")
scenario_metric_cols[1].metric("Scenario Demand", f"{scenario_demand:.0f} units")
scenario_metric_cols[2].metric("Demand Change", f"{scenario_demand - base_forecast:.0f} units")
