from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    FEATURE_IMPORTANCE_PATH,
    FORECAST_HORIZON_DAYS,
    FORECAST_PATH,
    LEAD_TIME_DAYS,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_COMPARISON_PATH,
    OUTPUT_DIR,
    RAW_DATA_PATH,
    RECOMMENDATIONS_PATH,
    RANDOM_STATE,
    SKU_ANALYSIS_PATH,
)
from src.business_analysis import add_business_priority, build_sku_abc_analysis
from src.features import TARGET_COLUMN, load_demand_data, prepare_model_data
from src.inventory import build_inventory_recommendations


def mean_absolute_percentage_error(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true_safe = y_true.replace(0, np.nan)
    return float((np.abs((y_true_safe - y_pred) / y_true_safe).dropna()).mean() * 100)


def split_train_test(data: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_date = data["date"].max() - pd.Timedelta(days=horizon_days)
    train = data[data["date"] <= cutoff_date].copy()
    test = data[data["date"] > cutoff_date].copy()
    return train, test


def build_latest_inventory(raw_data: pd.DataFrame, forecast: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    latest_inventory = (
        raw_data.sort_values("date")
        .groupby(["store_id", "product_id"], as_index=False)
        .tail(1)[["store_id", "product_id", "inventory_on_hand"]]
    )
    if latest_inventory["inventory_on_hand"].gt(0).any():
        latest_inventory["inventory_source"] = "provided"
        return latest_inventory, "provided"

    estimated_inventory = (
        forecast.groupby(["store_id", "product_id"], as_index=False)
        .agg(avg_daily_forecast=("predicted_units_sold", "mean"))
    )
    estimated_inventory["inventory_on_hand"] = (
        estimated_inventory["avg_daily_forecast"] * LEAD_TIME_DAYS * 0.75
    ).round().astype(int)
    estimated_inventory["inventory_source"] = "estimated_from_forecast"
    return estimated_inventory[["store_id", "product_id", "inventory_on_hand", "inventory_source"]], "estimated_from_forecast"


def compare_models(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[RandomForestRegressor, pd.DataFrame, np.ndarray]:
    baseline_predictions = test_data["lag_7"].clip(lower=0).to_numpy()
    rows = [
        {
            "model": "Weekly baseline",
            "mae": mean_absolute_error(test_data[TARGET_COLUMN], baseline_predictions),
            "rmse": np.sqrt(mean_squared_error(test_data[TARGET_COLUMN], baseline_predictions)),
            "mape": mean_absolute_percentage_error(test_data[TARGET_COLUMN], baseline_predictions),
        }
    ]

    random_forest = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    random_forest.fit(train_data[feature_columns], train_data[TARGET_COLUMN])
    rf_predictions = random_forest.predict(test_data[feature_columns]).clip(min=0)
    rows.append(
        {
            "model": "Random Forest",
            "mae": mean_absolute_error(test_data[TARGET_COLUMN], rf_predictions),
            "rmse": np.sqrt(mean_squared_error(test_data[TARGET_COLUMN], rf_predictions)),
            "mape": mean_absolute_percentage_error(test_data[TARGET_COLUMN], rf_predictions),
        }
    )

    gradient_boosting = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    gradient_boosting.fit(train_data[feature_columns], train_data[TARGET_COLUMN])
    gb_predictions = gradient_boosting.predict(test_data[feature_columns]).clip(min=0)
    rows.append(
        {
            "model": "Gradient Boosting (sklearn)",
            "mae": mean_absolute_error(test_data[TARGET_COLUMN], gb_predictions),
            "rmse": np.sqrt(mean_squared_error(test_data[TARGET_COLUMN], gb_predictions)),
            "mape": mean_absolute_percentage_error(test_data[TARGET_COLUMN], gb_predictions),
        }
    )

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        pass
    else:
        lightgbm = LGBMRegressor(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        lightgbm.fit(train_data[feature_columns], train_data[TARGET_COLUMN])
        lgbm_predictions = lightgbm.predict(test_data[feature_columns]).clip(min=0)
        rows.append(
            {
                "model": "LightGBM",
                "mae": mean_absolute_error(test_data[TARGET_COLUMN], lgbm_predictions),
                "rmse": np.sqrt(mean_squared_error(test_data[TARGET_COLUMN], lgbm_predictions)),
                "mape": mean_absolute_percentage_error(test_data[TARGET_COLUMN], lgbm_predictions),
            }
        )

    comparison = pd.DataFrame(rows)
    baseline_mae = comparison.loc[comparison["model"] == "Weekly baseline", "mae"].iloc[0]
    comparison["mae_improvement_vs_baseline_pct"] = (
        (baseline_mae - comparison["mae"]) / baseline_mae * 100
    )
    comparison = comparison.round(
        {"mae": 3, "rmse": 3, "mape": 3, "mae_improvement_vs_baseline_pct": 3}
    )
    return random_forest, comparison, rf_predictions


def train(data_path: Path = RAW_DATA_PATH, dataset_label: str = "demo_generated") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_demand_data(str(data_path))
    model_data, feature_columns = prepare_model_data(raw_data)
    train_data, test_data = split_train_test(model_data, FORECAST_HORIZON_DAYS)

    model, model_comparison, predictions = compare_models(train_data, test_data, feature_columns)
    rf_row = model_comparison[model_comparison["model"] == "Random Forest"].iloc[0]
    baseline_row = model_comparison[model_comparison["model"] == "Weekly baseline"].iloc[0]
    metrics = {
        "dataset_label": dataset_label,
        "data_path": str(data_path),
        "mae": float(rf_row["mae"]),
        "rmse": float(rf_row["rmse"]),
        "mape": float(rf_row["mape"]),
        "baseline_mae": float(baseline_row["mae"]),
        "mae_improvement_vs_baseline_pct": float(rf_row["mae_improvement_vs_baseline_pct"]),
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "model": "Random Forest",
        "feature_columns": feature_columns,
    }

    forecast = test_data[
        ["date", "store_id", "product_id", "category", TARGET_COLUMN, "inventory_on_hand"]
    ].copy()
    forecast["predicted_units_sold"] = predictions.round(2)
    forecast["absolute_error"] = (forecast[TARGET_COLUMN] - forecast["predicted_units_sold"]).abs().round(2)

    latest_inventory, inventory_source = build_latest_inventory(raw_data, forecast)
    metrics["inventory_source"] = inventory_source
    sku_analysis = build_sku_abc_analysis(raw_data)
    recommendations = build_inventory_recommendations(forecast, latest_inventory)
    recommendations = add_business_priority(recommendations, sku_analysis)

    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .round({"importance": 4})
    )

    joblib.dump({"model": model, "feature_columns": feature_columns}, MODEL_PATH)
    forecast.to_csv(FORECAST_PATH, index=False)
    recommendations.to_csv(RECOMMENDATIONS_PATH, index=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    sku_analysis.to_csv(SKU_ANALYSIS_PATH, index=False)
    model_comparison.to_csv(MODEL_COMPARISON_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Training complete")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved forecast: {FORECAST_PATH}")
    print(f"Saved recommendations: {RECOMMENDATIONS_PATH}")
    print(f"Saved SKU analysis: {SKU_ANALYSIS_PATH}")
    print(f"Saved model comparison: {MODEL_COMPARISON_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train demand forecasting models.")
    parser.add_argument(
        "--data",
        default=str(RAW_DATA_PATH),
        help="Path to project-ready demand data CSV.",
    )
    parser.add_argument(
        "--dataset-label",
        default="demo_generated",
        help="Label written to metrics, for example demo_generated or kaggle_store_item.",
    )
    args = parser.parse_args()
    train(Path(args.data), args.dataset_label)
