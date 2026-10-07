from __future__ import annotations

import pandas as pd


GROUP_COLUMNS = ["store_id", "product_id"]
TARGET_COLUMN = "units_sold"
OPTIONAL_FEATURE_COLUMNS = [
    "price",
    "promotion",
    "holiday",
    "competitor_price",
]


def load_demand_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])
    if "category" not in data.columns:
        data["category"] = "Unknown"
    if "inventory_on_hand" not in data.columns:
        data["inventory_on_hand"] = 0
    return data.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    featured = data.copy()
    featured["day_of_week"] = featured["date"].dt.dayofweek
    featured["month"] = featured["date"].dt.month
    featured["is_weekend"] = (featured["day_of_week"] >= 5).astype(int)
    featured["day_of_year"] = featured["date"].dt.dayofyear
    return featured


def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    featured = data.copy()
    grouped = featured.groupby(GROUP_COLUMNS, sort=False)[TARGET_COLUMN]

    for lag in [1, 7, 14, 28]:
        featured[f"lag_{lag}"] = grouped.shift(lag)

    for window in [7, 14, 28]:
        # shift(1) prevents leakage by excluding the current day's demand.
        shifted = grouped.shift(1)
        featured[f"rolling_mean_{window}"] = shifted.groupby(
            [featured["store_id"], featured["product_id"]]
        ).rolling(window).mean().reset_index(level=[0, 1], drop=True)

    return featured


def prepare_model_data(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    featured = add_time_features(data)
    featured = add_lag_features(featured)
    featured = featured.dropna().reset_index(drop=True)

    feature_columns = [
        column for column in OPTIONAL_FEATURE_COLUMNS if column in featured.columns
    ] + [
        "day_of_week",
        "month",
        "is_weekend",
        "day_of_year",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
    ]

    return featured, feature_columns
