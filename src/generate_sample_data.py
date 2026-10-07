from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import DATA_DIR, RANDOM_STATE, RAW_DATA_PATH


@dataclass(frozen=True)
class Product:
    product_id: str
    category: str
    base_demand: float
    base_price: float


PRODUCTS = [
    Product("P001", "Beverages", 95, 55),
    Product("P002", "Beverages", 70, 42),
    Product("P003", "Snacks", 115, 35),
    Product("P004", "Snacks", 90, 28),
    Product("P005", "Personal Care", 60, 120),
    Product("P006", "Personal Care", 45, 180),
    Product("P007", "Home Care", 50, 160),
    Product("P008", "Home Care", 40, 220),
]

STORES = ["S001", "S002", "S003", "S004", "S005"]
STORE_MULTIPLIER = {
    "S001": 1.15,
    "S002": 1.00,
    "S003": 0.85,
    "S004": 1.25,
    "S005": 0.75,
}


def generate_dataset(days: int = 730) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    rows = []

    for date in dates:
        day_of_week = date.dayofweek
        is_weekend = int(day_of_week >= 5)
        holiday = int((date.month, date.day) in {(1, 1), (8, 15), (10, 2), (12, 25)})
        annual_seasonality = 1 + 0.12 * np.sin(2 * np.pi * date.dayofyear / 365)
        weekly_seasonality = 1.18 if is_weekend else 1.0

        for store_id in STORES:
            store_factor = STORE_MULTIPLIER[store_id]

            for product in PRODUCTS:
                promotion = int(rng.random() < 0.12)
                promo_boost = 1.35 if promotion else 1.0
                holiday_boost = 1.25 if holiday else 1.0
                trend = 1 + 0.00035 * (date - dates.min()).days
                noise = rng.normal(0, product.base_demand * 0.08)

                price = product.base_price * rng.normal(1, 0.035)
                competitor_price = product.base_price * rng.normal(1.02, 0.055)
                price_effect = 1 - max((price - product.base_price) / product.base_price, -0.2) * 0.6

                expected_units = (
                    product.base_demand
                    * store_factor
                    * annual_seasonality
                    * weekly_seasonality
                    * promo_boost
                    * holiday_boost
                    * trend
                    * price_effect
                    + noise
                )
                units_sold = max(0, int(round(expected_units)))
                inventory_on_hand = max(
                    units_sold + rng.integers(10, 160),
                    int(product.base_demand * store_factor * rng.uniform(1.2, 2.8)),
                )

                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "store_id": store_id,
                        "product_id": product.product_id,
                        "category": product.category,
                        "units_sold": units_sold,
                        "price": round(price, 2),
                        "promotion": promotion,
                        "holiday": holiday,
                        "competitor_price": round(competitor_price, 2),
                        "inventory_on_hand": int(inventory_on_hand),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = generate_dataset()
    data.to_csv(RAW_DATA_PATH, index=False)
    print(f"Generated {len(data):,} rows at {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
