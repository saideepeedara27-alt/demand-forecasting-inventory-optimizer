from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, REAL_CANONICAL_DATA_PATH


def prepare_store_item(input_path: Path, output_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(input_path)
    required_columns = {"date", "store", "item", "sales"}
    missing_columns = required_columns - set(raw.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Store Item dataset is missing required columns: {missing}")

    data = raw.rename(
        columns={
            "store": "store_id",
            "item": "product_id",
            "sales": "units_sold",
        }
    )
    data["date"] = pd.to_datetime(data["date"])
    data["store_id"] = "S" + data["store_id"].astype(str).str.zfill(2)
    data["product_id"] = "I" + data["product_id"].astype(str).str.zfill(3)
    data["category"] = "StoreItem"

    # The Kaggle Store Item dataset has sales only, not price or inventory.
    # Revenue and inventory outputs use documented assumptions downstream.
    data["price"] = 1.0
    data["promotion"] = 0
    data["holiday"] = 0
    data["competitor_price"] = 1.0
    data["inventory_on_hand"] = 0

    output_columns = [
        "date",
        "store_id",
        "product_id",
        "category",
        "units_sold",
        "price",
        "promotion",
        "holiday",
        "competitor_price",
        "inventory_on_hand",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data[output_columns].to_csv(output_path, index=False)
    return data[output_columns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert manually downloaded Kaggle datasets into the project schema."
    )
    parser.add_argument(
        "--source",
        choices=["store_item"],
        required=True,
        help="Dataset format to convert. Use 'store_item' for Kaggle Store Item Demand Forecasting.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the manually downloaded Kaggle CSV, for example data/real/store_item_train.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(REAL_CANONICAL_DATA_PATH),
        help="Output path for the converted project-ready CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.source == "store_item":
        data = prepare_store_item(input_path, output_path)
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    print(f"Prepared {len(data):,} rows")
    print(f"Saved converted dataset to {output_path}")


if __name__ == "__main__":
    main()
