# Dataset Strategy

This project keeps generated retail demand data only as demo/test data so the full pipeline can be built, tested, and explained without external dependency issues.

For resume submission, run the pipeline on one of the following real datasets and update the README metrics.

## Option 1: Store Item Demand Forecasting

Best for a clean placement project.

Source: Kaggle Store Item Demand Forecasting Challenge

Why it fits:

- Direct product-store-date demand forecasting problem
- Simple enough to explain in interviews
- Good match for inventory planning
- Easy to adapt to the current project structure

Expected columns:

- date
- store
- item
- sales

How it maps to this project:

- `date` -> `date`
- `store` -> `store_id`
- `item` -> `product_id`
- `sales` -> `units_sold`

Best use:

- Use this when you want a clean, reliable version for GitHub and resume.

Current adapter:

```bash
python -m src.prepare_real_data --source store_item --input data/real/store_item_train.csv
python -m src.train_model --data data/retail_demand_real.csv --dataset-label kaggle_store_item
```

## Option 2: Rossmann Store Sales

Best for showing business features like promotions and competition.

Source: Kaggle Rossmann Store Sales

Why it fits:

- Includes store-level sales
- Includes promotion information
- Includes competitor-related data
- Stronger business story than a simple sales-only dataset

Important fields:

- Store
- Date
- Sales
- Promo
- SchoolHoliday
- CompetitionDistance

Best use:

- Use this when you want to explain promotion impact, competitor pressure, and store-level planning.

## Option 3: M5 Forecasting Accuracy

Best for an advanced version.

Source: Kaggle M5 Forecasting Accuracy

Why it fits:

- Real Walmart hierarchical retail sales data
- Product, store, state, category, and department hierarchy
- Standard forecasting benchmark
- Strongest dataset for a serious supply-chain forecasting project

Tradeoff:

- More complex data format
- Heavier preprocessing
- Takes more time to explain and compute

Best use:

- Use this after the first version is complete and stable.

## Recommendation

For placement shortlisting, use this order:

1. Keep the generated-data version only as a demo/test pipeline.
2. Replace the generated dataset with Store Item Demand Forecasting for resume metrics.
3. Add Rossmann or M5 only if there is enough time before submission.

This avoids the common mistake of spending all the time on data cleaning and never finishing a polished project.
