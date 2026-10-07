# Project Report

## Project Name

AI Demand Forecasting and Inventory Optimization Dashboard

## Objective

The objective is to forecast short-term product demand at a store-product level and convert the forecast into inventory actions such as reorder quantity, safety stock, stockout risk, and priority score.

## Business Problem

Retail and FMCG supply chains often face a planning tradeoff:

- Low inventory creates stockouts and lost sales.
- High inventory increases holding cost and working capital pressure.

The project addresses this by using machine learning to improve demand visibility and then translating predictions into clear replenishment recommendations.

## Solution Overview

The solution has four layers:

1. Data preparation and feature engineering
2. Demand forecasting model
3. Inventory decision logic
4. Business dashboard

## Model Summary

The primary model is Random Forest because it handles non-linear demand behavior, is robust with limited tuning, and provides feature importance. The pipeline also compares against LightGBM and a weekly baseline.

Key feature groups:

- Lag demand features
- Rolling demand averages
- Calendar features
- Optional promotion, holiday, price, and competitor price when available

Inventory is not used as a model feature. It is used after forecasting to calculate reorder logic.

## Latest Demo Results

| Metric | Value |
| --- | ---: |
| MAE | 4.964 |
| RMSE | 6.642 |
| MAPE | 6.080% |
| Baseline MAE | 11.26 |
| Improvement vs baseline | 55.931% |

These results are from demo/generated data and should be replaced with real Kaggle metrics before resume submission.

## Business Outputs

The project produces:

- Demand forecast by date, store, and product
- Safety stock
- Reorder point
- Reorder quantity
- Stockout risk level
- Lost-sales risk value
- ABC product class
- Priority score

## Key Insight

The strongest demand drivers in the first model run were rolling historical demand averages, promotion, and weekly seasonality. This is realistic for retail forecasting because recent demand and promotional activity usually have strong influence on near-term demand.

## Why This Is Resume-Relevant

This project demonstrates the full journey from raw data to business decision:

- Data science: preprocessing, feature engineering, modeling, evaluation
- ML concepts: supervised learning, time-series features, baseline comparison
- Business analytics: inventory decisions, stockout risk, ABC classification
- Communication: dashboard and clear executive outputs
