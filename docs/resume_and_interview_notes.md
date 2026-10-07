# Resume and Interview Notes

## Recommended Project Title

**AI Demand Forecasting and Inventory Optimization Dashboard**

## Best Resume Bullet

Built an AI-based demand forecasting and inventory optimization dashboard using Python, Scikit-learn, LightGBM, Pandas, Streamlit, and Plotly to forecast product-store demand, compare models against a weekly baseline, and convert forecasts into reorder quantity, safety stock, stockout risk, and ABC-based SKU priority recommendations.

Use exact performance numbers only after running the pipeline on a real Kaggle dataset. The current checked-in metrics are from demo/test data and should not be presented as real-world performance.

## Short Resume Version

Built a product-store demand forecasting system using Python and machine learning, converting forecasts into reorder quantity, safety stock, stockout risk, and business priority recommendations through a Streamlit dashboard.

## Strong Resume Version

Developed a supply-chain decision intelligence project for demand forecasting and inventory optimization, using lag/rolling time-series features, Random Forest regression, ABC product classification, feature importance, and price-promotion scenario analysis to support business inventory decisions.

## Skills To Mention

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Machine Learning
- Time-series feature engineering
- Forecasting metrics: MAE, RMSE, MAPE
- Inventory optimization
- Streamlit dashboarding
- Business analytics
- Supply-chain analytics

## 30-Second Interview Pitch

I built this project as a supply-chain analytics solution, not just a prediction model. It forecasts daily demand at a product-store level using lag features, rolling averages, and calendar features, with optional business drivers like price or promotion when available. I used a time-based validation split, compared Random Forest and LightGBM against a weekly baseline, and converted forecasts into business actions such as reorder point, safety stock, reorder quantity, stockout risk, lost-sales risk value, and priority score. I also built a Streamlit dashboard so a business user can explore forecasts and inventory actions.

## Questions You Should Be Ready For

### Why did you choose this project?

Because demand forecasting is a direct business problem in supply chain, sales, and planning. It shows both ML skills and business understanding.

### Why not only show forecast accuracy?

Accuracy alone is not enough for business decisions. I converted forecast output into reorder point, safety stock, reorder quantity, and stockout risk so the model supports actual inventory planning.

### Why compare with a baseline?

A model should prove that it adds value. The weekly baseline represents a simple business rule: predict demand from the same store-product 7 days earlier. I compare every ML model against that baseline.

### What features did you create?

I created date features, lag demand features, and rolling demand averages. When the dataset includes business fields, the pipeline can also use price, promotion, holiday, and competitor price. Inventory is not used as a model feature because it can create leakage; it is used after forecasting for reorder logic.

### What would you improve next?

I would run the project on the manually downloaded Kaggle Store Item dataset, add SHAP for explainability, deploy the dashboard, and add cost-based inventory optimization using holding cost and stockout penalty.

## Resume Bullet After Real Dataset Run

Built an end-to-end demand forecasting and inventory optimization system using Python, Scikit-learn, LightGBM, and Streamlit, predicting store-product demand with time-based validation and reducing MAE by X% against a weekly baseline.

Translated demand forecasts into safety stock, reorder points, stockout risk, lost-sales risk value, and ABC-based SKU prioritization through an interactive dashboard for supply-chain planning.
