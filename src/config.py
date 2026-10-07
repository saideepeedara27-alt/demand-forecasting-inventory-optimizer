from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REAL_DATA_DIR = DATA_DIR / "real"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_DATA_PATH = DATA_DIR / "sample_retail_demand.csv"
REAL_CANONICAL_DATA_PATH = DATA_DIR / "retail_demand_real.csv"
MODEL_PATH = OUTPUT_DIR / "model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
FORECAST_PATH = OUTPUT_DIR / "demand_forecast.csv"
RECOMMENDATIONS_PATH = OUTPUT_DIR / "inventory_recommendations.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
SKU_ANALYSIS_PATH = OUTPUT_DIR / "sku_abc_analysis.csv"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"

RANDOM_STATE = 42
FORECAST_HORIZON_DAYS = 14
LEAD_TIME_DAYS = 7
SERVICE_LEVEL_Z = 1.65
