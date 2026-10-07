from __future__ import annotations

import unittest

import pandas as pd

from src.inventory import build_inventory_recommendations


class InventoryRecommendationTest(unittest.TestCase):
    def test_reorder_point_and_quantity_are_calculated_from_forecast(self) -> None:
        forecast = pd.DataFrame(
            {
                "store_id": ["S001", "S001", "S001"],
                "product_id": ["P001", "P001", "P001"],
                "predicted_units_sold": [10.0, 12.0, 14.0],
            }
        )
        latest_inventory = pd.DataFrame(
            {
                "store_id": ["S001"],
                "product_id": ["P001"],
                "inventory_on_hand": [20],
            }
        )

        result = build_inventory_recommendations(
            forecast,
            latest_inventory,
            lead_time_days=2,
            service_level_z=1.65,
        )

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0, "avg_daily_forecast"], 12.0)
        self.assertAlmostEqual(result.loc[0, "lead_time_demand"], 24.0)
        self.assertAlmostEqual(result.loc[0, "safety_stock"], 4.67)
        self.assertAlmostEqual(result.loc[0, "reorder_point"], 28.67)
        self.assertAlmostEqual(result.loc[0, "reorder_quantity"], 8.67)
        self.assertEqual(result.loc[0, "stockout_risk"], "High")


if __name__ == "__main__":
    unittest.main()
