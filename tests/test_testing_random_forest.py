import unittest
import pandas as pd
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from TradeSim.testing_random_forest import create_buy_heap


class TestCreateBuyHeap(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            "score": [0.9, 0.8, 0.7],
            "allocated_cash": [1000, 800, 600],
            "current_price": [100, 80, 60],
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "strategy_name": ["strategy1", "strategy2", "strategy3"],
            "quantity": [10, 10, 10],
        }
        self.buy_df = pd.DataFrame(data)

    def test_create_buy_heap(self):
        buy_heap = create_buy_heap(self.buy_df)
        # Update the expected_heap according to the new structure
        expected_heap = [
            (-0.9, 10, "AAPL", 100, "strategy1"),
            (-0.8, 10, "MSFT", 80, "strategy2"),
            (-0.7, 10, "GOOGL", 60, "strategy3"),
        ]
        self.assertEqual(buy_heap, expected_heap)

    def test_create_buy_heap_empty(self):
        empty_df = pd.DataFrame(
            columns=[
                "score",
                "allocated_cash",
                "current_price",
                "ticker",
                "strategy_name",
                "quantity",
            ]
        )
        buy_heap = create_buy_heap(empty_df)
        self.assertEqual(buy_heap, [])


if __name__ == "__main__":
    unittest.main()
