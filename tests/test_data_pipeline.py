import unittest

import numpy as np
import pandas as pd

from src.data.preprocessing import TabularPreprocessor
from src.data.temporal_split import temporal_split
from src.data.loaders import add_paysim_causal_features
from src.graph.temporal_graph import build_temporal_entity_graph


class DataPipelineTests(unittest.TestCase):
    def test_temporal_split_preserves_order(self):
        frame = pd.DataFrame({"time": [5, 1, 4, 2, 3], "label": [0, 1, 0, 1, 0]})
        result = temporal_split(frame, "time", 0.6, 0.2)
        self.assertEqual(result.train["time"].tolist(), [1, 2, 3])
        self.assertEqual(result.validation["time"].tolist(), [4])
        self.assertEqual(result.test["time"].tolist(), [5])

    def test_preprocessor_uses_train_statistics(self):
        train = pd.DataFrame({"number": [1.0, 2.0, np.nan], "category": ["a", "a", "b"], "target": [0, 0, 1]})
        test = pd.DataFrame({"number": [100.0], "category": ["unseen"], "target": [1]})
        processor = TabularPreprocessor({"target"})
        train_array = processor.fit_transform(train)
        test_array = processor.transform(test)
        self.assertEqual(train_array.shape, (3, 2))
        self.assertEqual(test_array.shape, (1, 2))
        self.assertEqual(float(test_array[0, 1]), 0.0)

    def test_graph_edges_only_point_forward_in_time(self):
        frame = pd.DataFrame({"time": [0, 10, 20, 100], "entity": ["A", "A", "A", "A"]})
        graph = build_temporal_entity_graph(frame, "time", [["entity"]], window_seconds=30, max_previous_neighbors=2)
        edges = set(map(tuple, graph.edge_index.T.tolist()))
        self.assertEqual(edges, {(0, 1), (0, 2), (1, 2)})
        self.assertTrue(all(source < destination for source, destination in edges))

    def test_paysim_history_features_do_not_use_future_rows(self):
        frame = pd.DataFrame(
            {
                "step": [1, 2, 3],
                "type": ["TRANSFER", "TRANSFER", "TRANSFER"],
                "amount": [10.0, 20.0, 999.0],
                "nameOrig": ["A", "A", "A"],
                "nameDest": ["B", "B", "B"],
            }
        )
        original = add_paysim_causal_features(frame)
        changed = frame.copy()
        changed.loc[2, "amount"] = 1_000_000.0
        recalculated = add_paysim_causal_features(changed)
        historical = [column for column in original.columns if "previous" in column or "is_new" in column]
        pd.testing.assert_frame_equal(original.loc[:1, historical], recalculated.loc[:1, historical])
        self.assertEqual(original["origin_previous_count_log1p"].iloc[0], 0.0)
        self.assertGreater(original["origin_previous_count_log1p"].iloc[2], 0.0)


if __name__ == "__main__":
    unittest.main()
