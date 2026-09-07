import unittest

import numpy as np
import pandas as pd

from src.data.preprocessing import TabularPreprocessor
from src.data.temporal_split import temporal_split
from src.data.loaders import add_paysim_causal_features
from src.graph.temporal_graph import build_temporal_entity_graph


class DataPipelineTests(unittest.TestCase):
    def test_split_keeps_timestamp_ties_together(self):
        frame = pd.DataFrame({'time': [1]*4 + [2]*3 + [3]*3 + [4]*2})
        result = temporal_split(frame, 'time', .5, .25)
        self.assertLess(result.train.time.max(), result.validation.time.min())
        self.assertLess(result.validation.time.max(), result.test.time.min())
        self.assertEqual(sum(map(len, [result.train, result.validation, result.test])), len(frame))

    def test_explicit_temporal_boundaries(self):
        frame = pd.DataFrame({'time': [1, 2, 2, 3, 4, 5]})
        result = temporal_split(frame, 'time', .7, .15, 3, 5)
        self.assertEqual(result.train.time.tolist(), [1, 2, 2])
        self.assertEqual(result.validation.time.tolist(), [3, 4])
        self.assertEqual(result.test.time.tolist(), [5])

    def test_graph_excludes_ties_and_keeps_bounded_past(self):
        frame = pd.DataFrame({'time': [1, 2, 2, 3], 'entity': ['A']*4})
        graph = build_temporal_entity_graph(frame, 'time', [['entity']], 100, 1)
        self.assertEqual(set(map(tuple, graph.edge_index.T.tolist())), {(0, 1), (0, 2), (2, 3)})

    def test_history_cannot_observe_same_step(self):
        frame = pd.DataFrame({'step': [1, 2, 2, 3], 'amount': [10., 20., 900., 5.], 'nameOrig': ['A']*4, 'nameDest': ['B']*4})
        result = add_paysim_causal_features(frame)
        np.testing.assert_allclose(result.origin_previous_amount_sum_log1p, np.log1p([0, 10, 10, 930]))
        np.testing.assert_allclose(result.origin_previous_count_log1p, np.log1p([0, 1, 1, 3]))

    def test_log_scaling_handles_infinite_and_extreme_values(self):
        processor = TabularPreprocessor(set(), log_columns=('amount',), clip_value=3.)
        processor.fit(pd.DataFrame({'amount': [0., 1., 2.]}))
        result = processor.transform(pd.DataFrame({'amount': [np.inf, -np.inf, 1e30]}))
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(float(np.abs(result).max()), 3.)

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


def test_window_without_row_limit_does_not_load_outside_period(tmp_path):
    from src.data.loaders import _read_csv

    path = tmp_path / 'example.csv'
    pd.DataFrame({'step': [1, 2, 3, 4, 5]}).to_csv(path, index=False)
    result = _read_csv(path, None, 'window', 5, 42, 'step', {'min': 2, 'max': 4})
    assert result.step.tolist() == [2, 3, 4]
