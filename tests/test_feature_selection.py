import numpy as np
import pytest

from explore_hybrid_features import select_train_features


def test_train_only_feature_ranking_selects_predictive_column():
    labels = np.array([0, 0, 0, 1, 1, 1])
    features = np.column_stack([
        [1, 0, 1, 0, 1, 0],
        [0, 1, 2, 3, 4, 5],
        [7, 7, 7, 7, 7, 7],
    ]).astype(np.float32)

    selected, scores = select_train_features(features, labels, 1)

    assert selected.tolist() == [1]
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == 0


@pytest.mark.parametrize("top_k", [0, 4])
def test_feature_ranking_rejects_invalid_sizes(top_k):
    with pytest.raises(ValueError, match="top_k"):
        select_train_features(np.ones((4, 3), dtype=np.float32), np.array([0, 0, 1, 1]), top_k)
