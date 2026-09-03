import numpy as np


class TabularBaseline:
    """Histogram gradient boosting baseline with class-balanced sample weights."""

    def __init__(self, random_state: int = 42, max_iter: int = 200):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=max_iter,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "TabularBaseline":
        counts = np.bincount(labels.astype(int), minlength=2)
        weights = np.ones(len(labels), dtype=np.float64)
        for label in (0, 1):
            if counts[label]:
                weights[labels == label] = len(labels) / (2 * counts[label])
        self.model.fit(features, labels, sample_weight=weights)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]

