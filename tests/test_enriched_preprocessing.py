import numpy as np
import pandas as pd

from src.data.preprocessing import TabularPreprocessor


def test_category_identity_is_preserved_when_training_frequencies_are_equal():
    train = pd.DataFrame({"type": ["TRANSFER", "TRANSFER", "CASH_OUT", "CASH_OUT"]})
    processor = TabularPreprocessor(set(), categorical_encoding="onehot_frequency")
    features = processor.fit_transform(train)
    assert features[0, 0] == features[2, 0] == 0.5
    assert not np.array_equal(features[0], features[2])
    assert len(processor.output_columns) == features.shape[1]


def test_unknown_validation_categories_do_not_change_training_schema():
    processor = TabularPreprocessor(set(), categorical_encoding="onehot_frequency", max_categories=1)
    processor.fit(pd.DataFrame({"type": ["A", "A", "B"]}))
    columns = processor.output_columns.copy()
    encoded = processor.transform(pd.DataFrame({"type": ["UNSEEN", "A"]}))
    np.testing.assert_array_equal(encoded[0], [0, 0])
    assert processor.output_columns == columns
    assert "UNSEEN" not in processor.frequency_maps["type"]


def test_imputed_value_and_missing_value_have_different_flags():
    processor = TabularPreprocessor(set(), add_missing_indicators=True)
    processor.fit(pd.DataFrame({"amount": [1.0, np.nan, 3.0]}))
    encoded = processor.transform(pd.DataFrame({"amount": [2.0, np.nan, np.inf]}))
    np.testing.assert_array_equal(encoded[:, 0], [0, 0, 0])
    np.testing.assert_array_equal(encoded[:, 1], [0, 1, 1])
    assert processor.output_columns == ["amount", "amount__missing"]


def test_legacy_preprocessor_without_new_fields_still_transforms():
    processor = TabularPreprocessor(set()).fit(pd.DataFrame({"amount": [1.0, 3.0], "type": ["A", "B"]}))
    expected = processor.transform(pd.DataFrame({"amount": [2.0], "type": ["A"]}))
    del processor.missing_indicator_columns
    np.testing.assert_array_equal(processor.transform(pd.DataFrame({"amount": [2.0], "type": ["A"]})), expected)
