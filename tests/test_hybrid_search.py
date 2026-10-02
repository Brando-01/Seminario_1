from explore_hybrid import choose_candidate


def test_candidate_choice_uses_only_validation_ap():
    records = [
        {"candidate": "longer_training", "validation_ap": 0.40, "test_ap": 0.99},
        {"candidate": "longer_training", "validation_ap": 0.42, "test_ap": 0.99},
        {"candidate": "ae_pretrained", "validation_ap": 0.44, "test_ap": 0.01},
        {"candidate": "ae_pretrained", "validation_ap": 0.45, "test_ap": 0.01},
    ]

    assert choose_candidate(records) == "ae_pretrained"
