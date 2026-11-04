from src.data import ID, TARGET, feature_columns, load_all


def test_real_data_contract():
    train, test, sample = load_all()
    assert train.shape == (699_960, 26)
    assert test.shape == (299_984, 25)
    assert sample.shape == (299_984, 2)
    assert sample.columns.tolist() == [ID, TARGET]
    assert len(feature_columns(train)) == 24
    assert train[TARGET].sum() == 11_689
    assert not train.isna().any().any()
    assert not test.isna().any().any()
