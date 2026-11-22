from project_acl import data, preprocess


def test_data():
    data.download()
    df = data.load()
    assert len(data.columns()) == 15
    assert (df["split"] == "train").sum() == 32561
    assert (df["split"] == "test").sum() == 16281
    assert not df.isin(["?"]).any().any()
    assert set(df["income"].unique()) == {0, 1}

    assert not preprocess.clean(df, dropna=True).isna().any().any()

    X, y, source = preprocess.encode(df)
    assert set(source) == set(X.columns)
    assert set(source.values()) == set(data.columns()[:-1])

    X_tr, X_te, y_tr, y_te = preprocess.split(X, y, df)
    assert len(X_tr) == len(y_tr) == 32561 and len(X_te) == len(y_te) == 16281
    assert list(X_tr.columns) == list(X_te.columns)
