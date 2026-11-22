from project_acl import data, ontology, preprocess


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
    assert set(source.values()) == set(data.columns()[:-1]) - set(preprocess.EXCLUDED)

    X_tr, X_te, y_tr, y_te = preprocess.split(X, y, df)
    assert len(X_tr) == len(y_tr) == 32561 and len(X_te) == len(y_te) == 16281
    assert list(X_tr.columns) == list(X_te.columns)


def test_ontology(tmp_path):
    path = ontology.build(tmp_path / "adult.owl")
    onto = ontology.load(path)  # reloads from disk
    assert onto.Person and onto.mapsToFeature

    feats = set(data.columns()[:-1]) - set(preprocess.EXCLUDED)
    leaves = {ontology.concept_of(f, 2, path) for f in feats}
    tops = {ontology.concept_of(f, 1, path) for f in feats}
    assert None not in leaves | tops  # every model feature is covered
    assert len(leaves) == 10 and len(tops) == 5
    assert ontology.concept_of("fnlwgt", 2, path) is None
    assert ontology.concept_of("education-num", 2, path) == ontology.concept_of("education", 2, path)
    assert ontology.is_protected("relationship", path) and ontology.concept_of("relationship", 1, path) == "Family"

    assert ontology.information_content("Person", path) == 0
    assert ontology.information_content("Age", path) == 1
    assert 0 < ontology.information_content("Employment", path) < ontology.information_content("Finance", path) < 1
