"""Checks for the non-trivial logic: data loading, ontology lookup, contribution sums, concept metrics."""

from project_acl import data, ontology, preprocess


def test_data():
    """Official row counts, no `?` left, binary target, every encoded column traced to a source feature."""
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
    """The OWL file reloads and every model feature maps to exactly one concept per level."""
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


def test_ensemble():
    """Per-member contributions add up exactly to the member's log-odds change."""
    import numpy as np
    from project_acl import ensemble

    df = preprocess.read()
    X, y, _ = preprocess.encode(df)
    X_tr, X_te, y_tr, _ = preprocess.split(X, y, df)
    X_te = X_te.to_numpy(float)[:500]
    models = ensemble.build(X_tr, y_tr, n_members=5)

    for name, model in models.items():
        phi = ensemble.member_contributions(model, X_te, dtype=float)
        p = ensemble.member_proba(model, X_te)
        assert phi.shape == (5, *X_te.shape) and p.shape == (5, len(X_te))
        for k, (m, f) in enumerate(ensemble.members(model)):
            if name == "RF":  # Saabas in log-odds: contributions sum to logit(leaf) - logit(root)
                logits = ensemble.node_logit(m)
                expected = logits[m.apply(X_te)] - logits[0]
            else:  # LR: contributions sum to logit(x) - logit(member's training mean)
                expected = m.decision_function(X_te[:, f]) - m.decision_function(m[0].mean_[None])[0]
                assert (phi[k][:, np.setdiff1d(np.arange(X_te.shape[1]), f)] == 0).all()
            np.testing.assert_allclose(phi[k].sum(axis=1), expected, atol=1e-6)


def test_concepts():
    """Concept vectors, vectorised Kendall tau-b, vote margin and camp difference against direct computations."""
    import numpy as np
    from scipy.stats import kendalltau
    from project_acl import concepts

    rng = np.random.default_rng(0)
    phi = rng.normal(size=(6, 40, 5))
    phi[:, :, 4] = 0  # a column never used: ties in every ranking
    labels = np.array(["A", "B", "B", None, "C"], object)
    v, names = concepts.concept_vectors(phi, labels)
    assert names == ["A", "B", "C"]
    np.testing.assert_allclose(v.sum(axis=2), 1)
    np.testing.assert_allclose(v[..., 1] * np.abs(phi[..., :3]).sum(2), np.abs(phi[..., 1:3]).sum(2))

    # vectorised tau-b == mean of scipy's tau-b over member pairs
    v = np.round(rng.random((6, 30, 6)), 1)  # rounding creates ties
    expected = [np.mean([kendalltau(v[a, s], v[b, s]).statistic for a in range(6) for b in range(a + 1, 6)])
                for s in range(30)]
    np.testing.assert_allclose(concepts.mean_kendall(v, chunk=7), expected)

    proba = np.array([[0.9, 0.9], [0.8, 0.1], [0.2, 0.7]])
    np.testing.assert_allclose(concepts.vote_margin(proba), [1 / 3, 1 / 3])
    v = rng.random((3, 2, 4))
    diff = concepts.camp_difference(v, proba)
    np.testing.assert_allclose(diff[0], (v[0, 0] + v[1, 0]) / 2 - v[2, 0])
    assert np.isnan(concepts.camp_difference(v, np.full((3, 2), 0.9))).all()
