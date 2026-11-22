"""The two ensembles (RF of shallow trees, bagged LR) and per-member feature contributions.

Members are the small classifiers inside each ensemble. Both models are deterministic (`random_state=42`),
so later notebooks rebuild them with `build()` instead of loading pickles.
"""

import numpy as np
from scipy import sparse
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from project_acl.log import get_logger

log = get_logger(__name__)

SEED = 42


def build(X, y, n_members: int = 100, max_depth: int = 5, tree_features: float = 0.3, lr_features: float = 0.5) -> dict:
    """Fit both ensembles on the same data. Returns {"RF": ..., "LR": ...}.

    Trees: depth 5 with 30% of the columns per split, so members reach about LR-member F1 (default `sqrt`, ~10 of
    104 columns, gave many members that almost never predict `>50K`).
    LR members standardise their own bootstrap sample + feature subset (scaler inside each member), so coefficients
    are comparable across features and `lbfgs` converges on the raw capital-gain scale.
    """
    X = np.asarray(X)
    models = {
        "RF": RandomForestClassifier(
            n_members, max_depth=max_depth, max_features=tree_features, random_state=SEED, n_jobs=-1
        ),
        "LR": BaggingClassifier(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            n_estimators=n_members, max_features=lr_features, random_state=SEED, n_jobs=-1,
        ),
    }
    for name, m in models.items():
        m.fit(X, y)
        log.info("fitted %s: %d members", name, len(m.estimators_))
    return models


def members(model) -> list[tuple[object, np.ndarray]]:
    """(member, feature indices it sees) for each member of a fitted RF or Bagging model."""
    if isinstance(model, BaggingClassifier):
        return list(zip(model.estimators_, model.estimators_features_))
    all_feats = np.arange(model.n_features_in_)
    return [(t, all_feats) for t in model.estimators_]


def member_proba(model, X) -> np.ndarray:
    """Class-1 probability of every member, shape (members, samples). Votes are `proba > 0.5`."""
    X = np.asarray(X)
    return np.stack([m.predict_proba(X[:, f])[:, 1] for m, f in members(model)])


def node_logit(tree) -> np.ndarray:
    """Class-1 log-odds at every node, Laplace-smoothed `(p n + 1) / (n + 2)` so pure nodes stay finite.

    `n` is the bootstrap-weighted sample count; `tree_.value` is normalised or counts depending on sklearn version.
    """
    t = tree.tree_
    v = t.value[:, 0, :]  # (nodes, classes)
    n = t.weighted_n_node_samples
    p = (v[:, 1] / v.sum(axis=1) * n + 1) / (n + 2)  # dividing by the row sum works for both value formats
    return np.log(p / (1 - p))


def tree_contributions(tree, X) -> np.ndarray:
    """Saabas path decomposition in log-odds space, shape (samples, features). Rows sum to `logit(leaf) - logit(root)`.

    Each non-root node carries `logit(node) - logit(parent)` (see `node_logit`), credited to the parent's split feature;
    a sample's contribution is the sum over the nodes on its decision path.
    """
    t = tree.tree_
    p = node_logit(tree)
    # parent[node] = index of its parent; -1 for the root. sklearn stores children, so invert that map.
    parent = np.full(t.node_count, -1)
    for kids in (t.children_left, t.children_right):
        is_split = kids >= 0  # leaves have child index -1
        parent[kids[is_split]] = np.flatnonzero(is_split)
    child = np.flatnonzero(parent >= 0)
    # credit[node, feature]: change in log-odds when entering `node`, booked on the feature its parent splits on
    credit = sparse.csr_matrix(
        (p[child] - p[parent[child]], (child, t.feature[parent[child]])), shape=(t.node_count, t.n_features)
    )
    # decision_path is a (samples, nodes) 0/1 matrix, so the product sums the credits along each sample's path
    return np.asarray((tree.decision_path(X) @ credit).todense())


def lr_contributions(pipe, X) -> np.ndarray:
    """`coef_f * (x_f - mean_f)` in log-odds space on standardised inputs, shape (samples, member features).

    `mean_f` is the member's own training mean, so rows sum to `logit(x) - logit(mean)`.
    """
    scaler, lr = pipe[0], pipe[-1]
    # after scaling, x_f - mean_f is the standardised value itself, so coef * scaled input is the contribution
    return scaler.transform(X) * lr.coef_[0]


def member_contributions(model, X, dtype=np.float32) -> np.ndarray:
    """phi[m, i, f]: contribution of feature f to member m's class-1 score on sample i.

    Features a member never sees get 0. Both member types are in log-odds space, so raw values are comparable
    across ensembles (tree leaves are Laplace-smoothed, LR inputs standardised).
    float32 keeps the full test set (100 x 16k x ~100) under 1 GB.
    """
    X = np.asarray(X, dtype=float)
    phi = np.zeros((len(model.estimators_), len(X), X.shape[1]), dtype=dtype)
    for k, (m, f) in enumerate(members(model)):
        # LR members see only their feature subset `f`; trees see every column (f covers all of them)
        phi[k][:, f] = lr_contributions(m, X[:, f]) if isinstance(model, BaggingClassifier) else tree_contributions(m, X)
    return phi
