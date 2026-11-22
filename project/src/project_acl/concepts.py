"""Concept vectors and the agreement / disagreement metrics of plan section 3.2-3.3.

Shapes: `phi` (members, samples, columns) from `ensemble.member_contributions`; `proba` (members, samples) from
`ensemble.member_proba`; concept vectors `v` (members, samples, concepts).
"""

import numpy as np
import pandas as pd

from project_acl import ensemble, ontology

PROTECTED_VIEW = "protected"  # level name for the ProtectedAttribute view: ProtectedAttribute vs Other


def column_concepts(columns, source: dict[str, str], level: int | str) -> np.ndarray:
    """Concept of every encoded column at level 1 (top), 2 (leaf) or `PROTECTED_VIEW`. None = outside the ontology."""
    if level == PROTECTED_VIEW:
        return np.array([ontology.PROTECTED if ontology.is_protected(source[c]) else "Other" for c in columns], object)
    return np.array([ontology.concept_of(source[c], level) for c in columns], object)


def concept_vectors(phi: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """c_m(x, k) = sum of |phi_m(x, f)| over the columns f of concept k, normalised to sum to 1 per member and sample.

    Columns labelled None are dropped. Returns v (members, samples, concepts) and the concept names (sorted).
    """
    concepts = sorted({c for c in labels if c is not None})
    onehot = (labels[:, None] == np.array(concepts, object)[None]).astype(phi.dtype)
    v = np.stack([np.abs(p) @ onehot for p in phi])  # per member: avoids a full-size |phi| copy
    total = v.sum(axis=2, keepdims=True)
    return np.divide(v, total, out=np.zeros_like(v), where=total > 0), concepts


def mean_kendall(v: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Mean pairwise Kendall tau-b between the members' concept vectors, per sample.

    tau-b of two rankings is the cosine of their pair-sign vectors `sign(v_i - v_j)` over concept pairs i < j (ties
    give 0), so the mean over member pairs follows from the sum of the unit sign vectors: no loop over 4950 pairs.
    Members whose concept vector is constant (no ranking) are left out; NaN if fewer than two remain.
    """
    m, n, k = v.shape
    i, j = np.triu_indices(k, 1)
    out = np.empty(n)
    for s in range(0, n, chunk):
        sign = np.sign(v[:, s : s + chunk, i] - v[:, s : s + chunk, j])
        norm = np.linalg.norm(sign, axis=2, keepdims=True)
        unit = np.divide(sign, norm, out=np.zeros_like(sign), where=norm > 0)
        total = unit.sum(axis=0)
        valid = (norm[..., 0] > 0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s : s + chunk] = ((total**2).sum(-1) - (unit**2).sum((0, 2))) / (valid * (valid - 1))
    return out


def agreement(v: np.ndarray) -> pd.DataFrame:
    """Per-sample agreement (RQ1):

    - `top_agreement`: share of members whose top concept is the most common top concept;
    - `entropy`: entropy of the members' top-concept distribution, divided by log(concepts) so levels compare (0..1);
    - `kendall`: mean pairwise Kendall tau-b of the concept vectors.
    """
    m, _, k = v.shape
    top = v.argmax(axis=2)
    p = np.stack([(top == c).mean(axis=0) for c in range(k)], axis=1)
    logp = np.log(p, out=np.zeros_like(p), where=p > 0)
    return pd.DataFrame({
        "top_agreement": p.max(axis=1),
        "entropy": -(p * logp).sum(axis=1) / np.log(k),
        "kendall": mean_kendall(v),
    })


def votes(proba: np.ndarray) -> np.ndarray:
    return proba > 0.5


def vote_margin(proba: np.ndarray) -> np.ndarray:
    """|share voting 1 - share voting 0| per sample; 0 = split vote, 1 = unanimous."""
    return np.abs(2 * votes(proba).mean(axis=0) - 1)


def camp_difference(v: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Mean concept vector of members voting 1 minus that of members voting 0, shape (samples, concepts).

    NaN where one camp is empty (unanimous vote).
    """
    one = votes(proba)[..., None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (v * one).sum(0) / one.sum(0) - (v * ~one).sum(0) / (~one).sum(0)


def permutation_drop(model, X, y, labels: np.ndarray, concepts: list[str], seed: int = ensemble.SEED) -> np.ndarray:
    """Faithfulness check: accuracy drop of every member when all columns of one concept are permuted together.

    One row permutation shared by the concept's columns keeps one-hot groups valid. Shape (members, concepts).
    """
    X, y = np.asarray(X, float), np.asarray(y)
    perm = np.random.default_rng(seed).permutation(len(X))
    base = (votes(ensemble.member_proba(model, X)) == y).mean(axis=1)
    drops = []
    for c in concepts:
        cols = labels == c
        Xp = X.copy()
        Xp[:, cols] = X[perm][:, cols]
        drops.append(base - (votes(ensemble.member_proba(model, Xp)) == y).mean(axis=1))
    return np.stack(drops, axis=1)
