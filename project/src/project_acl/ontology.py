"""Adult ontology: build, save and load the OWL file, feature -> concept lookup, information content (IC).

The hierarchy is derived from the feature associations in `01_data_eda.ipynb`. `HIERARCHY` is used only to build
the OWL file; every lookup reads the saved file, so the analysis depends on the ontology and not on this dict.
"""

import math
import types
from functools import cache
from pathlib import Path

from owlready2 import AnnotationProperty, Thing, World

from project_acl.data import DATA_DIR
from project_acl.log import get_logger

log = get_logger(__name__)

OWL_PATH = DATA_DIR / "ontology" / "adult.owl"
IRI = "http://project-acl.local/adult.owl#"
ROOT = "Person"
PROTECTED = "ProtectedAttribute"

# top concept (level 1) -> leaf concept (level 2) -> source features of the dataset
HIERARCHY = {
    "Demographics": {"Age": ["age"], "Origin": ["race", "native-country"], "Sex": ["sex"]},
    "Family": {"MaritalStatus": ["marital-status"], "HouseholdRole": ["relationship"]},
    "Education": {"EducationLevel": ["education", "education-num"]},
    "Employment": {"Sector": ["workclass"], "Occupation": ["occupation"], "WorkTime": ["hours-per-week"]},
    "Finance": {"Capital": ["capital-gain", "capital-loss"]},
}
# Separate view over existing features: relationship is included because Husband/Wife reveal sex
PROTECTED_FEATURES = ["race", "sex", "relationship"]


def build(path: Path = OWL_PATH) -> Path:
    """Write the hierarchy as OWL classes; leaves carry `mapsToFeature` annotations."""
    onto = World().get_ontology(IRI)
    with onto:
        class mapsToFeature(AnnotationProperty):  # noqa: N801 - OWL property name
            pass

        # types.new_class creates OWL classes from names held in variables (owlready2 maps Python classes to OWL)
        root = types.new_class(ROOT, (Thing,))
        for top, leaves in HIERARCHY.items():
            parent = types.new_class(top, (root,))
            for leaf, feats in leaves.items():
                types.new_class(leaf, (parent,)).mapsToFeature = feats
        # Separate view outside Person, so each feature still has exactly one leaf under Person
        types.new_class(PROTECTED, (Thing,)).mapsToFeature = PROTECTED_FEATURES
    path.parent.mkdir(parents=True, exist_ok=True)
    onto.save(file=str(path))
    log.info("saved ontology %s", path)
    return path


def load(path: Path = OWL_PATH):
    """Ontology from disk (built first if missing), in its own owlready2 world."""
    if not path.exists():
        build(path)
    return World().get_ontology(path.as_uri()).load()


@cache
def _lookup(path: Path = OWL_PATH) -> tuple[dict[str, dict[int, str]], frozenset[str], dict[str, float]]:
    """(feature -> {1: top, 2: leaf}, protected features, concept -> IC), read from the OWL file.

    Cached: the OWL file is parsed once per path.
    """
    onto = load(path)
    root = onto[ROOT]
    concepts = {}
    for leaf in root.descendants():
        for f in leaf.mapsToFeature:
            if f in concepts:
                raise ValueError(f"{f} maps to two leaves: {concepts[f][2]}, {leaf.name}")
            (top,) = leaf.is_a  # leaves have exactly one parent
            concepts[f] = {1: top.name, 2: leaf.name}

    # Intrinsic IC (Seco et al.), as in TREPAN Reloaded: 1 - log(sub-concepts + 1) / log(concepts in hierarchy).
    # owlready2's descendants() includes the class itself, so len(descendants) = sub-concepts + 1.
    n = len(root.descendants())
    ic = {c.name: 1 - math.log(len(c.descendants())) / math.log(n) for c in root.descendants()}
    return concepts, frozenset(onto[PROTECTED].mapsToFeature), ic


def concept_of(feature: str, level: int, path: Path = OWL_PATH) -> str | None:
    """Concept of a source feature at level 1 (top) or 2 (leaf); None for features outside the ontology."""
    if level not in (1, 2):
        raise ValueError(f"level must be 1 or 2, got {level}")
    return _lookup(path)[0].get(feature, {}).get(level)


def is_protected(feature: str, path: Path = OWL_PATH) -> bool:
    """True if the feature belongs to the ProtectedAttribute view."""
    return feature in _lookup(path)[1]


def information_content(concept: str, path: Path = OWL_PATH) -> float:
    """IC of a concept under Person: 0 for the root, 1 for leaves."""
    return _lookup(path)[2][concept]
