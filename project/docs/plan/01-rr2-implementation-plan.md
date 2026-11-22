# RR2 - Implementation Plan

Source spec: [01-rr2-project.md](../spec/01-rr2-project.md).

## 1. Decisions (agreed with supervisor)

| Topic | Decision |
|---|---|
| Dataset | UCI Adult / Census Income (48,842 rows, 14 features, target `income > 50K`). Downloaded with `requests` from the UCI archive; no Kaggle auth needed. |
| Ontology | Built by us from EDA results. Stored as OWL via `owlready2` (opens in Protégé). |
| Concept measure | Feature-group attribution: each ensemble member's per-sample feature contributions are summed per ontology concept. |
| Ensembles | Two ensembles over the same data, compared against each other: (A) `RandomForestClassifier` with shallow trees; (B) `BaggingClassifier(LogisticRegression)` with feature subsampling. Members = the small classifiers. |
| ML library | `scikit-learn` only. No algorithm written from scratch, except the per-member contribution helper (sklearn exposes the tree structure but not path contributions). |
| Plotting | `seaborn` (a real matplotlib wrapper; figures export cleanly into the report). |
| Logging | `rich` (`RichHandler`). |
| Data I/O | `requests` for download, `pandas` + `numpy` for processing. |
| Package manager | `uv` for every dependency/entry-point change. |

## 2. Research questions

- **RQ1 - Concept agreement.** When the ensemble classifies a sample, do its members rely on the same ontology concepts? Does agreement change with the ontology level (fine concepts vs top-level concepts)?
- **RQ2 - Concept disagreement.** When members disagree on the predicted class, which concepts do they read differently? Is prediction disagreement correlated with concept disagreement?
- **RQ3 - Members compared.** Do tree members and logistic-regression members show different agreement patterns?

## 3. Method

### 3.1 Ontology (approved after P2, evidence in `01_data_eda.ipynb` section 7)

```
Person
├── Demographics
│   ├── Age              age
│   ├── Origin           race, native-country
│   └── Sex              sex
├── Family
│   ├── MaritalStatus    marital-status
│   └── HouseholdRole    relationship
├── Education
│   └── EducationLevel   education, education-num
├── Employment
│   ├── Sector           workclass
│   ├── Occupation       occupation
│   └── WorkTime         hours-per-week
└── Finance
    └── Capital          capital-gain, capital-loss
ProtectedAttribute (separate view)   race, sex, relationship  (relationship leaks sex)
(excluded)                           fnlwgt  -> census sampling weight, not a person property
```

- Two levels only: level 1 = 5 top concepts, level 2 = 10 leaf concepts. No value-level sub-concepts.
- Each feature is linked to its concept with an OWL annotation property (`mapsToFeature`). One-hot columns inherit the concept of their source column.
- Features outside the ontology are dropped from the concept analysis, as in TREPAN Reloaded.
- Information content (IC) per concept is computed from the number of sub-concepts, as in TREPAN Reloaded. It is reported, not used for training.

### 3.2 Per-member contributions

For a sample `x` and member `m`, we compute a contribution `phi_m(x, f)` for each feature `f`:

- **Tree member:** path decomposition (Saabas). Walk the decision path; at each split, the change in the class-1 probability is credited to the split feature. Contributions sum to `p_m(x) - p_m(root)`.
- **LR member:** `coef_f * (x_f - mean_f)` on the member's feature subset, in log-odds space.

Concept vector: `c_m(x, k) = sum over f in concept k of |phi_m(x, f)|`, normalised to sum to 1. It is computed at each ontology level (leaf concepts, top-level concepts).

### 3.3 Metrics

Agreement (RQ1), per sample, then averaged:
- top-concept agreement: share of members whose top concept equals the most common top concept;
- concept entropy: entropy of the members' top-concept distribution;
- mean pairwise rank correlation (Kendall tau) of the members' concept vectors.

Disagreement (RQ2):
- vote margin: `|share voting 1 - share voting 0|`; low margin = high disagreement;
- for low-margin samples: mean concept vector of members voting 1 minus that of members voting 0, per concept;
- correlation between vote margin and concept agreement over the test set.

Sanity check (faithfulness): permute all features of one concept and measure each member's accuracy drop. Concepts with high attribution should show high drops.

## 4. Project layout

```
project/
├── pyproject.toml
├── data/                    # gitignored: raw/, processed/, ontology/
├── docs/{spec,plan}/
├── notebooks/
│   ├── 01_data_eda.ipynb
│   ├── 02_ontology.ipynb
│   ├── 03_ensembles.ipynb
│   ├── 04_concept_agreement.ipynb
│   └── 05_concept_disagreement.ipynb
├── src/project_acl/
│   ├── log.py               # rich logger
│   ├── data.py              # download, clean, encode, split
│   ├── ontology.py          # build/save/load OWL, feature->concept lookup, IC
│   ├── ensemble.py          # build ensembles, per-member contributions
│   └── concepts.py          # concept vectors, agreement/disagreement metrics
└── tests/test_core.py       # one small test file for the non-trivial logic
```

Notebooks hold the analysis and the plots; modules hold reusable code only.

## 5. Phases (agent tasks)

Each phase is one agent task. **Gate** = supervisor review before the next dependent phase starts.

| # | Phase | Depends on | Parallel with |
|---|---|---|---|
| P0 | Setup | - | - |
| P1 | Logging + data module | P0 | - |
| P2 | EDA notebook | P1 | - |
| P3 | Ontology | P2 (gate) | P4 |
| P4 | Ensembles + contributions | P1 | P3 |
| P5 | Concept agreement (RQ1, RQ3) | P3, P4 | P6 |
| P6 | Concept disagreement (RQ2) | P3, P4 | P5 |
| P7 | Report figures + summary | P5, P6 | - |

### P0 - Setup
- `uv add numpy pandas requests rich scikit-learn owlready2 seaborn`; `uv add --dev jupyter ipykernel pytest`.
- Create the folders from section 4; add `data/` to `.gitignore`.
- Check that every dependency installs on Python 3.14. If one has no wheel, report it; do not downgrade Python without approval.
- **Done when:** `uv run python -c "import sklearn, owlready2, seaborn, rich"` passes.

### P1 - Logging + data module
- `log.py`: `get_logger(name)` returning a `logging` logger with `RichHandler`.
- `data.py`:
  - `download()`: fetch `adult.data` and `adult.test` from the UCI archive into `data/raw/`; skip files already present.
  - `load()`: read with column names; strip whitespace; `?` -> NaN; remove the trailing `.` from test labels; binary target.
  - `encode()`: one-hot encode categoricals; return `X`, `y`, and a `column -> source feature` map (needed for the concept lookup).
  - `split()`: use the official train/test split; fixed `random_state`.
- **Done when:** the test checks row counts, no `?` left, binary target, and that every encoded column maps to a source feature.

### P2 - EDA notebook (`01_data_eda.ipynb`)
- Distributions, missing values, class balance.
- Associations between features (Cramér's V for categoricals, correlation for numerics) and with the target (mutual information).
- Value-level grouping candidates for high-cardinality columns (`occupation`, `education`, `native-country`).
- End the notebook with a **proposed ontology** derived from these findings.
- **Gate:** supervisor approves the ontology before P3 builds it.

### P3 - Ontology (`ontology.py`, `02_ontology.ipynb`)
- Build the approved hierarchy in OWL with `owlready2`; save to `data/ontology/adult.owl`.
- `concept_of(feature, level)` lookup; `information_content(concept)`.
- Notebook: hierarchy figure, feature coverage table, IC per concept.
- **Done when:** the OWL file reloads, and every non-excluded feature maps to exactly one concept per level (the `ProtectedAttribute` second parent is handled as a separate view).

### P4 - Ensembles + contributions (`ensemble.py`, `03_ensembles.ipynb`)
- RF: `max_depth` 3-5, about 100 trees. Bagging-LR: about 100 members, `max_features` about 0.5 so the members differ.
- `member_contributions(model, X) -> array (members, samples, features)` for both member types.
- Notebook: accuracy / F1 of the ensembles and of single members; member diversity (pairwise prediction disagreement).
- **Done when:** the test checks that tree contributions sum to `p_m(x) - p_m(root)` for each member.

### P5 - Concept agreement (`concepts.py`, `04_concept_agreement.ipynb`)
- Concept vectors per level; the agreement metrics from 3.3.
- Plots: agreement distribution per ensemble, agreement per ontology level, member x concept heatmap.
- Faithfulness check (concept permutation).
- Answers RQ1 and RQ3.

### P6 - Concept disagreement (`05_concept_disagreement.ipynb`)
- Vote-margin buckets; concept difference between the two voting camps; margin vs concept agreement scatter.
- Case studies: the 3-5 most contested test samples, explained by concept.
- Answers RQ2.

### P7 - Report material
- Export final figures to `docs/figures/`; a short findings summary in `docs/` with the answer to each RQ and its limitations.

## 6. Rules for agents

- Use `uv` for every package/entry-point change; never `pip`.
- Use the `log.py` logger in modules; do not use `print`.
- Fixed seeds everywhere (`random_state=42`).
- Notebooks must run top to bottom (`uv run jupyter nbconvert --execute`) before a phase is reported as done.
- One phase per agent; report what changed, how it was checked, and any open question.

## 7. Risks

- **Python 3.14 wheels.** `owlready2` or `scikit-learn` may lack wheels; P0 checks this first.
- **Similar LR members.** Bagged LR members may be near identical, so agreement is trivially high. `max_features` subsampling reduces this; if they are still near identical, that is a valid RQ3 finding.
- **Attribution is not reasoning.** Path contributions show what a member used, not why it was trained that way. The same caveat applies as for the mapping-network justifications reviewed in RR1.
- **Fairness.** `sex` and `race` are model inputs, and `relationship` is a proxy for `sex`. The `ProtectedAttribute` view reports how much members rely on them; this is a finding to discuss, not something to remove silently.
