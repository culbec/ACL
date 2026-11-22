# RR2 - Helper Guide: decisions and how to read the results

A companion to the report (`docs/report/report.tex`) and the findings (`findings.md`). It explains **why** each
choice in the code and notebooks was made, and **what each number means** for the research questions.
Numbers match the executed notebooks.

---

## 1. The idea in one paragraph

A Random Forest (or any bagging ensemble) is a committee of small classifiers that vote. We ask whether the committee
members use the **same reasons** for a sample. "Reasons" are not raw columns but **ontology concepts** (e.g. `Family`,
`Education`), so the question is asked at a human level, as in TREPAN Reloaded (RR1). Pipeline:

```
UCI Adult -> EDA -> ontology (OWL)          
          -> 2 ensembles x 100 members -> per-member feature contributions phi (log-odds)
                                       -> sum |phi| per concept -> concept vector per member and sample
                                       -> agreement / disagreement metrics -> RQ1, RQ2, RQ3
```

---

## 2. Decision log

### 2.1 Data (`data.py`, `preprocess.py`, `01_data_eda.ipynb`)

| Decision | Why | Where |
|---|---|---|
| UCI Adult instead of the Kaggle datasets in the spec | Tabular, well known, features have clear human meaning, so an ontology can be built from it; no Kaggle auth | `data.download` |
| Column names parsed from `adult.names` | No hard-coded column list to drift from the source | `data.columns` |
| Official train/test split (32,561 / 16,281) | Standard benchmark split; no randomness; same class ratio (24% vs 23.6%) so no stratification needed | `preprocess.split` |
| Keep rows with missing values; NaN -> all-zero one-hot group | Missing rows have a lower positive rate (0.13 vs 0.25) -> **not missing at random**. Dropping them would bias data; an all-zero group means "unknown" without inventing a value | `preprocess.encode` |
| Drop `fnlwgt` | Census sampling weight, not a person property; no association with any feature or the target (V near 0) | `preprocess.EXCLUDED` |
| Keep both `education` and `education-num` | Same variable (V = 1). Both go to the same concept, and attribution is summed per concept, so duplication does not distort concept vectors | `preprocess.encode` docstring |
| Report F1 beside accuracy | 24% positives -> accuracy alone hides poor minority-class recall (baseline already 0.764) | `03_ensembles` |

### 2.2 Ontology (`ontology.py`, `02_ontology.ipynb`)

```
Person
├── Demographics: Age (age) | Origin (race, native-country) | Sex (sex)
├── Family:       MaritalStatus (marital-status) | HouseholdRole (relationship)
├── Education:    EducationLevel (education, education-num)
├── Employment:   Sector (workclass) | Occupation (occupation) | WorkTime (hours-per-week)
└── Finance:      Capital (capital-gain, capital-loss)
ProtectedAttribute (separate view): race, sex, relationship
```

| Decision | Why |
|---|---|
| Built from EDA evidence | The spec requires it: associations (Cramér's V) decide what belongs together; mutual information decides what is a strong signal |
| Exactly two levels (5 top, 10 leaf) | Enough to test "does agreement change with the level?" (RQ1). Value-level groups (WhiteCollar/BlueCollar...) were rejected: not homogeneous (`Adm-clerical` 0.14 inside WhiteCollar) |
| `Family` split into `MaritalStatus` / `HouseholdRole` | Both are the strongest signals (MI 0.109 / 0.115) and strongly linked (V = 0.49). Keeping them as separate leaves lets the leaf level show which one a member uses. **This turned out to be the key to RQ1** |
| `Finance` = `capital-gain` + `capital-loss` though V = 0.03 | Domain choice (income from assets); each alone is too thin for a concept |
| `ProtectedAttribute` as a separate class outside `Person` | Each feature must have exactly one leaf/top parent for the analysis; the protected view is an extra lens. `relationship` is in it because `Husband`/`Wife` reveal sex (V = 0.65) |
| `mapsToFeature` OWL annotation; all lookups read the saved `.owl` | The analysis depends on the ontology file (opens in Protégé), not on a Python dict |
| Information content (Seco et al.) computed but not used | Required for parity with TREPAN Reloaded; reported only. IC ranks top concepts with fewer leaves (`Education`, `Finance` 0.75) as more specific than `Employment`/`Demographics` (0.50) |

### 2.3 Ensembles (`ensemble.py`, `03_ensembles.ipynb`)

| Decision | Why |
|---|---|
| Two ensembles over the same data | RQ3 needs two member types: trees (non-linear, local) vs linear models (global) |
| RF: 100 trees, `max_depth=5`, `max_features=0.3` | Shallow = "small/dumb" members as the spec asks. Depth 4 + default `sqrt` (~10 of 104 cols) gave members that almost never predicted `>50K`. Chosen setting makes tree member F1 (0.58) match LR member F1 |
| LR: `BaggingClassifier(StandardScaler + LogisticRegression)`, `max_features=0.5` | Without feature subsampling, bagged LR members are near clones -> agreement trivially high (risk in plan §7). Half the columns per member gives real diversity (pairwise disagreement 0.10) |
| Scaler inside each member | Coefficients comparable across features; `lbfgs` converges despite raw `capital-gain` scale; contribution = coef × standardised value |
| `random_state=42`, rebuild instead of pickle | Deterministic, so notebooks 04/05 call `ensemble.build()` and get identical models |

### 2.4 Contributions (`ensemble.member_contributions`)

The only hand-written algorithm (sklearn exposes the tree, not path contributions).

- **Tree (Saabas):** follow the sample's path root -> leaf. At every split, the change in class-1 log-odds is credited
  to the split feature. Sum of credits = `logit(leaf) - logit(root)`.
  - *Laplace smoothing* `(p·n + 1)/(n + 2)`: a pure node has p = 1 -> log-odds = infinity. Smoothing keeps it finite.
- **LR:** `coef_f × z_f` (z = standardised value). Sum = `logit(x) - logit(mean)`.
- **Why log-odds for both:** puts both member types on one scale (median total |phi| 2.7 RF vs 4.0 LR, same order).
- **Checked exact:** max error 8e-7 (RF), 1e-5 (LR) = float32 rounding. Also in `tests/test_core.py`.

### 2.5 Concept vectors and metrics (`concepts.py`)

- `c_m(x, k)` = sum of `|phi|` over the columns of concept k, normalised to 1.
  - **Absolute value:** a concept is "used" whether it pushes up or down. Consequence: vectors say *how much*, never
    *which direction*.
  - **Summed, not averaged:** `Origin` has 46 one-hot columns, `Age` 1; summing keeps column count from inflating a concept.
- Agreement per sample, over 100 members:

| Metric | Question it answers | Range | Chance |
|---|---|---|---|
| top-concept agreement | What share of members name the same #1 concept? | 1/K .. 1 | 0.25 top, 0.15 leaf |
| normalised entropy | How spread are the members' #1 concepts? | 0 (all agree) .. 1 | ~0.99 |
| mean pairwise Kendall tau-b | Do members rank *all* concepts in the same order? | -1 .. 1 | 0 |

- **Chance reference ("shuffled"):** permute each member's concept vector at random -> same values, meaningless
  labels. Any observed value well above it means real agreement.
- **Kendall computed without a pair loop:** tau-b = cosine of sign vectors; sum of all pairwise cosines =
  `(|Σu|² - Σ|u|²)/2`. Pure speed trick, same result.
- **Vote margin** `|share voting 1 - share voting 0|`: 0 = 50/50 split, 1 = unanimous. `< 0.2` = *contested*.
- **Camp difference:** mean concept vector of `>50K` voters minus `<=50K` voters. Positive = the `>50K` camp
  weighs that concept more.
- **Faithfulness:** permute all columns of one concept together (same row permutation, so one-hot stays valid),
  measure each member's accuracy drop. If attribution is honest, high-share concepts should hurt most when broken.

---

## 3. Results and what they mean

### 3.0 Baseline: are the ensembles worth explaining?

| | acc | F1 | member F1 | pairwise disagreement | unanimous |
|---|---|---|---|---|---|
| RF | 0.852 | 0.625 | 0.58 | 0.07 | 62% |
| LR | 0.850 | 0.629 | 0.58 | 0.10 | 51% |

**Meaning:** both beat the 0.764 baseline equally, and members are equally strong, so later differences between RF and
LR are about *how* they reason, not *how well*. Members disagree on 7-10% of samples: enough that agreement on concepts
is a real finding, not a side-effect of identical models.

### 3.1 RQ1 - Do members rely on the same concepts?

| level | ensemble | top agreement (chance) | entropy (chance) | Kendall (chance 0) |
|---|---|---|---|---|
| top | RF | 0.83 (0.25) | 0.30 (0.99) | 0.41 |
| top | LR | 0.79 (0.25) | 0.38 (0.99) | 0.48 |
| leaf | RF | 0.53 (0.15) | 0.45 (0.98) | 0.35 |
| leaf | LR | 0.51 (0.15) | 0.50 (0.98) | 0.37 |

**How to read it:**
- "0.83" = on a typical sample, 83 of 100 trees name the same top concept as their main reason. Chance would be 25.
- At the leaf level this drops to ~half: members still agree far above chance (0.15) but often pick a different leaf.
- **Why the jump between levels:** the heatmap shows two member "schools": one leans on `MaritalStatus`, the other on
  `HouseholdRole`. These two features say the same thing (married / husband-wife). At leaf level the schools look like
  disagreement; at top level both are `Family` (dominant in 76-77% of member-sample pairs).
- **Take-away for the report:** the ontology turns *feature substitution* (members picking different but equivalent
  columns) into *concept agreement*. This is the practical value of a hierarchy for explanation, the same intuition
  TREPAN Reloaded uses.
- Dominant concepts (`Family`, `Education`, `Finance`) are those the EDA's mutual information predicted -> members follow the data.

### 3.2 RQ2 - When members vote differently, do they read different concepts?

| Finding | Number | Meaning |
|---|---|---|
| Contested samples are rare | 190 RF / 884 LR of 16,281 | Most samples are easy; the analysis focuses on a small borderline set |
| Margin vs top-concept entropy | Spearman -0.63 (RF top), -0.32 (RF leaf), -0.39/-0.42 (LR) | Negative = the more split the vote, the more scattered the members' #1 concepts. Prediction disagreement **does** go with concept disagreement |
| RF top agreement unanimous vs contested | 0.89 -> 0.51 | Concrete version of the correlation above |
| Margin vs Kendall | Spearman 0.08-0.20 (weak) | Members that vote differently still order *most* concepts alike; they disagree on **which concept comes first** |
| Camp coherence (Kendall, contested) | `>50K` camp 0.47/0.46 vs whole 0.33/0.36; `<=50K` camp 0.33/0.39 | A `>50K` vote on a borderline case comes from members reading the same evidence; `<=50K` votes come from mixed reasons |
| Camp difference | `Finance/Capital` +0.15 (both); `EducationLevel` +0.08 (RF); `Family` -0.11/-0.09; `WorkTime` -0.10 (RF); `ProtectedAttribute` -0.06 (87% of contested samples) | `>50K` voters lean on money and education; `<=50K` voters lean on family, demographics, working hours and protected attributes |

**Careful wording:** "the `<=50K` camp relies more on `Family`" means it *weighs* `Family` more, **not** that
`Family` pushes towards `<=50K` (vectors are absolute).

**Case studies (4 most contested samples):** all married, each pits two strong cues:

| Sample | true | cue for `>50K` | cue for `<=50K` |
|---|---|---|---|
| 13037 | >50K | capital loss 1,977 (`Capital`) | 10th grade (`EducationLevel`) |
| 13211 | <=50K | Masters (`EducationLevel`) | 15 h/week (`WorkTime`) |
| 13135 | <=50K | Masters (`EducationLevel`) | 4 h/week (`WorkTime`) |
| 7090 | >50K | capital gain 5,178 (`Capital`) | being a wife (`HouseholdRole`/`MaritalStatus`) |

Labels split 2/2 -> genuinely ambiguous people; neither camp is "the correct one". The same cue pairs appear in both
ensembles, so the conflict is in the data, not in one model type.

### 3.3 RQ3 - Trees vs logistic regressions

| | RF - LR (top level) | Meaning |
|---|---|---|
| top agreement | +0.05 | Trees agree more on the **main** concept |
| entropy | -0.075 | same thing, other metric |
| Kendall | -0.07 (LR higher on 83% of samples) | LR members agree more on the **full ranking** |

**Why:** a depth-5 path uses at most 5 features, so a tree's attribution is concentrated on a few concepts; unused
concepts all get 0 and tie, which hurts rank correlation. An LR member gives some weight to every column it sees, so
its full ranking is stable across members. RF members also **specialise** more (member spread on
`HouseholdRole`/`MaritalStatus` 0.20 vs 0.12-0.16 LR), and RF camps split more sharply on contested samples (camp
differences up to 0.59 vs 0.20).

**One-line answer:** trees = "agree on the headline, ignore the rest"; LR = "agree on the whole story, less sharply on
the headline".

### 3.4 Protected attributes

22% (RF) / 30% (LR) of a member's attribution goes to `race`, `sex`, `relationship`; up to 65-68% for single
members. Mostly `relationship` (proxy for sex). LR also uses `sex` and `race` directly (it sees them in half its
members and spreads weight). **Meaning:** the models lean substantially on a sex proxy; this is reported as a
fairness finding, the features were deliberately not removed.

### 3.5 Faithfulness

Per-member Spearman(attribution share, accuracy drop when the concept is permuted): median 0.78 (RF), 0.90 (LR),
positive for every member. **Meaning:** the concept vectors reflect what members actually depend on, so RQ1-RQ3 are not
artefacts of the attribution method. The one mismatch, `HouseholdRole` (high share, smaller drop), is explained by
redundancy: when `relationship` is scrambled, `marital-status` still carries the signal. Attribution = *use*, not
*irreplaceability*.

---

## 4. Figures cheat-sheet (`docs/figures/`)

| Figure | Shows | Look for |
|---|---|---|
| `eda_cramers_v.png` | feature-feature association | education pair = 1; relationship-sex 0.65; fnlwgt row empty |
| `eda_mutual_information.png` | feature-target signal | relationship, marital-status on top |
| `ontology_hierarchy.png` | the ontology | 5 top / 10 leaf, protected view |
| `member_performance.png` | member vs ensemble acc/F1 | members slightly below ensemble, same for RF/LR |
| `member_diversity.png` | pairwise vote disagreement | LR a bit more diverse |
| `agreement_by_level.png` | RQ1 metrics per level | top level higher than leaf |
| `top_concepts.png` | how often each concept is #1 | `Family` dominates |
| `member_concept_heatmap.png` | member x concept shares | two schools: MaritalStatus vs HouseholdRole |
| `agreement_by_margin.png` | RQ2 agreement per margin bucket | RF top-agreement climbs 0.51 -> 0.89 |
| `kendall_vs_margin.png` | RQ2 full-ranking link | nearly flat = weak |
| `camp_difference.png` | RQ2 concepts each camp uses | Finance positive, Family negative |
| `case_studies.png` | 4 most contested samples | each camp lights a different cell |
| `protected_share.png` | protected-attribute share per member | LR higher on average |
| `faithfulness.png` | share vs permutation drop | points near a rising line; HouseholdRole below it |

---

## 5. Limitations to mention when asked

- Attribution shows what a member used, not why it was trained that way.
- Absolute concept vectors: no direction.
- Saabas credits are order-dependent, not Shapley values (TreeSHAP would fix that); Laplace smoothing changes small-node values.
- `marital-status` / `relationship` are substitutes: leaf disagreement partly reflects which one a tree split on first.
- RF contested set is small (190 samples).
- One split, one seed (no variance estimate).
- Hand-built ontology: another grouping would change top-level numbers.

## 6. Where to find things

| What | File |
|---|---|
| Download / load | `src/project_acl/data.py` |
| Clean, encode, split | `src/project_acl/preprocess.py` |
| OWL build, `concept_of`, IC | `src/project_acl/ontology.py` |
| Ensembles, contributions | `src/project_acl/ensemble.py` |
| Concept vectors, metrics, permutation test | `src/project_acl/concepts.py` |
| Tests (contribution sums, coverage, metrics) | `tests/test_core.py` |
| Report source / PDF | `docs/report/report.tex` -> `docs/report/build/report.pdf` (`latexmk -pdf -outdir=build report.tex`) |
