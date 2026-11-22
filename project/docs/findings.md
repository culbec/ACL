# RR2 - Findings

Phase P7 of the [implementation plan](plan/01-rr2-implementation-plan.md). Numbers come from the executed notebooks
(`notebooks/03`-`05`); figures are in [`figures/`](figures/) and are regenerated when the notebooks run.

## Setup

- **Data.** UCI Adult, official split: 32,561 train / 16,281 test, 104 encoded columns, 23.6% positive in test.
- **Ontology.** `Person` -> 5 top concepts -> 10 leaf concepts, plus a separate `ProtectedAttribute` view
  (`race`, `sex`, `relationship`); `fnlwgt` excluded. Figure: `ontology_hierarchy.png`.
- **Ensembles.** RF: 100 trees, depth 5, `max_features=0.3`. LR: `BaggingClassifier` of 100
  `StandardScaler + LogisticRegression`, `max_features=0.5`.

| | accuracy | F1 | mean member F1 | mean pairwise member disagreement | unanimous samples |
|---|---|---|---|---|---|
| RF | 0.852 | 0.625 | 0.58 | 0.07 | 62% |
| LR | 0.850 | 0.629 | 0.58 | 0.10 | 51% |

Majority baseline accuracy: 0.764. Figures: `member_performance.png`, `member_diversity.png`.

- **Concept measure.** Per-member contributions in log-odds (Saabas path decomposition for trees, `coef * (x - mean)`
  for LR), absolute values summed per concept and normalised to 1. Decompositions are exact up to float rounding
  (max error 8e-7 RF, 1e-5 LR).

## RQ1 - Concept agreement

**Members agree far above chance, and agreement is higher at the top ontology level.**

| level | ensemble | top-concept agreement (chance) | normalised entropy (chance) | mean Kendall tau-b (chance 0) |
|---|---|---|---|---|
| top | RF | 0.83 (0.25) | 0.30 (0.99) | 0.41 |
| top | LR | 0.79 (0.25) | 0.38 (0.99) | 0.48 |
| leaf | RF | 0.53 (0.15) | 0.45 (0.98) | 0.35 |
| leaf | LR | 0.51 (0.15) | 0.50 (0.98) | 0.37 |

- Agreement is partial: at the leaf level about half of the members name a different top concept than the majority.
- The gain at the top level comes mostly from `Family`. Members split into a `MaritalStatus` group and a
  `HouseholdRole` group, two features that carry the same signal (married, husband/wife). At the leaf level the groups
  disagree; at the top level both count as `Family`, the top concept in 76-77% of (member, sample) pairs. The ontology
  turns a substitution between correlated features into agreement on one concept.

Figures: `agreement_by_level.png`, `top_concepts.png`, `member_concept_heatmap.png`.

## RQ2 - Concept disagreement

**Prediction disagreement goes with concept disagreement, mostly on which concept dominates.**

- Contested samples (vote margin < 0.2) are rare: 190 (RF) and 884 (LR) of 16,281.
- Spearman correlation between vote margin and top-concept entropy: -0.63 (RF, top), -0.32 (RF, leaf),
  -0.39 / -0.42 (LR, top / leaf). For RF at the top level, top-concept agreement falls from 0.89 on unanimous samples
  to 0.51 on contested ones.
- The link with the full ranking is weak (Spearman of margin vs Kendall 0.08-0.20): members that vote differently
  still order most concepts alike and differ in which concept dominates.
- On contested samples the `>50K` camp is coherent (Kendall 0.47 RF, 0.46 LR vs 0.33 / 0.36 for the whole
  ensemble); the `<=50K` camp is about as mixed as the whole ensemble (0.33 / 0.39).
- **Concepts read differently.** The `>50K` camp weights `Finance`/`Capital` more (+0.15 in both ensembles) and, for
  RF, `EducationLevel` (+0.08). The `<=50K` camp weights `Family` more (-0.11 RF, -0.09 LR), `Demographics`, and for RF
  `WorkTime` (-0.10). It also weights `ProtectedAttribute` more (-0.06, in 87% of contested samples in both ensembles).
- **Case studies.** The four most contested test samples are all married and each pits two strong cues against each
  other, the same pair in both ensembles: capital loss vs 10th-grade education; a Masters degree vs 15 or 4 hours per
  week; a 5,178 capital gain vs being a wife. True labels split 2/2, so these are genuine borderline cases.

Figures: `agreement_by_margin.png`, `kendall_vs_margin.png`, `camp_difference.png`, `case_studies.png`.

## RQ3 - Tree members vs LR members

**Trees agree more on the main concept; LR members agree more on the full ranking.**

- Paired over the same samples, RF has higher top-concept agreement (+0.05 top, +0.03 leaf) and lower entropy, but
  lower Kendall (-0.07 top, -0.02 leaf; LR higher on 83% of samples at the top level).
- Reason: a depth-5 path touches at most 5 features, so tree attribution is concentrated. `Origin`, `Sector` and `Sex`
  get almost nothing and the unused concepts tie. LR members spread weight over every concept they see
  (`Occupation` 0.13, `Sex` 0.06, `Origin` and `Sector` 0.04).
- RF members specialise more: the spread of members' mean share on `HouseholdRole` / `MaritalStatus` is 0.20 vs
  0.12-0.16 for LR.
- In contested samples the RF camps differ more sharply (camp differences up to 0.59 vs 0.20 for LR).

## Protected attributes

On average 22% (RF) and 30% (LR) of a member's attribution goes to `race`, `sex` or `relationship`, up to 65-68% for
single members. Most of it is `relationship` (`HouseholdRole`, a proxy for sex); LR members also use `sex` and `race`
directly. These features were kept as model inputs on purpose; the share is reported, not removed.
Figure: `protected_share.png`.

## Faithfulness check

Permuting all columns of one leaf concept and measuring each member's accuracy drop ranks concepts like the
attribution does: per-member Spearman median 0.78 (RF) and 0.90 (LR), positive for every member (min 0.29 / 0.63).
The main mismatch is `HouseholdRole`: high attribution but a smaller drop than `EducationLevel` or `MaritalStatus`,
because `marital-status` still carries the signal when `relationship` is permuted. Attribution measures use, not
irreplaceability. Figure: `faithfulness.png`.

## Limitations

- **Attribution is not reasoning.** Contributions show which features a member used for a sample, not why it was
  trained that way.
- **Absolute concept vectors.** They show how much a concept is used, not the direction. "The `<=50K` camp relies more
  on `Family`" does not mean `Family` pushes it to `<=50K`.
- **Tree attribution method.** Saabas path credits are order-dependent and not Shapley values; Laplace smoothing
  changes the values at small nodes.
- **Correlated features.** `marital-status` and `relationship` are near substitutes, so leaf-level disagreement
  between them partly reflects which one a member happened to split on.
- **Small contested set.** The RF `margin < 0.2` bucket holds only 190 samples.
- **One split, one seed.** All results use the official test split and `random_state=42`; no variance over seeds.
- **Hand-built ontology.** The hierarchy was derived from EDA and supervisor review; a different grouping would change
  the top-level results.
