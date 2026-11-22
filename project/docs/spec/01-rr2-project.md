# RR2 - Spec

## Introduction

This document describes the idea around the project for RR2. It is based
on the conclusions of the first report (RR1) that studied how ontologies
can be used for xAI (Explainable AI). Two papers were studied in order to
review how a concept based knowledge extraction framework is used to explain
the decisions of black-box ML models; the paper review can be consulted in the [paper review document](../../../lectures/ACL-Paper-Review.pdf).

The speaker notes that outline the presentation of the review + future directions for RR2 is located in the [speaker notes document](../../../lectures/ACL-RR1-Speaker-Notes.pdf).

## Description

For RR2 I am thinking to tackle the easiest of the proposed methods, that being concept agreement. I am planning to use an ontology dataset in a Random Forest or other ensemble model and see whether the smaller / dumber classifiers that form the ensemble agree on the same concepts when a classification is made. That can be combined with the other idea, that implies concept disagreement and see where the small classifiers disagree most. The project developed can be tackled in both manners.

### Datasets

- https://www.kaggle.com/datasets/therohk/cyc-kb
- https://www.kaggle.com/datasets/drahung/cancer-categories-and-clinical-research-figures 
- suggestions of datasets are welcome where we can build our own ontology

## Technical Requirements

- the Python package manager is `uv`; the project is initialized and located at [pyproject.toml](../../pyproject.toml). All project altering work regarding packages, sources, entry points, and so on is done using `uv`.
- No need to implement ML algorithms from scratch; we can use libraries.
- Plotting will be done using a `matplotlib` wrapper such as `plotly` or `seaborn`.
- All work will be outlined in `Jupyter Notebooks`; the notebooks will serve as project work / demo, and not oriented towards plain script execution.
- Other datasets are welcome as examples online are not oriented towards tabular data for ontologies, but rather towards LLM usage and question answering.
  - If you suggest another dataset, you are responsible for building the ontology by analyzing the data; the data will be analyzed using exploratory data analysis and the ontology will be built on the results of the analysis.

## Other Requirements / Indications

- The choice of the ensemble model + small classifiers is not mandatory; you can propose other combination.
- I would prefer if we would use a library-ready ML library such as `sklearn`.
- You will prepare a logging module using `rich`, and a data downloading + processing + building module using `requests` for downloading and `pandas+numpy` for processing.
- The project work will be done using agents, and I will be the one supervising the work.

## Papers

- https://homes.cs.washington.edu/~pedrod/papers/hois.pdf
- https://academic.oup.com/bib/article-pdf/22/4/bbaa199/39132158/bbaa199.pdf
- https://arxiv.org/pdf/2401.07744
