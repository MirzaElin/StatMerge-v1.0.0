---
title: 'StatMerge: unified statistical analysis and inter-rater reliability for research'
tags:
  - Python
  - statistics
  - reliability
  - agreement
  - reproducibility
authors:
  - name: Mirza Niaz Zaman Elin
  - orcid: https://orcid.org/0000-0001-9577-7821
    affiliation: 1
affiliations:
  - name: Research and Development Advisor, AMAL Youth & Family Centre, St. John's, NL, Canada
    index: 1
date: 2025-09-06
bibliography: paper.bib
---

# Summary

**StatMerge** is a consolidated Python package and desktop application that brings together day–to–day research statistics with specialist routines for inter‑rater agreement and reliability. It merges two previously independent code bases into a coherent library and GUI so that researchers, students, and practitioners can compute common effect sizes, confusion‑matrix‑based diagnostics, ROC/PR summaries, and agreement statistics such as Fleiss' kappa and intraclass correlation coefficients (ICC) within a single, documented tool. The package emphasizes pragmatic reproducibility: every algorithm is accessible from the Python API for scripted workflows and also from a minimal cross‑platform GUI for teaching, demonstrations, and quick exploratory data analysis. Where possible, StatMerge provides references to standard definitions and implements widely used small‑sample corrections and confidence intervals, such as Hedges' \(g\) and Wilson intervals for binomial proportions [@hedges1985; @wilson1927].

# Statement of need

Many applied research projects rely on a scattered collection of small scripts to perform tasks like computing Cohen's \(d\), checking model sensitivity/specificity from a table of predictions, or measuring the agreement of multiple annotators on categorical labels[@Elin2025]. These scripts are often ad hoc, inconsistently documented, and hard to reuse or validate. Existing libraries cover parts of this space—e.g., scikit‑learn for ROC/PR utilities [@pedregosa2011], pandas for data handling [@mckinney2010], and domain‑specific packages for reliability—but there is value in a cohesive, light‑weight toolkit that puts the most common analyses and agreement metrics behind a stable API, accompanies them with a small GUI, and ships with tests and examples. StatMerge aims to meet that need. It reduces friction for researchers who switch between notebook‑based computation and quick desktop exploration, and it offers a compact entry point for teaching methodological concepts such as effect sizes, rating agreement, and diagnostic trade‑offs on ROC vs. precision–recall curves [@fawcett2006; @davis2006].

# Design and scope

The package follows a simple structure under `src/statmerge`. The `analysis` module provides general‑purpose effect sizes (Cohen's \(d\) and Hedges' \(g\)), deliberately implemented with transparent formulas to ease review and teaching [@cohen1988; @hedges1985]. The `metrics_mod` module focuses on agreement and diagnostic utilities: summary measures derived from confusion counts (accuracy, sensitivity/TPR, specificity/TNR, PPV/NPV, \(F_1\), balanced accuracy, Youden's \(J\), and Matthews correlation coefficient), plus binomial Wilson confidence intervals for proportions [@wilson1927]. It also includes helpers to generate ROC and precision–recall points and to compute area‑under‑curve summaries using the scikit‑learn reference implementations when available [@pedregosa2011]. For rater agreement, StatMerge implements Fleiss' kappa from raw label tallies for multiple raters [@fleiss1971] and the ICC(2,1) consistency model for continuous ratings, with an optional bootstrap to approximate confidence intervals [@shrout1979; @koo2016].

A small Qt‑based desktop application (PySide6) exposes these capabilities to non‑programmers. The GUI bundles a “Data Lab” panel that can load a CSV, preview and describe columns, compute simple correlations, render quick plots via Matplotlib [@hunter2007], and export a short DOCX report. Two buttons launch the legacy graphical tools that motivated this consolidation; they are shipped as vendored modules and loaded in‑process with a thin compatibility layer so that users can continue familiar workflows while benefiting from the unified codebase.

The design choices prioritize clarity over novelty. All public functions accept plain Python sequences or NumPy/pandas arrays as appropriate, return lightweight Python objects (e.g., dictionaries for confusion‑derived summaries), and avoid heavy dependencies beyond NumPy, pandas, scikit‑learn, and Matplotlib [@harris2020; @mckinney2010; @pedregosa2011; @hunter2007]. Statistical formulas are implemented explicitly and cross‑checked in tests against closed‑form results or reference implementations to facilitate review.

# Functionality

The principal features include:

- **Effect sizes**: Cohen's \(d\) with a pooled‑variance denominator and Hedges' \(g\) small‑sample correction \(J = 1 - 3/(4N-9)\), suitable for teaching standardized mean differences and for quick meta‑analytic sketches [@cohen1988; @hedges1985].
- **Binary classification summaries**: accuracy, sensitivity (TPR/recall), specificity (TNR), positive and negative predictive values, \(F_1\), balanced accuracy, Youden's \(J\), and Matthews correlation coefficient. Each proportion is accompanied by a Wilson score interval that behaves well for small samples and extreme rates [@wilson1927].
- **ROC/PR utilities**: generation of ROC and precision–recall points and their areas under the curve via scikit‑learn wrappers, enabling consistent comparison across models and thresholds [@fawcett2006; @davis2006; @pedregosa2011].
- **Inter‑rater agreement**: Fleiss' kappa from raw categorical ratings for any number of raters and categories, returning per‑item agreement values and marginal label proportions to aid diagnostics [@fleiss1971].
- **Reliability for continuous ratings**: ICC(2,1) (two‑way random effects, single measurement) following Shrout and Fleiss's variance‑component decomposition, with optional bootstrap confidence intervals for exploratory uncertainty quantification [@shrout1979; @koo2016].
- **Desktop GUI**: a launcher that opens the Data Lab and the two legacy tools; the Data Lab reads CSVs, previews/cleans basic structure, computes descriptive summaries and correlations, plots simple histograms/scatter plots, and exports a short Word report with key tables and (optionally) the last plot.
- **Reproducibility aids**: a tiny command‑line entry point to launch the GUI (`statmerge`), unit tests for the numerical routines, and a CI workflow that runs across multiple Python versions in headless mode.

# Implementation and architecture

Internally, StatMerge keeps analysis and agreement code separate to avoid hidden coupling. Agreement utilities are self‑contained and can be imported independently of the GUI, making it straightforward to use them in notebooks or pipelines. The desktop code is intentionally thin: it delegates computations to the library and focuses on layout and file I/O. The legacy graphical tools are loaded via adapters that ensure missing dependencies do not crash the main application. Where those tools expect a `metrics` module, StatMerge provides a compatibility layer that maps to its own tested implementations. The GUI is implemented with PySide6 but does not expose binding details in the user interface.

The choice to implement Wilson score intervals, Hedges' correction, and the Shrout–Fleiss ICC explicitly is pedagogical as much as practical. These formulas are compact, traceable to the literature, and easy to verify in unit tests. Where high‑quality reference implementations exist (e.g., scikit‑learn’s ROC/PR), StatMerge calls into them to reduce duplication and risk.

# Quality control and validation

Unit tests cover three levels. First, **consistency tests** verify shape and type constraints (e.g., that `binary_metrics` returns counts that sum to \(N\) and that keys are present). Second, **numerical tests** check closed‑form identities on small synthetic inputs (e.g., that balanced accuracy equals the average of sensitivity and specificity, that the Wilson interval is symmetrical when \(p=0.5\), and that Cohen's \(d\) reduces to zero when the means are equal). Third, **cross‑reference tests** compare outputs against scikit‑learn for ROC/PR AUC on randomly generated inputs to ensure glue code is correct. The CI workflow runs these tests on Linux across Python 3.10–3.13 in a headless Qt environment to catch import and backend issues early.

For agreement metrics, we include example matrices that reproduce values near those reported in the literature (e.g., moderate kappa for mixed agreement scenarios) and sanity‑check the impact of marginal label imbalance on kappa. For ICC, we verify that the implementation responds sensibly to added rater noise and that bootstrap intervals shrink with more items, as expected.

# Usage examples

A typical scripted workflow computes agreement on raw categorical ratings and then explores classifier diagnostics:

```python
from statmerge.metrics_mod.agreement import fleiss_kappa_from_raw, binary_metrics
ratings = [['A','A','B'], ['B','B','B'], ['A','B','B'], ['A','A','A']]
kappa, per_item, marginals = fleiss_kappa_from_raw(ratings)

y_true = [1,1,0,0,1,0,1,0]
y_pred = [1,0,0,0,1,1,1,0]
report = binary_metrics(y_true, y_pred)
```

Educators can instead launch the GUI (`statmerge`) to load a CSV of annotations or predictions, preview distributions, and show the interpretive differences between ROC and precision–recall curves when class imbalance is severe [@davis2006]. The exported DOCX report is intended for quick lab records rather than publication‑ready figures.

# Availability and community guidelines

StatMerge is distributed under the MIT license. The repository includes a contribution guide and a code of conduct to set expectations for respectful collaboration. We welcome small, focused pull requests that add tests alongside new functionality or bug fixes. Issue templates encourage minimal reproducible examples and references when proposing statistical extensions. A `CITATION.cff` file provides standard metadata for citation.

# Limitations and future work

StatMerge intentionally covers a pragmatic subset of research statistics. Several natural extensions remain out of scope for the initial release but would be valuable contributions: additional ICC models (e.g., ICC(1) and ICC(3)), weighted kappas for ordinal data, permutation‑ or bootstrap‑based confidence intervals for kappa, power/sample‑size calculators for agreement studies, and more extensive plotting utilities. We also plan to add optional integration notebooks that demonstrate end‑to‑end workflows on small public datasets.

# Conflict of Interests

The author declares no conflict of interests regarding this work.

# References
