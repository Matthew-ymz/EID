# Changelog

This is the version history for EID. Add the newest version first, with its date, main improvements and fixes, and any compatibility or migration notes. Keep source synchronization and package publication status separate.

## 0.3.1 — Unreleased, 2026-11-17

- Added opt-in RW-EI to let callers compute EI and SPT from reweighted observations without fitting a transition model. `estimate_ei` and `analyze_samples` accept `estimator="rw_ei"` with `estimator_options={"sample_weight": w, "assume_identified": True}`. The caller supplies the full-source density ratio for a declared `UniformBox` or `IndependentGaussian`; the library reuses it for every subset and reports effective sample size. The existing Syn checks, units and default search policy are unchanged.
- Added the scalar marginal TM reader with known additive Gaussian response noise to reproduce the original frequency experiment. Use `estimate_ei` with `tm_marginal`, or select `readout="marginal_tm"` under `rw_ei`. This entropy subtraction is for the full input only; it does not supply singleton EI or SPT.

## 0.3.0 — Unreleased, 2026-11-04

- Added Queyranne search for the standard SPT residual. Each split uses the shared EI cache and Syn audit.
- The default `search="auto"` uses exhaustive search on small nodes and Queyranne on larger nodes of exact discrete or analytic Gaussian channels. The Gaussian rule also applies to the fitted local linear channel in `local_gaussian`.
- Sample-based EI keeps the existing exhaustive and spectral policy under `"auto"`. Use `search="queyranne"` to opt in when the subset objective is suitable.
- Use `search="spectral"` to recover the previous search policy on all channels or `search="exact"` to exhaustively check every node. Queyranne can resolve tied optima differently.

## 0.2.0 — Unreleased, 2026-10-11

- Added a unified EI/MI estimator API and explicit configuration for discrete TPM, polynomial/affine/lifted/readout/neural TM, Gaussian, KDE, KSG, linear/local Gaussian channels, and the limited NIS Jacobian approximation.
- Added independent Gaussian and empirical block interventions, source grouping and fixed feature transforms, correlated response noise, and held-out density evaluation while preserving existing defaults.
- Implemented shared conditional neural-channel marginalization as an optional `neural` extra; base dependencies remain NumPy and SciPy.
- Retained raw estimates, units, estimator diagnostics and strict Syn tolerance checks; unsupported protocols and options fail explicitly.
- Rechecked the current Zotero manuscript attachments and documented estimator coverage, version ambiguity and domain-reproduction boundaries. Added a switching example, notebook section and analytic-reference tests.
- Added paired estimator comparisons with analytic/channel-integral references, Syn/cost and applicability figures, and six Boolean response-network diagrams with editable sources.

## 0.1.1 — 2026-10-08

- Changed the distribution name to `eid-toolbox` after PyPI rejected `eid` as too similar to an existing project.
- Kept the EID project name, `eid` import namespace and analysis algorithms.
- Updated publishing metadata and instructions; preserved the published GitHub `v0.1.0` release.

## 0.1.0 — 2026-10-08

- Prepared the first public Alpha release under the user-selected MIT license.
- Named the project EID and its distribution/import namespace `eid`, as requested by the user.
- Added repository metadata and a release workflow for PyPI Trusted Publishing.
- Kept the extracted EI/TM/SPT algorithms unchanged.

## 0.1.0.dev0 — 2026-10-08

- Extracted discrete EI/TPM utilities, transport-map density estimation and SPT from EISyn.
- Added standalone packaging, explicit intervention/result adapters and model integration.
- Added explicit unit metadata, public tolerance checks and finite-value validation.
- Migrated core regression tests and added three small examples.
- Recorded manuscript evidence, source hashes, estimator differences and pending publication work.
