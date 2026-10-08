# AGENTS.md

## Git workflow

- Unless the user explicitly asks for a separate branch, work directly on the current main branch and update it in place.
- Do not create a feature branch or worktree by default.
- After completing work on the main branch, do not ask whether to merge, open a pull request, keep the branch, or discard the work.
- Only use a separate branch, worktree, pull request, or branch-completion workflow when the user explicitly requests it.

## Delegation and heavy workflows

- Before invoking subagents or starting a materially complex workflow, first explain in writing why its scope is necessary, what lighter alternative exists, and its expected cost.
- Obtain the user's explicit approval before proceeding. If the user considers the additional rigor unnecessary, use the lighter approach and do not expand the work.

## Long-running experiments

- For long or expensive experiments, persist reusable computed results only when recomputation would be costly or the user asks for reusable data.
- Prefer lightweight cache formats that match the code path, such as `json`, `jsonl`, `npz`, or notebook-adjacent cache files. Do not create `csv` summaries by default.
- Treat machine-readable caches as internal support artifacts; the user-facing deliverable should prioritize the visualization figure and concise interpretation.
- If an experiment is executed from a notebook, make sure it can reuse existing cache artifacts on subsequent runs when such artifacts are needed.

## EI estimation

- When computing EI over continuous variable spaces, prefer TM-based estimation first.
- Use an alternative EI estimator only when TM is inapplicable, computationally prohibitive, or explicitly requested.
- If using a non-TM estimator, state the reason and document the tradeoff.

## Syn nonnegativity

- Treat PEID Syn as nonnegative by definition.
- Never apply an undocumented `max(0, Syn)`, clipping, or another silent projection to enforce nonnegativity.
- Each numerical experiment that consumes estimated Syn must declare a nonnegative tolerance in the native Syn units. Values in `[-tolerance, 0)` may be treated as numerical zero, but the tolerance and affected count must be recorded.
- Any Syn estimate below `-tolerance` is a significant nonnegativity violation. Downstream code must fail explicitly and report the minimum value, threshold, and affected count.
- A tolerance rule absorbs numerical estimation error only; it must not be described as evidence for negative Syn.

## Current manuscript and code consistency

- For tasks involving PEID, EI/Syn, SPT, or related theory, method design, implementation, experiments, or interpretation, first use Zotero to locate *Emergent hierarchical organization of causal interactions in complex systems* and read the relevant sections of its latest available manuscript. This manuscript is the primary reference for the project's current methods. Its current Zotero parent item key is `P6UJCVG8`; use it as a locator, verify the title, and search by title if the item has moved or been replaced.
- Recheck the manuscript and its attachments on each relevant task because the user updates it over time. Select the latest draft using explicit version information or manuscript dates, supported by attachment metadata; an item metadata edit alone does not establish a new manuscript version. Do not pin a PDF attachment key, file path, arXiv version, or stale extracted-text cache. State any unresolved version ambiguity before making version-dependent method decisions.
- Before derivations or methodological code changes, read the relevant full-text definitions, equations, methods, and appendices. Record the parent/attachment keys, available version or date, and relevant section/equation locations in existing task notes or the report when those details support a decision. Do not infer method details from the title or abstract alone.
- Keep the manuscript and repository behavior consistent: check metric definitions, intervention/input distributions and supports, source/target variables, time horizons, estimator assumptions and approximations, normalization and units, Syn validity/tolerances, and hierarchy construction or stopping rules as relevant to the task. Identify any discrepancy explicitly, distinguish intended method from implementation approximation, and align the affected code, notebook, and documentation within the current task's scope; record remaining discrepancies rather than claiming full consistency.
- The older PEID arXiv paper, *Partial Effective Information Decomposition for Synergistic Causality*, may be used as historical or supplementary context, but must not silently replace the current manuscript as the methodological authority.
- If Zotero, the target manuscript, or its relevant full text is unavailable, state the exact limitation and continue only with the best available repository context. Do not claim that the latest manuscript was read or that manuscript/code consistency was verified when it was not.

## Toolbox scope

- Base dependencies remain NumPy and SciPy. Keep domain data, weights and experiment orchestration outside src/eid.
- Public results declare information units and retain raw estimated values with numerical diagnostics.
- Read docs/method_contract.md and docs/source_manifest.json for the extraction boundary; they do not establish that an old manuscript remains current.
- The package uses the MIT license approved for publication. Distribution name: eid-toolbox; import namespace: eid. Confirm remote release status before claiming publication success.
