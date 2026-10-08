# Contributing

Install with `python -m pip install -e '.[dev]'`, then run `python -m pytest` and the three scripts in `examples/`.

Keep the base package independent of datasets, trained weights, experiment paths, plotting backends and training frameworks. An estimator change must describe its assumptions, units, intervention protocol and applicable dimensions, and include a reference case or regression check.

Use one common intervention protocol within every decomposition. Declare a finite nonnegative Syn tolerance in the output unit. Retain and count tolerance-scale negative estimates; fail on significant violations without clipping.

Check the current manuscript in Zotero before changes to PEID/EI/SPT methods. Record which manuscript/attachment and relevant definitions informed the change. Distinguish standard SPT from alternative objectives and approximate search.
