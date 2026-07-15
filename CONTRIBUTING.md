# Contributing

Development uses [pixi](https://pixi.sh): `pixi install`, then `pixi run test`
and `pixi run lint`. CI additionally runs every README code block
(`tests/test_readme.py`) and the viz tests against the oldest supported
matplotlib.

For pull requests:

- New behavior needs tests. Anything that changes scores needs a check
  against exact refitting; `tests/test_functionals.py` shows the pattern.
- Warning and error messages are API: tests match on their text.
- Docstrings follow scikit-learn conventions (numpydoc).
