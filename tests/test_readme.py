"""The README is executable documentation: every python block must run.

Blocks are executed cumulatively in one namespace, exactly as a reader
typing along would experience them. If a block needs setup the README
doesn't provide, that is a documentation bug and this test fails.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.md"

try:
    import matplotlib

    matplotlib.use("Agg")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@pytest.mark.slow
@pytest.mark.skipif(not HAS_MPL, reason="README blocks use pyinfluence.viz")
def test_readme_blocks_run_cumulatively():
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)
    assert len(blocks) >= 5, "README lost its code blocks?"
    ns: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, block in enumerate(blocks):
            try:
                exec(compile(block, f"<README block {i}>", "exec"), ns)
            except Exception as e:  # pragma: no cover - failure reporting
                pytest.fail(
                    f"README python block {i} failed when run as written: "
                    f"{type(e).__name__}: {e}\n---\n{block[:400]}"
                )
