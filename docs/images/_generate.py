"""Regenerate the README figures.

Run from the repo root:
    PYTHONPATH=. pixi run python docs/images/_generate.py

Uses the same synthetic label-corruption scenario as examples/showcase.ipynb.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from pyinfluence import InfluenceFunctions, removal_curve, self_influence, viz

OUT = Path(__file__).parent

rng = np.random.default_rng(0)
n_train, n_test, p = 200, 50, 8

X_train = rng.normal(size=(n_train, p))
beta = rng.normal(size=p)
y_clean = X_train @ beta + 0.1 * rng.normal(size=n_train)
sources = np.array(["A", "B", "C", "D"]).repeat(50)
c_idx = np.where(sources == "C")[0]
corrupted = rng.choice(c_idx, size=10, replace=False)
y_train = y_clean.copy()
y_train[corrupted] += rng.choice([-1, 1], size=10) * (8 + 4 * rng.uniform(size=10))
X_test = rng.normal(size=(n_test, p))
y_test = X_test @ beta + 0.1 * rng.normal(size=n_test)

model = Ridge(alpha=1.0).fit(X_train, y_train)
attr = InfluenceFunctions(mode="loss", damping=1e-3).fit(model, X_train, y_train)

# --- report dashboard ---------------------------------------------------------
train_err = np.abs(model.predict(X_train) - y_train)
fig = viz.report(
    attr, X_test, y_test, groups=sources, errors=train_err,
    test_idx=int(np.argmax(np.abs(model.predict(X_test) - y_test))), k=8, top_k=20,
)
fig.savefig(OUT / "report.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# --- validation curves: removal + detection -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
curve = removal_curve(
    attr, X_test, y_test, fractions=np.linspace(0.0, 0.3, 7),
    direction="harmful", n_random=5, random_state=0,
)
viz.plot_removal_curve(curve, ax=axes[0])

self_inf = self_influence(attr)
is_corrupted = np.zeros(n_train, dtype=bool)
is_corrupted[corrupted] = True
viz.plot_detection_curve(self_inf, is_corrupted, ax=axes[1])
fig.tight_layout()
fig.savefig(OUT / "validation_curves.png", dpi=110, bbox_inches="tight")
plt.close(fig)

print(f"wrote {OUT / 'report.png'} and {OUT / 'validation_curves.png'}")
