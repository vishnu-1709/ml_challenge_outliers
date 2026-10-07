"""Macro-F1 decision rules, all tuned with cross-fitting (never scored on the rows they were tuned on).

A decision rule maps a probability matrix P (n, K) to labels.
  argmax       : argmax P
  multipliers  : argmax P * w            (w tuned by coordinate ascent, log-spaced grid)
  prior        : argmax P * pi_target / pi_model   (no tuning, pure prior correction)
"""
import numpy as np

from src.validation import macro_f1, K

GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 41))


def apply(P, w):
    return (P * w).argmax(1)


def tune_multipliers(P, y, regime, rounds=3, grid=GRID):
    """Coordinate ascent on per-class multipliers, w[ref] fixed at 1 (scale invariance)."""
    w = np.ones(K)
    best = macro_f1(y, apply(P, w), regime)
    for _ in range(rounds):
        improved = False
        for k in range(K):
            for v in grid:
                w2 = w.copy(); w2[k] = v
                s = macro_f1(y, apply(P, w2), regime)
                if s > best + 1e-9:
                    best, w, improved = s, w2, True
        if not improved:
            break
    return w / w.max()


def crossfit_predict(oof, y, fold_id, regime, tuner=tune_multipliers):
    """For every repeat r and fold k: tune on the OOF rows of the *other* folds, apply to fold k.
    Returns preds (R, n) and the list of fitted weight vectors."""
    R, n, _ = oof.shape
    preds = np.zeros((R, n), int)
    ws = []
    for r in range(R):
        for k in np.unique(fold_id[r]):
            te = fold_id[r] == k
            w = tuner(oof[r, ~te], y[~te], regime)
            preds[r, te] = apply(oof[r, te], w)
            ws.append(w)
    return preds, np.array(ws)


def prior_weights(pi_model, pi_target):
    return np.asarray(pi_target) / np.asarray(pi_model)
