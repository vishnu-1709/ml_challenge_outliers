"""Metrics and the cross-validation harness.

Two evaluation regimes:
  train    : ordinary macro F1 on the evaluation rows
  balanced : macro F1 where each row gets weight 1/(K * freq of its class in the evaluation rows),
             i.e. the score we would get on a class-balanced test set drawn from the same p(x|y).
"""
import time
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import config as C

K = len(C.CLASSES)


def row_weights(y, regime):
    if regime == "train":
        return np.ones(len(y))
    freq = np.bincount(y, minlength=K) / len(y)
    return (1.0 / K) / freq[y]


def per_class_f1(y, pred, regime="balanced"):
    """Fast weighted per-class F1 (identical to sklearn f1_score(average=None, sample_weight=...))."""
    w = row_weights(y, regime)
    cm = np.bincount(y * K + pred, weights=w, minlength=K * K).reshape(K, K)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)


def macro_f1(y, pred, regime="balanced"):
    return per_class_f1(y, pred, regime).mean()


def confusion(y, pred):
    return confusion_matrix(y, pred, labels=range(K))


def folds(y, seed):
    return list(StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def run_cv(fit_predict, X, y, seeds=None, verbose=True):
    """fit_predict(X_tr, y_tr, X_va, seed) -> proba (n_va, K).
    Returns dict with oof (R, n, K), fold_id (R, n), seeds, runtime."""
    seeds = seeds or C.CV_REPEAT_SEEDS
    R = len(seeds)
    oof = np.zeros((R, len(y), K))
    fold_id = np.zeros((R, len(y)), int)
    t0 = time.time()
    for r, s in enumerate(seeds):
        for k, (a, b) in enumerate(folds(y, s)):
            oof[r, b] = fit_predict(X.iloc[a], y[a], X.iloc[b], seed=s * 100 + k)
            fold_id[r, b] = k
        if verbose:
            print(f"  repeat seed {s}: train-F1 {macro_f1(y, oof[r].argmax(1), 'train'):.4f} | "
                  f"balanced-F1 {macro_f1(y, oof[r].argmax(1), 'balanced'):.4f}  ({time.time()-t0:.0f}s)")
    return {"oof": oof, "fold_id": fold_id, "seeds": seeds, "runtime": time.time() - t0}


def fold_scores(y, preds, fold_id, regime):
    """preds (R, n) int -> list of per-fold macro F1 (R*K_folds values)."""
    out = []
    for r in range(preds.shape[0]):
        for k in range(C.N_FOLDS):
            m = fold_id[r] == k
            out.append(macro_f1(y[m], preds[r, m], regime))
    return np.array(out)


def summarize(y, preds, fold_id, regime):
    fs = fold_scores(y, preds, fold_id, regime)
    pooled = np.mean([macro_f1(y, preds[r], regime) for r in range(preds.shape[0])])
    pcf = np.mean([per_class_f1(y, preds[r], regime) for r in range(preds.shape[0])], axis=0)
    return {"macro_f1": pooled, "fold_mean": fs.mean(), "fold_std": fs.std(ddof=1), "per_class": pcf, "folds": fs}


def paired(fs_new, fs_ref):
    d = np.asarray(fs_new) - np.asarray(fs_ref)
    return {"mean_diff": d.mean(), "sd_diff": d.std(ddof=1), "se": d.std(ddof=1) / np.sqrt(len(d)),
            "wins": int((d > 0).sum()), "n": len(d)}
