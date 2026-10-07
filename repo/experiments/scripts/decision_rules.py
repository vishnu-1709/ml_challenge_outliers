"""Decision-rule research on saved OOF probabilities (no retraining). All tuned rules are cross-fitted.

usage: python experiments/scripts/decision_rules.py E003 [E007 ...]

Rules
  argmax         : argmax P
  prior          : argmax P * (pi_target / pi_model)   (no tuning)
  mult           : 5 per-class multipliers, coordinate ascent      (4 free params)
  logit_tau      : argmax log P + tau * log(pi_target/pi_train)     (1 free param)
  ordinal_cut    : expected class index E[k] = sum_k k P_k, 4 cut points tuned (4 free params)
  Temperature scaling is not tested separately: argmax(P^(1/T) * w) == argmax(P * w^T),
  so it is exactly a re-parameterisation of 'mult'.

Robustness: every rule is tuned under one regime and scored under BOTH regimes.
"""
import sys; sys.path.insert(0, '.')
import numpy as np
from src import experiment as X, validation as V, optimization as O

K = 5


def tune_tau(P, y, regime, pi_train, pi_target):
    lr = np.log(pi_target / pi_train)
    best, bt = -1, 0
    for t in np.linspace(-1, 2, 61):
        s = V.macro_f1(y, (np.log(P + 1e-12) + t * lr).argmax(1), regime)
        if s > best: best, bt = s, t
    return bt


def cut_predict(P, cuts):
    e = P @ np.arange(K)
    return np.searchsorted(np.sort(cuts), e)


def tune_cuts(P, y, regime, rounds=3):
    e = P @ np.arange(K)
    cuts = np.quantile(e, np.cumsum(np.bincount(y, minlength=K) / len(y))[:-1])
    best = V.macro_f1(y, np.searchsorted(cuts, e), regime)
    for _ in range(rounds):
        for j in range(K - 1):
            lo = cuts[j - 1] if j > 0 else e.min(); hi = cuts[j + 1] if j < K - 2 else e.max()
            for v in np.linspace(lo, hi, 41)[1:-1]:
                c2 = cuts.copy(); c2[j] = v
                s = V.macro_f1(y, np.searchsorted(c2, e), regime)
                if s > best + 1e-9: best, cuts = s, c2
    return cuts


def crossfit(oof, y, fid, tune_regime, rule, pi_train):
    R, n, _ = oof.shape
    preds = np.zeros((R, n), int)
    for r in range(R):
        for k in np.unique(fid[r]):
            te = fid[r] == k; tr = ~te
            P_tr, P_te = oof[r, tr], oof[r, te]
            if rule == "argmax":
                p = P_te.argmax(1)
            elif rule == "prior":
                pi_model = P_tr.mean(0)  # effective prior the model predicts with
                pi_target = np.full(K, 1 / K) if tune_regime == "balanced" else pi_train
                p = (P_te * pi_target / pi_model).argmax(1)
            elif rule == "mult":
                p = O.apply(P_te, O.tune_multipliers(P_tr, y[tr], tune_regime))
            elif rule == "logit_tau":
                pi_model = P_tr.mean(0); tgt = np.full(K, 1 / K)
                t = tune_tau(P_tr, y[tr], tune_regime, pi_model, tgt)
                p = (np.log(P_te + 1e-12) + t * np.log(tgt / pi_model)).argmax(1)
            elif rule == "ordinal_cut":
                p = cut_predict(P_te, tune_cuts(P_tr, y[tr], tune_regime))
            preds[r, te] = p
    return preds


if __name__ == "__main__":
    Xraw, y_all, dev, hold, _ = X.dev_data()
    y = y_all[dev]; pi_train = np.bincount(y, minlength=K) / len(y)
    for eid in sys.argv[1:]:
        r = X.load_oof(eid); oof, fid = r["oof"].astype(float), r["fold_id"]
        print(f"\n##### {eid}   (cells: macro F1 pooled-OOF, fold mean±sd in brackets)")
        print(f"{'rule':12s} {'tuned for':9s} | {'score: BALANCED':28s} | {'score: TRAIN-MIX':28s} | pred dist (balanced-tuned)")
        for rule in ["argmax", "prior", "mult", "logit_tau", "ordinal_cut"]:
            for treg in (["balanced"] if rule == "argmax" else ["balanced", "train"]):
                pr = crossfit(oof, y, fid, treg, rule, pi_train)
                sb = V.summarize(y, pr, fid, "balanced"); st = V.summarize(y, pr, fid, "train")
                dist = np.bincount(pr[0], minlength=K) / len(y)
                print(f"{rule:12s} {treg if rule!='argmax' else '-':9s} | {sb['macro_f1']:.4f} [{sb['fold_mean']:.4f}±{sb['fold_std']:.4f}] | "
                      f"{st['macro_f1']:.4f} [{st['fold_mean']:.4f}±{st['fold_std']:.4f}] | {np.round(dist, 3)}", flush=True)
