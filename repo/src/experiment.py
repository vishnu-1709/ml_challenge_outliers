"""Experiment runner + logger. Every run is tied to a hypothesis and appended to experiments/experiment_log.csv.

Reported numbers (never mixed):
  DEV CV      : 5-fold x 3 repeats on the 85% development set (OOF)
  DEV CV + cf : same OOF, with decision multipliers tuned by cross-fitting (other folds only)
  CONFIRMATION: the 15% holdout, used only for finalists (see confirm())
"""
import csv
import datetime as dt
import json
import resource
import time

import numpy as np
import pandas as pd

import config as C
from src import data, features, validation as V, optimization as O

LOG = C.EXP_DIR / "experiment_log.csv"
RES = C.EXP_DIR / "results"
COLS = ["experiment_id", "date", "hypothesis", "model", "features", "na_handling", "validation_protocol", "parameters",
        "decision_rule", "macro_f1", "A_f1", "B_f1", "C_f1", "D_f1", "E_f1", "fold_mean", "fold_std",
        "bal_argmax", "train_argmax", "train_cf", "diff_vs_ref", "se_diff", "fold_wins", "ref",
        "runtime_s", "peak_mem_mb", "result", "decision"]

_CACHE = {}


def dev_data():
    if "dev" not in _CACHE:
        train, test = data.load()
        y = data.encode_target(train)
        dev, hold = data.holdout_split(y)
        Xraw = train[data.raw_features(train)]
        _CACHE["dev"] = (Xraw, y, dev, hold, test)
    return _CACHE["dev"]


def pipeline(fit_predict, feat="v1", na="native"):
    """Wrap a model so that NA handling + feature building happen inside each fold."""
    def fp(Xa, ya, Xb, seed):
        A, (B,) = data.impute(Xa, [Xb], na)
        A, B = features.build(A, B, feat)
        return fit_predict(A, ya, B, seed)
    return fp


def evaluate_oof(res, y):
    oof, fid = res["oof"], res["fold_id"]
    out = {}
    for regime in ("balanced", "train"):
        am = oof.argmax(2)
        cf, ws = O.crossfit_predict(oof, y, fid, regime)
        out[regime] = {"argmax": V.summarize(y, am, fid, regime), "cf": V.summarize(y, cf, fid, regime), "w": ws.mean(0)}
    return out


def run(exp_id, hypothesis, model_name, fit_predict, feat="v1", na="native", params=None, ref="E001",
        seeds=None, save=True):
    Xraw, y, dev, hold, _ = dev_data()
    Xd, yd = Xraw.iloc[dev].reset_index(drop=True), y[dev]
    print(f"\n=== {exp_id}: {hypothesis}")
    t0 = time.time()
    res = V.run_cv(pipeline(fit_predict, feat, na), Xd, yd, seeds)
    ev = evaluate_oof(res, yd)
    runtime = time.time() - t0
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    prim = ev["balanced"]["cf"]
    row = {"experiment_id": exp_id, "date": dt.date.today().isoformat(), "hypothesis": hypothesis, "model": model_name,
           "features": feat, "na_handling": na,
           "validation_protocol": f"dev85% {C.N_FOLDS}fold x{len(res['seeds'])} rep; balanced-regime; cross-fit multipliers",
           "parameters": json.dumps(params, default=str) if params is not None else "",
           "decision_rule": "cf-multipliers", "macro_f1": prim["macro_f1"],
           **{f"{c}_f1": prim["per_class"][i] for i, c in enumerate(C.CLASSES)},
           "fold_mean": prim["fold_mean"], "fold_std": prim["fold_std"],
           "bal_argmax": ev["balanced"]["argmax"]["macro_f1"], "train_argmax": ev["train"]["argmax"]["macro_f1"],
           "train_cf": ev["train"]["cf"]["macro_f1"], "runtime_s": round(runtime), "peak_mem_mb": round(mem),
           "ref": ref, "result": "", "decision": ""}
    if ref and (RES / f"{ref}.npz").exists() and ref != exp_id:
        r = np.load(RES / f"{ref}.npz")
        if list(r["seeds"]) == list(res["seeds"]):
            pr = V.paired(prim["folds"], r["folds_bal_cf"])
            row.update(diff_vs_ref=pr["mean_diff"], se_diff=pr["se"], fold_wins=f"{pr['wins']}/{pr['n']}")
    print(f"  BALANCED  argmax {ev['balanced']['argmax']['macro_f1']:.4f} | cf {prim['macro_f1']:.4f} "
          f"(fold {prim['fold_mean']:.4f} ± {prim['fold_std']:.4f})  per-class {np.round(prim['per_class'], 3)}")
    print(f"  TRAIN-MIX argmax {ev['train']['argmax']['macro_f1']:.4f} | cf {ev['train']['cf']['macro_f1']:.4f}")
    if row.get("diff_vs_ref") not in (None, ""):
        print(f"  vs {ref}: {row['diff_vs_ref']:+.4f} (SE {row['se_diff']:.4f}, wins {row['fold_wins']})")
    print(f"  mean cf weights (balanced): {np.round(ev['balanced']['w'], 2)}  runtime {runtime:.0f}s")
    if save:
        RES.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(RES / f"{exp_id}.npz", oof=res["oof"].astype(np.float32), fold_id=res["fold_id"],
                            seeds=np.array(res["seeds"]), folds_bal_cf=prim["folds"],
                            folds_bal_am=ev["balanced"]["argmax"]["folds"], folds_train_cf=ev["train"]["cf"]["folds"])
        _append(row)
    return res, ev, row


def _append(row):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    rows = [] if new else list(csv.DictReader(open(LOG)))
    rows = [r for r in rows if r["experiment_id"] != row["experiment_id"]]
    rows.append({k: (f"{v:.5f}" if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)


def decide(exp_id, result, decision):
    rows = list(csv.DictReader(open(LOG)))
    for r in rows:
        if r["experiment_id"] == exp_id:
            r["result"], r["decision"] = result, decision
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)


def load_oof(exp_id):
    return np.load(RES / f"{exp_id}.npz")
