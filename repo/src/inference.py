"""Final training / prediction, driven entirely by experiments/champion.json.

Final protocol (same scheme as the original notebook, so OOF and test probabilities share one scale):
  for each member, for each seed: 5-fold stratified CV on 100% of train
     -> OOF probabilities  (final CV report; decision rule is fitted on these)
     -> every fold model is saved; test proba = mean over all fold models x seeds
  members are blended with fixed weights from champion.json, then the frozen decision rule is applied.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

import config as C
from src import data, features, models as M, optimization as O, validation as V


def load_champion(path=None):
    return json.load(open(path or C.CHAMPION_FILE))


def make_member(spec):
    n = spec["name"]
    if n == "lgbm":
        return M.lgbm(spec["params"], spec.get("cw", "balanced"), n_bags=spec.get("n_bags", 1))
    if n == "xgb":
        return M.xgb(spec.get("params"), spec.get("cw", "balanced"))
    if n == "catboost":
        return M.catboost(spec.get("params"), spec.get("cw", "balanced"))
    if n == "extratrees":
        return M.extratrees(cw=spec.get("cw", "balanced"), **spec.get("params", {}))
    raise ValueError(n)


def prepare(champ):
    train, test = data.load()
    y = data.encode_target(train)
    Xtr, Xte = train[data.raw_features(train)], test[data.raw_features(test)]
    A, (B,) = data.impute(Xtr, [Xte], champ["na"])
    A, B = features.build(A, B, champ["features"])
    assert list(A.columns) == list(B.columns)
    return train, test, y, A, B


def blend(probs, weights):
    w = np.asarray(weights, float); w = w / w.sum()
    return sum(wi * p for wi, p in zip(w, probs))


def decide(P, rule):
    if rule["rule"] == "argmax":
        return P.argmax(1)
    if rule["rule"] == "multipliers":
        return O.apply(P, np.asarray(rule["weights"]))
    raise ValueError(rule["rule"])


def train(champ, out_dir=C.ARTIFACT_DIR, verbose=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df, y, A, B = prepare(champ)
    t0 = time.time()
    oofs = []
    for mi, spec in enumerate(champ["members"]):
        mem = make_member(spec)
        oof = np.zeros((len(y), 5))
        for s in spec["seeds"]:
            o = np.zeros((len(y), 5))
            for k, (a, b) in enumerate(V.folds(y, s)):
                model = mem.fit(A.iloc[a], y[a], seed=s * 100 + k)
                o[b] = model.predict_proba(A.iloc[b])
                joblib.dump(model, out_dir / f"m{mi}_{spec['name']}_s{s}_f{k}.joblib", compress=3)
            oof += o / len(spec["seeds"])
            if verbose:
                print(f"  [{spec['name']}] seed {s}: balanced-F1 {V.macro_f1(y, o.argmax(1), 'balanced'):.4f} "
                      f"train-F1 {V.macro_f1(y, o.argmax(1), 'train'):.4f}  ({time.time()-t0:.0f}s)", flush=True)
        oofs.append(oof)
    P = blend(oofs, [m["blend_weight"] for m in champ["members"]])
    rule = dict(champ["decision"])
    if rule["rule"] == "multipliers" and rule.get("fit_on") == "final_oof":
        rule["weights"] = O.tune_multipliers(P, y, champ["regime"]).tolist()
    pred = decide(P, rule)
    report = {
        "final_oof_balanced_f1": V.macro_f1(y, pred, "balanced"),
        "final_oof_train_f1": V.macro_f1(y, pred, "train"),
        "final_oof_balanced_argmax": V.macro_f1(y, P.argmax(1), "balanced"),
        "per_class_balanced": V.per_class_f1(y, pred, "balanced").round(4).tolist(),
        "decision": rule, "feature_columns": list(A.columns), "runtime_s": round(time.time() - t0),
        "note": "final_oof_* use the same OOF rows the decision rule was fitted on when fit_on=final_oof (in-sample for the rule)",
    }
    np.save(out_dir / "oof_blend.npy", P)
    json.dump(report, open(out_dir / "train_report.json", "w"), indent=1)
    json.dump(rule, open(out_dir / "decision_rule.json", "w"), indent=1)
    json.dump(list(A.columns), open(out_dir / "feature_columns.json", "w"), indent=1)
    return report


def predict(champ, out_dir=C.ARTIFACT_DIR, sub_path=C.SUBMISSION_CSV):
    _, test_df, _, _, B = prepare(champ)
    cols = json.load(open(out_dir / "feature_columns.json"))
    assert list(B.columns) == cols, "feature columns differ from training"
    probs = []
    for mi, spec in enumerate(champ["members"]):
        files = sorted(out_dir.glob(f"m{mi}_{spec['name']}_s*_f*.joblib"))
        assert len(files) == len(spec["seeds"]) * C.N_FOLDS, f"expected {len(spec['seeds'])*C.N_FOLDS} models, found {len(files)}"
        probs.append(np.mean([joblib.load(f).predict_proba(B) for f in files], axis=0))
    P = blend(probs, [m["blend_weight"] for m in champ["members"]])
    rule = json.load(open(out_dir / "decision_rule.json"))
    pred = decide(P, rule)
    sub = pd.DataFrame({C.ID_COL: test_df[C.ID_COL].values, C.TARGET: np.array(C.CLASSES)[pred]})
    # ---- submission checks
    assert len(sub) == len(test_df) == len(pred)
    assert sub[C.ID_COL].nunique() == len(test_df)
    assert set(sub[C.ID_COL]) == set(test_df[C.ID_COL]), "missing ids"
    assert not sub[C.ID_COL].duplicated().any()
    assert sub[C.TARGET].isin(C.CLASSES).all()
    assert list(sub.columns) == [C.ID_COL, C.TARGET]
    sub.to_csv(sub_path, index=False)
    np.save(out_dir / "test_proba.npy", P)
    return sub, P
