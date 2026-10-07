"""Train the frozen champion (experiments/champion.json) on 100% of train.csv and save fold models.
usage: python train.py"""
import json

import numpy as np

import config as C
from src import data, inference, optimization as O, validation as V

if __name__ == "__main__":
    champ = inference.load_champion()
    print("champion:", champ["experiment_id"], "-", champ["description"])
    rep = inference.train(champ)
    # honest estimate of the decision rule: cross-fit the multipliers over a fresh 5-fold split of the OOF rows
    P = np.load(C.ARTIFACT_DIR / "oof_blend.npy")
    train, _ = data.load()
    y = data.encode_target(train)
    fid = np.zeros((1, len(y)), int)
    for k, (_, b) in enumerate(V.folds(y, 999)):
        fid[0, b] = k
    cf, _ = O.crossfit_predict(P[None], y, fid, champ["regime"])
    rep["final_oof_balanced_f1_crossfit"] = V.macro_f1(y, cf[0], "balanced")
    rep["final_oof_train_f1_crossfit"] = V.macro_f1(y, cf[0], "train")
    json.dump(rep, open(C.ARTIFACT_DIR / "train_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "feature_columns"}, indent=1))
