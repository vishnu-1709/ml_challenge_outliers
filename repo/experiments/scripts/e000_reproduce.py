"""E000: reproduce the original notebook exactly (full train, 5 seeds x 5 folds, seeds 100+s)."""
import sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd, lightgbm as lgb, json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from src import data, features, models, validation as V
train, test = data.load(); y = data.encode_target(train)
TR, TE = features.build(train[data.raw_features(train)], test[data.raw_features(test)], 'v1')
X, Xt = TR, TE
assert X.shape[1] == 71
params = dict(**models.LGBM_ORIG)
macro = lambda a, b: f1_score(a, b, average='macro')
oof = np.zeros((len(X), 5)); tst = np.zeros((len(Xt), 5)); t0 = time.time()
for s in range(5):
    o = np.zeros((len(X), 5))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + s).split(X, y):
        m = lgb.LGBMClassifier(**params, class_weight='balanced', random_state=s, verbose=-1).fit(X.iloc[a], y[a])
        o[b] = m.predict_proba(X.iloc[b]); tst += m.predict_proba(Xt) / 25
    oof += o / 5; print(f'seed {s}: {macro(y, o.argmax(1)):.4f}  ({time.time()-t0:.0f}s)', flush=True)
print('seed-avg OOF argmax (train regime):', round(macro(y, oof.argmax(1)), 4))
def tune_weights(P, yt, rounds=3):
    w = np.ones(5)
    for _ in range(rounds):
        for k in range(5):
            best_s, best_v = macro(yt, (P*w).argmax(1)), w[k]
            for v in np.linspace(0.3, 4, 38):
                w2 = w.copy(); w2[k] = v; s_ = macro(yt, (P*w2).argmax(1))
                if s_ > best_s: best_s, best_v = s_, v
            w[k] = best_v
    return w
idx = np.random.RandomState(0).permutation(len(y)); H = [idx[:len(y)//2], idx[len(y)//2:]]
honest = np.mean([macro(y[H[1-i]], (oof[H[1-i]]*tune_weights(oof[H[i]], y[H[i]])).argmax(1)) for i in (0, 1)])
w = tune_weights(oof, y)
print('honest tuned (train regime):', round(honest, 4), '| weights', w.round(2))
print('balanced-regime argmax:', round(V.macro_f1(y, oof.argmax(1), 'balanced'), 4), ' with notebook weights:', round(V.macro_f1(y, (oof*w).argmax(1), 'balanced'), 4))
pred = (tst * w).argmax(1)
orig = pd.read_csv('submission.csv'); ours = pd.Series(np.array(list('ABCDE'))[pred], index=test.id)
agree = (orig.set_index('id').target_popularity == ours.reindex(orig.id).values).mean()
print('agreement with committed submission.csv:', round(agree, 4))
print('test pred dist:', np.round(np.bincount(pred, minlength=5) / len(pred), 3))
np.savez_compressed('experiments/results/E000.npz', oof=oof, tst=tst, w=w)
json.dump({'argmax_train': macro(y, oof.argmax(1)), 'honest_train': honest, 'bal_argmax': V.macro_f1(y, oof.argmax(1), 'balanced'),
           'weights': w.tolist(), 'agreement_with_submission': agree}, open('experiments/results/E000.json', 'w'), indent=1)
