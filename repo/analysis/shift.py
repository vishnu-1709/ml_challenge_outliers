import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
F = [c for c in te.columns if c!='id']; cls=list('ABCDE')
y = tr.target_popularity.map({c:i for i,c in enumerate(cls)}).values
# --- 1. adversarial validation
def adv(wtr=None, seed=0):
    X = pd.concat([tr[F], te[F]]); z = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    w = np.r_[wtr if wtr is not None else np.ones(len(tr)), np.ones(len(te))*1.0]
    # normalise so train total weight = len(tr)
    p = np.zeros(len(z))
    for a,b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, z):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=seed)
        m.fit(X.iloc[a], z[a], sample_weight=w[a]); p[b] = m.predict_proba(X.iloc[b])[:,1]
    return roc_auc_score(z, p, sample_weight=w), m
auc, m = adv(); print('adversarial AUC (raw):', round(auc,4))
imp = pd.Series(m.booster_.feature_importance('gain'), F).sort_values(ascending=False); print(imp.head(8).round(0).to_dict())
# id alone
print('id-only adversarial AUC:', round(roc_auc_score(np.r_[np.zeros(len(tr)),np.ones(len(te))], np.r_[tr.id, te.id]),4))
# --- 2. label-shift estimation: OOF probs from a plain (unweighted) model => calibrated to train priors
oof = np.zeros((len(tr),5)); pt = np.zeros((len(te),5))
for a,b in StratifiedKFold(5, shuffle=True, random_state=1).split(tr, y):
    mm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=24, colsample_bytree=0.5, subsample=0.75, subsample_freq=1, verbose=-1, random_state=1)
    mm.fit(tr[F].iloc[a], y[a]); oof[b] = mm.predict_proba(tr[F].iloc[b]); pt += mm.predict_proba(te[F])/5
pi_tr = np.bincount(y)/len(y)
# (a) EM (Saerens 2002)
pi = pi_tr.copy()
for it in range(200):
    q = pt * (pi/pi_tr); q /= q.sum(1, keepdims=True); new = q.mean(0)
    if np.abs(new-pi).max()<1e-7: break
    pi = new
print('train priors:', pi_tr.round(3)); print('EM test priors:', pi.round(3))
# (b) BBSE: C[i,j] = P(pred=i, y=j) on train OOF; mu = P(pred=i) on test -> solve C w = mu
pred_tr = oof.argmax(1); pred_te = pt.argmax(1)
C = np.zeros((5,5))
for i in range(5):
    for j in range(5): C[i,j] = np.mean((pred_tr==i)&(y==j))
mu = np.bincount(pred_te, minlength=5)/len(pred_te)
wts = np.linalg.lstsq(C, mu, rcond=None)[0]; print('BBSE test priors:', (wts*pi_tr/ (wts*pi_tr).sum()).round(3))
# soft BBSE with probabilities
Cs = np.array([[np.mean(oof[:,i]*(y==j)) for j in range(5)] for i in range(5)]); mus = pt.mean(0)
ws = np.linalg.lstsq(Cs, mus, rcond=None)[0]; print('soft-BBSE test priors:', (ws*pi_tr/(ws*pi_tr).sum()).round(3))
# bootstrap CI for EM
rng = np.random.RandomState(0); bs=[]
for _ in range(200):
    idx = rng.randint(0,len(pt),len(pt)); P = pt[idx]; p_ = pi_tr.copy()
    for it in range(200):
        q = P*(p_/pi_tr); q/=q.sum(1,keepdims=True); p_ = q.mean(0)
    bs.append(p_)
bs=np.array(bs); print('EM bootstrap 95% CI:\n', np.percentile(bs,[2.5,97.5],axis=0).round(3))
# --- 3. Does label shift explain the covariate shift? reweight train classes to balanced, re-run adversarial
for name, tgt in [('balanced', np.full(5,0.2)), ('EM', pi)]:
    wtr = (tgt/pi_tr)[y]; wtr = wtr/wtr.mean()
    print(f'adversarial AUC after reweighting train to {name} priors:', round(adv(wtr)[0],4))
np.save('analysis/em_priors.npy', pi)
