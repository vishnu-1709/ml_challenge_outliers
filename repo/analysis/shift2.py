import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import nnls
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
F = [c for c in te.columns if c!='id']; cls=list('ABCDE')
y = tr.target_popularity.map({c:i for i,c in enumerate(cls)}).values
cc = ~tr[F].isna().any(axis=1).values
def adv(Xtr, Xte, wtr=None, seeds=(0,1,2)):
    X = pd.concat([Xtr, Xte]); z = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    w = np.r_[wtr if wtr is not None else np.ones(len(Xtr)), np.ones(len(Xte))]
    out=[]
    for s in seeds:
        p = np.zeros(len(z))
        for a,b in StratifiedKFold(5, shuffle=True, random_state=s).split(X, z):
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50, verbose=-1, random_state=s)
            m.fit(X.iloc[a], z[a], sample_weight=w[a]); p[b] = m.predict_proba(X.iloc[b])[:,1]
        out.append(roc_auc_score(z, p, sample_weight=w))
    return np.mean(out), np.std(out), m
pi_tr = np.bincount(y[cc], minlength=5)/cc.sum()
a,s,m = adv(tr.loc[cc,F], te[F]); print(f'adv AUC complete-case train vs test: {a:.4f} ± {s:.4f}')
print(pd.Series(m.booster_.feature_importance('gain'), F).sort_values(ascending=False).head(8).round(0).to_dict())
# null: random 2500 complete-case train rows vs rest
rng=np.random.RandomState(0); idx=np.where(cc)[0]; rng.shuffle(idx)
a0,s0,_ = adv(tr.loc[idx[2500:],F], tr.loc[idx[:2500],F]); print(f'NULL adv AUC (train vs train holdout): {a0:.4f} ± {s0:.4f}')
for name,tgt in [('balanced',np.full(5,.2))]:
    w=(tgt/pi_tr)[y[cc]]; w/=w.mean(); a,s,_=adv(tr.loc[cc,F], te[F], w); print(f'adv AUC, train reweighted to {name}: {a:.4f} ± {s:.4f}')
    # null under reweighting: does reweighting alone change AUC? reweight a train-vs-train split
    w2=(tgt/pi_tr)[y[idx[2500:]]]; w2/=w2.mean(); a2,s2,_=adv(tr.loc[idx[2500:],F], tr.loc[idx[:2500],F], w2); print(f'  (null with reweighting, train vs train: {a2:.4f})')
# moment matching on rank-transformed features: test mean = sum_k pi_k * classmean_k
Xall = pd.concat([tr[F], te[F]]).rank(pct=True)
R_tr, R_te = Xall.iloc[:len(tr)].values, Xall.iloc[len(tr):].values
M = np.array([np.nanmean(R_tr[y==k],0) for k in range(5)]).T   # features x classes
mu = R_te.mean(0)
A = np.vstack([M, 100*np.ones((1,5))]); b = np.r_[mu, 100]
pi_mm,_ = nnls(A,b); print('moment-matching test priors (rank feats):', pi_mm.round(3), 'sum', pi_mm.sum().round(3))
# bootstrap
bs=[]
for _ in range(300):
    ii=rng.randint(0,len(R_te),len(R_te)); p,_=nnls(A, np.r_[R_te[ii].mean(0),100]); bs.append(p)
print('bootstrap 95% CI:\n', np.percentile(bs,[2.5,97.5],axis=0).round(3))
# sanity: does moment matching recover a KNOWN prior? draw 2500 train rows with balanced priors (from complete cases)
for tgt in [np.full(5,.2), pi_tr]:
    res=[]
    for r in range(30):
        pick = np.concatenate([rng.choice(np.where(y==k)[0], int(round(tgt[k]*2500)), replace=False) for k in range(5)])
        p,_ = nnls(A, np.r_[R_tr[pick].mean(0) if not np.isnan(R_tr[pick]).any() else np.nanmean(R_tr[pick],0),100]); res.append(p)
    print('true', tgt.round(3), '-> recovered mean', np.mean(res,0).round(3), 'sd', np.std(res,0).round(3))
