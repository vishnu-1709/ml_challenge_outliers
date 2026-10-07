import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
F = [c for c in te.columns if c!='id']; cls=list('ABCDE')
y = tr.target_popularity.map({c:i for i,c in enumerate(cls)}).values
cc = ~tr[F].isna().any(axis=1).values
Xtr, ytr = tr.loc[cc,F], y[cc]; pi_tr = np.bincount(ytr)/len(ytr); U=np.full(5,.2)
X = pd.concat([Xtr, te[F]]); z = np.r_[np.zeros(len(Xtr)), np.ones(len(te))]
def auc(tgt, seeds=(0,1,2,3)):
    w=(tgt/pi_tr)[ytr]; w=w/w.sum()*len(te)  # train total weight == test total weight
    W=np.r_[w, np.ones(len(te))]; out=[]
    for s in seeds:
        p=np.zeros(len(z))
        for a,b in StratifiedKFold(5,shuffle=True,random_state=s).split(X,z):
            m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.05,num_leaves=15,min_child_samples=50,verbose=-1,random_state=s)
            m.fit(X.iloc[a],z[a],sample_weight=W[a]); p[b]=m.predict_proba(X.iloc[b])[:,1]
        out.append(roc_auc_score(z,p,sample_weight=W))
    return np.mean(out), np.std(out)
cands = {f't={t}':(1-t)*pi_tr+t*U for t in [0,0.25,0.5,0.75,1.0]}
cands['A,B x2 only'] = pi_tr*np.array([2,2,1,1,1]); cands['A,B x2 only']/=cands['A,B x2 only'].sum()
cands['400/500/500/550/550'] = np.array([400,500,500,550,550])/2500
for k,v in cands.items():
    a,s=auc(v); print(f'{k:22s} priors={np.round(v,3)}  AUC={a:.4f}±{s:.4f}')
