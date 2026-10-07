import pandas as pd, numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
print('shape', tr.shape, te.shape)
print('cols equal (minus target):', list(tr.columns.drop('target_popularity')) == list(te.columns))
print('dtypes:', tr.dtypes.value_counts().to_dict())
print('target dist:\n', tr.target_popularity.value_counts().sort_index().to_frame('n').assign(pct=lambda d: (d.n/len(tr)*100).round(2)))
na = tr.isna().sum(); print('NA cols train:', (na>0).sum(), 'total NA', na.sum(), '| test NA:', te.isna().sum().sum())
print('rows with any NA:', tr.isna().any(axis=1).mean().round(4), ' NA per row dist:', tr.isna().sum(axis=1).value_counts().sort_index().to_dict())
# ids
print('id range train', tr.id.min(), tr.id.max(), 'test', te.id.min(), te.id.max(), 'overlap', len(set(tr.id)&set(te.id)), 'union size', len(set(tr.id)|set(te.id)))
allid = np.sort(np.r_[tr.id, te.id]); print('ids contiguous 0..N-1?', (allid==np.arange(len(allid))).all() if allid[0]==0 else (allid==np.arange(allid[0],allid[0]+len(allid))).all())
print('corr id vs timedelta train:', np.corrcoef(tr.id, tr.timedelta)[0,1].round(4), ' spearman', tr[['id','timedelta']].corr('spearman').iloc[0,1].round(4))
# duplicates
F = [c for c in te.columns if c!='id']
print('dup full rows (ex id):', tr.duplicated(F+['target_popularity']).sum(), ' dup feature vectors:', tr.duplicated(F).sum())
both = pd.concat([tr[F], te[F]]); print('train-test identical feature rows:', both.duplicated(keep=False).sum() - tr.duplicated(F, keep=False).sum() - te.duplicated(F,keep=False).sum())
d = tr[tr.duplicated(F, keep=False)].sort_values(F[:5])
if len(d): print(d.groupby(F, dropna=False).target_popularity.agg(lambda s: ''.join(sorted(s))).value_counts().head())
# constant / near-constant
nun = tr[F].nunique(); print('constant:', list(nun[nun<=1].index))
top = tr[F].apply(lambda s: s.value_counts(normalize=True, dropna=False).iloc[0]); print('near-constant (>98% one value):', top[top>0.98].round(4).to_dict())
# shift
rows=[]
for c in F:
    a, b = tr[c].dropna(), te[c]
    ks = ks_2samp(a, b)
    rows.append((c, ks.statistic, ks.pvalue, a.mean(), b.mean()))
sh = pd.DataFrame(rows, columns=['f','ks','p','mean_tr','mean_te']).sort_values('ks', ascending=False)
print('top KS shift:\n', sh.head(12).round(4).to_string())
print('features with p<0.001:', (sh.p<0.001).sum(), 'of', len(sh))
# timedelta / id by target
print(tr.groupby('target_popularity')[['timedelta','id']].mean().round(1))
print('test timedelta quantiles', te.timedelta.quantile([0,.25,.5,.75,1]).values, 'train', tr.timedelta.quantile([0,.25,.5,.75,1]).values)
