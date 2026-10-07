"""Model factories.

Each factory returns a Member:
  member.fit(X, y, seed) -> fitted object with .predict_proba(X)
  member(X_tr, y_tr, X_va, seed) -> proba (n_va, K)          (fit + predict, used by the CV harness)

Training class weights (cw) are separate from decision weights (src/optimization.py):
  cw = 'none' | 'balanced' | float alpha  -> sample weight  freq_k ** (-alpha)  (balanced == alpha 1)
"""
import numpy as np
import lightgbm as lgb

K = 5

# Original notebook champion (Optuna, 25 trials, 3-fold CV) — kept verbatim
LGBM_ORIG = dict(n_estimators=600, num_leaves=24, min_child_samples=18, colsample_bytree=0.512,
                 subsample=0.725, reg_lambda=0.192, reg_alpha=0.0025, learning_rate=0.03, subsample_freq=1)


def class_sample_weight(y, cw):
    if cw in (None, "none", 0, 0.0):
        return np.ones(len(y))
    alpha = 1.0 if cw == "balanced" else float(cw)
    freq = np.bincount(y, minlength=K) / len(y)
    w = freq[y] ** (-alpha)
    return w / w.mean()


class Bag:
    """Average of several fitted models."""
    def __init__(self, ms): self.ms = ms
    def predict_proba(self, X): return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


class Member:
    def __init__(self, fit_fn): self._fit = fit_fn
    def fit(self, X, y, seed): return self._fit(X, y, seed)
    def __call__(self, Xa, ya, Xb, seed): return self._fit(Xa, ya, seed).predict_proba(Xb)


def lgbm(params=None, cw="balanced", n_bags=1):
    p = dict(LGBM_ORIG if params is None else params)

    def fit(Xa, ya, seed):
        sw = class_sample_weight(ya, cw)
        return Bag([lgb.LGBMClassifier(**p, random_state=seed + 7919 * b, verbose=-1, n_jobs=2).fit(Xa, ya, sample_weight=sw)
                    for b in range(n_bags)])
    return Member(fit)


def xgb(params=None, cw="balanced"):
    import xgboost as xgbm
    p = dict(n_estimators=700, learning_rate=0.03, max_depth=5, min_child_weight=3, subsample=0.75,
             colsample_bytree=0.5, reg_lambda=1.0, tree_method="hist", max_bin=256)
    if params: p.update(params)

    def fit(Xa, ya, seed):
        m = xgbm.XGBClassifier(**p, objective="multi:softprob", random_state=seed, n_jobs=2, verbosity=0)
        return m.fit(Xa, ya, sample_weight=class_sample_weight(ya, cw))
    return Member(fit)


def catboost(params=None, cw="balanced"):
    from catboost import CatBoostClassifier
    p = dict(iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, loss_function="MultiClass")
    if params: p.update(params)

    def fit(Xa, ya, seed):
        m = CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=2, allow_writing_files=False)
        return m.fit(Xa, ya, sample_weight=class_sample_weight(ya, cw))
    return Member(fit)


def _sk(est_fn, cw, scale=False):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import QuantileTransformer

    def fit(Xa, ya, seed):
        steps = [SimpleImputer(strategy="median")]
        if scale:
            steps.append(QuantileTransformer(output_distribution="normal", random_state=0))
        est = est_fn(seed)
        m = make_pipeline(*steps, est)
        name = m.steps[-1][0]
        return m.fit(Xa.to_numpy(float), ya, **{f"{name}__sample_weight": class_sample_weight(ya, cw)})

    class _W:
        def __init__(self, m): self.m = m
        def predict_proba(self, X): return self.m.predict_proba(X.to_numpy(float))
    return Member(lambda Xa, ya, seed: _W(fit(Xa, ya, seed)))


def extratrees(cw="balanced", **kw):
    from sklearn.ensemble import ExtraTreesClassifier
    p = dict(n_estimators=500, min_samples_leaf=5, max_features=0.5, n_jobs=2); p.update(kw)
    return _sk(lambda s: ExtraTreesClassifier(**p, random_state=s), cw)


def randomforest(cw="balanced", **kw):
    from sklearn.ensemble import RandomForestClassifier
    p = dict(n_estimators=500, min_samples_leaf=5, max_features=0.3, n_jobs=2); p.update(kw)
    return _sk(lambda s: RandomForestClassifier(**p, random_state=s), cw)


def histgb(cw="balanced"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return _sk(lambda s: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=24,
                                                        min_samples_leaf=40, l2_regularization=1.0, random_state=s), cw)


def logreg(cw="balanced"):
    from sklearn.linear_model import LogisticRegression
    return _sk(lambda s: LogisticRegression(C=0.5, max_iter=3000), cw, scale=True)


def majority():
    class _M:
        def __init__(self, k): self.k = k
        def predict_proba(self, X):
            P = np.zeros((len(X), K)); P[:, self.k] = 1; return P
    return Member(lambda Xa, ya, seed: _M(np.bincount(ya, minlength=K).argmax()))
