"""Loading, target encoding, holdout split and missing-value handling."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C

LDA = [f"LDA_0{i}" for i in range(5)]
ZERO_IF_EMPTY = ["n_unique_tokens", "average_token_length", "global_subjectivity",
                 "global_sentiment_polarity", "avg_positive_polarity", "avg_negative_polarity"]


def load():
    train = pd.read_csv(C.TRAIN_CSV)
    test = pd.read_csv(C.TEST_CSV)
    assert list(train.columns.drop(C.TARGET)) == list(test.columns), "train/test columns differ"
    assert train[C.ID_COL].is_unique and test[C.ID_COL].is_unique
    assert not set(train[C.ID_COL]) & set(test[C.ID_COL]), "id overlap between train and test"
    return train, test


def encode_target(train):
    y = train[C.TARGET].map({c: i for i, c in enumerate(C.CLASSES)})
    assert y.notna().all(), "unknown class label"
    return y.to_numpy(int)


def holdout_split(y):
    """Fixed stratified split -> (dev_idx, holdout_idx). Holdout is the confirmation set."""
    idx = np.arange(len(y))
    dev, hold = train_test_split(idx, test_size=C.HOLDOUT_FRAC, stratify=y, random_state=C.HOLDOUT_SEED)
    return np.sort(dev), np.sort(hold)


def raw_features(df):
    return [c for c in df.columns if c not in (C.ID_COL, C.TARGET)]


# ---------------------------------------------------------------- missing values
def reconstruct_exact(df):
    """Fill NaNs that are *exactly* determined by other columns of the same row.
    Uses no labels and no other rows, so it is identical for train and test (test has no NaN)."""
    d = df.copy()
    # 1. LDA topic probabilities sum to 1 (max deviation 2e-6 in complete rows)
    for c in LDA:
        m = d[c].isna() & d[[x for x in LDA if x != c]].notna().all(axis=1)
        d.loc[m, c] = (1 - d.loc[m, [x for x in LDA if x != c]].sum(axis=1)).clip(lower=0)
    # 2. avg of self-reference shares equals min when min == max (holds in 100% of complete rows)
    m = d["self_reference_avg_sharess"].isna() & (d.self_reference_min_shares == d.self_reference_max_shares)
    d.loc[m, "self_reference_avg_sharess"] = d.loc[m, "self_reference_min_shares"]
    # 3. empty articles: every text statistic is exactly 0
    empty = d.n_tokens_content == 0
    for c in ZERO_IF_EMPTY:
        d.loc[empty & d[c].isna(), c] = 0.0
    # n_tokens_content missing but the article is evidently empty
    m = d.n_tokens_content.isna() & (d.n_non_stop_unique_tokens == 0) & (d.n_non_stop_words == 0)
    d.loc[m, "n_tokens_content"] = 0
    return d


def impute(train_X, other_Xs, method):
    """method in {'native','median','median_ind','exact','exact_median'}.
    Statistics are fitted on train_X only and applied to the others."""
    if method == "native":
        return train_X, other_Xs
    if method.startswith("exact"):
        train_X = reconstruct_exact(train_X)
        other_Xs = [reconstruct_exact(o) for o in other_Xs]
        if method == "exact":
            return train_X, other_Xs
    na_cols = [c for c in train_X.columns if train_X[c].isna().any()]
    med = train_X[na_cols].median()
    def f(X):
        X = X.copy()
        if method == "median_ind":
            for c in na_cols:
                X[f"{c}_isna"] = X[c].isna().astype(np.int8)
        X[na_cols] = X[na_cols].fillna(med)
        return X
    return f(train_X), [f(o) for o in other_Xs]
