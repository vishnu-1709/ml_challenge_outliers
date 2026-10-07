"""Feature sets. Every feature is row-wise (no target, no cross-row statistics), so there is
no fold leakage and train/test are processed identically.

  base : the 59 raw columns
  v1   : original notebook features, reproduced exactly (including its NaN argmax quirk)
  v2   : v1 with NaN-safe topic features (fixes the lda_top bug)
  v3   : v2 + hypothesis-driven batch (topic shape, keyword dispersion, sentiment balance)
"""
import numpy as np
import pandas as pd

CH = ["lifestyle", "entertainment", "bus", "socmed", "tech", "world"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
LDA = [f"LDA_0{i}" for i in range(5)]


def _common(d):
    chc = [f"data_channel_is_{c}" for c in CH]
    d["channel"] = np.where(d[chc].sum(axis=1) == 0, 6, np.argmax(d[chc].values, axis=1))
    d["weekday"] = np.argmax(d[[f"weekday_is_{x}" for x in DAYS]].values, axis=1)
    d["empty_article"] = (d.n_tokens_content == 0).astype(int)
    words = d.n_tokens_content.replace(0, np.nan)
    d["imgs_per_word"] = d.num_imgs / words
    d["hrefs_per_word"] = d.num_hrefs / words
    d["self_href_ratio"] = d.num_self_hrefs / (d.num_hrefs + 1)
    d["kw_avg_spread"] = d.kw_max_avg - d.kw_min_avg
    d["kw_avg_ratio"] = d.kw_avg_avg / (d.kw_max_avg + 1)
    d["selfref_spread"] = d.self_reference_max_shares - d.self_reference_min_shares
    d["sent_x_subj"] = d.global_sentiment_polarity * d.global_subjectivity
    return d


def v1(df):
    d = df.copy()
    d = _common(d)
    # original: np.argmax returns the NaN position when LDA_02 is missing (bug kept for reproduction)
    d["lda_top"] = np.argmax(d[LDA].values, axis=1)
    d["lda_max"] = d[LDA].max(axis=1)
    order = ["channel", "weekday", "lda_top", "lda_max", "empty_article", "imgs_per_word", "hrefs_per_word",
             "self_href_ratio", "kw_avg_spread", "kw_avg_ratio", "selfref_spread", "sent_x_subj"]
    return d[[c for c in df.columns] + order]


def _lda_block(d):
    L = d[LDA].values
    Lf = np.where(np.isnan(L), -np.inf, L)
    top = np.argmax(Lf, axis=1).astype(float)
    top[np.isnan(L).any(axis=1)] = np.nan  # unknown if a topic is missing (exact reconstruction removes most)
    d["lda_top"] = top
    d["lda_max"] = d[LDA].max(axis=1)
    return d, L


def v2(df):
    d = df.copy()
    d = _common(d)
    d, _ = _lda_block(d)
    return d


def v3(df):
    d = v2(df)
    L = d[LDA].values
    S = np.sort(np.where(np.isnan(L), 0, L), axis=1)
    # H: a single dominant topic => clearer audience; mixed topics => diffuse appeal
    d["lda_margin"] = S[:, -1] - S[:, -2]
    Lc = np.clip(L, 1e-9, 1)
    d["lda_entropy"] = -(Lc * np.log(Lc)).sum(axis=1)
    # H: keyword-share statistics are the strongest signal; dispersion/position adds information
    d["kw_avg_pos"] = (d.kw_avg_avg - d.kw_min_avg) / (d.kw_max_avg - d.kw_min_avg + 1)
    d["kw_max_ratio"] = d.kw_avg_max / (d.kw_max_max + 1)
    d["kw_min_ratio"] = d.kw_avg_min / (d.kw_max_min + 1)
    d["log_kw_avg_avg"] = np.log1p(d.kw_avg_avg.clip(lower=0))
    d["log_selfref_avg"] = np.log1p(d.self_reference_avg_sharess)
    # H: positive/negative balance and title-vs-body tone mismatch
    d["pos_neg_balance"] = d.avg_positive_polarity + d.avg_negative_polarity
    d["title_body_sent_gap"] = d.title_sentiment_polarity - d.global_sentiment_polarity
    d["log_content"] = np.log1p(d.n_tokens_content)
    d["title_content_ratio"] = d.n_tokens_title / (d.n_tokens_content + 1)
    d["kw_per_word"] = d.num_keywords / (d.n_tokens_content + 1)
    d["videos_per_word"] = d.num_videos / (d.n_tokens_content + 1)
    # H: channel x weekday publishing pattern
    d["channel_weekday"] = d.channel * 7 + d.weekday
    return d


FEATURE_SETS = {"base": lambda df: df.copy(), "v1": v1, "v2": v2, "v3": v3}


def build(train_X, test_X, name):
    f = FEATURE_SETS[name]
    A, B = f(train_X), f(test_X)
    assert list(A.columns) == list(B.columns)
    return A, B
