"""Batch 1: reference on new protocol, NA handling, feature bug fix, training class weights."""
import sys; sys.path.insert(0, '.')
from src import experiment as X, models as M
P = M.LGBM_ORIG
runs = [
 ("E001", "Reference: original champion (v1 feats, native NA, balanced cw) on the new dev protocol", "lgbm", M.lgbm(P, "balanced"), "v1", "native"),
 ("E002", "Fixing the lda_top NaN bug (v2) does not hurt and removes train/test inconsistency", "lgbm", M.lgbm(P, "balanced"), "v2", "native"),
 ("E003", "Exact NA reconstruction (LDA sum, selfref min==max, empty articles) recovers signal", "lgbm", M.lgbm(P, "balanced"), "v2", "exact"),
 ("E004", "Median imputation beats native NaN handling (test has no NaN)", "lgbm", M.lgbm(P, "balanced"), "v2", "median"),
 ("E005", "Median + missing indicators", "lgbm", M.lgbm(P, "balanced"), "v2", "median_ind"),
 ("E006", "Exact reconstruction then median for the rest", "lgbm", M.lgbm(P, "balanced"), "v2", "exact_median"),
 ("E007", "Training class weights: none (decision rule does the rebalancing)", "lgbm", M.lgbm(P, "none"), "v2", "exact"),
 ("E008", "Training class weights: moderate alpha=0.5", "lgbm", M.lgbm(P, 0.5), "v2", "exact"),
 ("E009", "Training class weights: alpha=0.75", "lgbm", M.lgbm(P, 0.75), "v2", "exact"),
 ("E010", "Training class weights: over-balanced alpha=1.25", "lgbm", M.lgbm(P, 1.25), "v2", "exact"),
]
only = sys.argv[1:]
for eid, hyp, name, fp, feat, na in runs:
    if only and eid not in only: continue
    X.run(eid, hyp, name, fp, feat=feat, na=na, params={**P, "cw": hyp.split(":")[-1].strip() if "weights" in hyp else "balanced"})
