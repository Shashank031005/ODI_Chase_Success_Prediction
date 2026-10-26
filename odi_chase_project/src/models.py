"""Phases 5-6: match-grouped evaluation of baseline / Decision Tree / Naive Bayes / Random Forest /
Logistic Regression, and feature-set ablations.

All splitting is by match_id. Preprocessing (scaling, one-hot encoding) lives inside sklearn Pipelines
and is therefore fitted on training folds only. Hyper-parameter search for the Decision Tree is
nested (inner GroupKFold inside each outer training fold). No resampling is used; class imbalance is
handled with class_weight='balanced'.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from prep import (FEATURES_BASIC, FEATURES_MOMENTUM, FEATURES_CONTEXT_NUM, FEATURES_CONTEXT_CAT)

warnings.filterwarnings("ignore")

FEATURE_SETS = {
    "A_basic": (FEATURES_BASIC, [], []),
    "B_basic+momentum": (FEATURES_BASIC + FEATURES_MOMENTUM, [], []),
    "C_full(+context)": (FEATURES_BASIC + FEATURES_MOMENTUM, FEATURES_CONTEXT_NUM, FEATURES_CONTEXT_CAT),
}
# extra ablation (not in the main grid) to isolate context
EXTRA_SETS = {"D_basic+context": (FEATURES_BASIC, FEATURES_CONTEXT_NUM, FEATURES_CONTEXT_CAT)}


def make_preprocessor(num, bin_, cat, scale=True):
    tr = []
    if num or bin_:
        tr.append(("num", StandardScaler() if scale else "passthrough", num + bin_))
    if cat:
        tr.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(tr)


def make_model(name, num, bin_, cat, seed=0):
    if name == "Dummy(majority)":
        return Pipeline([("pre", make_preprocessor(num, bin_, cat, False)),
                         ("clf", DummyClassifier(strategy="prior"))])
    if name == "Dummy(always-win)":
        return Pipeline([("pre", make_preprocessor(num, bin_, cat, False)),
                         ("clf", DummyClassifier(strategy="constant", constant=1))])
    if name == "LogReg":
        return Pipeline([("pre", make_preprocessor(num, bin_, cat)),
                         ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))])
    if name == "NaiveBayes":
        return Pipeline([("pre", make_preprocessor(num, bin_, cat)), ("clf", GaussianNB())])
    if name == "DecisionTree":
        grid = {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [50, 200]}
        base = Pipeline([("pre", make_preprocessor(num, bin_, cat, False)),
                         ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=seed))])
        return ("search", base, grid)
    if name == "RandomForest":
        return Pipeline([("pre", make_preprocessor(num, bin_, cat, False)),
                         ("clf", RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=25,
                                                        max_features="sqrt", class_weight="balanced_subsample",
                                                        n_jobs=1, random_state=seed))])
    raise ValueError(name)


MODEL_NAMES = ["Dummy(majority)", "Dummy(always-win)", "LogReg", "NaiveBayes", "DecisionTree", "RandomForest"]


def fit_predict(name, fs, Xtr, ytr, gtr, Xte, seed=0):
    num, bin_, cat = fs
    mdl = make_model(name, num, bin_, cat, seed)
    if isinstance(mdl, tuple):                       # nested, match-grouped hyper-parameter search
        _, base, grid = mdl
        inner = GroupKFold(n_splits=3)
        gs = GridSearchCV(base, grid, cv=inner, scoring="neg_brier_score", n_jobs=1)
        gs.fit(Xtr, ytr, groups=gtr)
        mdl = gs.best_estimator_
        best = gs.best_params_
    else:
        mdl.fit(Xtr, ytr)
        best = None
    return mdl, mdl.predict_proba(Xte)[:, 1], best


def repeated_group_splits(groups, n_splits=5, n_repeats=5, seed=42):
    """Repeated GroupKFold: match ids are shuffled each repeat, whole matches stay together."""
    ids = np.unique(groups)
    rng = np.random.default_rng(seed)
    out = []
    for r in range(n_repeats):
        perm = rng.permutation(ids)
        fold_of = {m: i % n_splits for i, m in enumerate(perm)}
        f = np.array([fold_of[g] for g in groups])
        out.append([(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_splits)])
    return out


def point_metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, yh), "precision": precision_score(y, yh, zero_division=0),
            "recall": recall_score(y, yh, zero_division=0), "f1": f1_score(y, yh, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 and np.ptp(p) > 0 else 0.5,
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}


def run_cv(df, model_names, feature_sets, n_splits=5, n_repeats=5, seed=42, verbose=True):
    """Returns (metrics per repeat, OOF probabilities, partition summary, chosen DT params)."""
    y = df.chase_win.to_numpy()
    g = df.match_id.to_numpy()
    splits = repeated_group_splits(g, n_splits, n_repeats, seed)
    metric_rows, oof, part_rows, tree_params = [], {}, [], []
    for r, folds in enumerate(splits):
        for k, (tr, te) in enumerate(folds):
            part_rows.append({"repeat": r, "fold": k, "train_matches": len(np.unique(g[tr])),
                              "test_matches": len(np.unique(g[te])),
                              "train_chase_wins": int(df.iloc[tr].groupby("match_id").chase_win.first().sum()),
                              "test_chase_wins": int(df.iloc[te].groupby("match_id").chase_win.first().sum()),
                              "train_rows": len(tr), "test_rows": len(te)})
    for fsname, fs in feature_sets.items():
        cols = fs[0] + fs[1] + fs[2]
        X = df[cols]
        for name in model_names:
            for r, folds in enumerate(splits):
                p_oof = np.zeros(len(df))
                for k, (tr, te) in enumerate(folds):
                    mdl, p, best = fit_predict(name, fs, X.iloc[tr], y[tr], g[tr], X.iloc[te], seed=r)
                    p_oof[te] = p
                    if best:
                        tree_params.append({"feature_set": fsname, "repeat": r, "fold": k, **best})
                oof[(fsname, name, r)] = p_oof
                m = point_metrics(y, p_oof)
                if name.startswith("Dummy"):          # constant scores carry no ranking information
                    m["roc_auc"], m["pr_auc"] = 0.5, float(y.mean())
                m.update({"feature_set": fsname, "model": name, "repeat": r})
                metric_rows.append(m)
            if verbose:
                sub = [x for x in metric_rows if x["feature_set"] == fsname and x["model"] == name]
                print(f"{fsname:20s} {name:16s} acc={np.mean([x['accuracy'] for x in sub]):.3f} "
                      f"f1={np.mean([x['f1'] for x in sub]):.3f} auc={np.mean([x['roc_auc'] for x in sub]):.3f} "
                      f"brier={np.mean([x['brier'] for x in sub]):.3f}", flush=True)
    return pd.DataFrame(metric_rows), oof, pd.DataFrame(part_rows), pd.DataFrame(tree_params)


def match_cluster_bootstrap(df, oof_p, n_boot=500, seed=7, metrics=("roc_auc", "brier", "f1", "accuracy")):
    """95% CI by resampling whole matches (the unit of independence)."""
    rng = np.random.default_rng(seed)
    g = df.match_id.to_numpy()
    y = df.chase_win.to_numpy()
    uniq = np.unique(g)
    pos = {m: np.where(g == m)[0] for m in uniq}
    vals = {k: [] for k in metrics}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([pos[m] for m in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        pm = point_metrics(y[idx], oof_p[idx])
        for k in metrics:
            vals[k].append(pm[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in vals.items()}


def checkpoint_table(df, oof_p, checkpoints=(60, 120, 180, 240, 270)):
    """At fixed innings checkpoints (legal balls bowled) take, for every match whose chase is still live,
    the first eligible state at/after the checkpoint. One row per match -> match-level evaluation."""
    d = df.assign(p=oof_p)
    rows = []
    for cp in checkpoints:
        sub = d[d.legal_balls >= cp].sort_values(["match_id", "delivery_idx"]).groupby("match_id").head(1)
        sub = sub[sub.legal_balls <= cp + 5]               # state must actually be close to the checkpoint
        if len(sub) < 6 or sub.chase_win.nunique() < 2:
            continue
        m = point_metrics(sub.chase_win.to_numpy(), sub.p.to_numpy())
        m.update({"checkpoint_balls": cp, "over": cp // 6, "n_matches": len(sub), "n_wins": int(sub.chase_win.sum())})
        rows.append(m)
    return pd.DataFrame(rows)


def match_weighted_metrics(df, p, thr=0.5):
    """Every match gets total weight 1 (row weight = 1 / rows in that match): match-level view."""
    y = df.chase_win.to_numpy()
    w = 1.0 / df.groupby("match_id").match_id.transform("size").to_numpy()
    yh = (p >= thr).astype(int)
    return {"w_accuracy": accuracy_score(y, yh, sample_weight=w),
            "w_brier": brier_score_loss(y, p, sample_weight=w),
            "w_roc_auc": roc_auc_score(y, p, sample_weight=w) if np.ptp(p) > 0 else 0.5}


def paired_bootstrap_diff(df, p_new, p_old, n_boot=500, seed=11):
    """Match-cluster bootstrap of metric(new) - metric(old) (AUC up is good, Brier down is good)."""
    rng = np.random.default_rng(seed)
    g = df.match_id.to_numpy(); y = df.chase_win.to_numpy()
    uniq = np.unique(g); pos = {m: np.where(g == m)[0] for m in uniq}
    da, db = [], []
    for _ in range(n_boot):
        idx = np.concatenate([pos[m] for m in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        da.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_old[idx]))
        db.append(brier_score_loss(y[idx], p_new[idx]) - brier_score_loss(y[idx], p_old[idx]))
    f = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"d_auc": f(da), "d_brier": f(db)}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def run_calibrated_rf(df, fs, n_splits=5, n_repeats=5, seed=42):
    """Random Forest + Platt scaling fitted on INNER grouped out-of-fold scores of the training fold."""
    y = df.chase_win.to_numpy(); g = df.match_id.to_numpy()
    cols = fs[0] + fs[1] + fs[2]; X = df[cols]
    out = {}
    for r, folds in enumerate(repeated_group_splits(g, n_splits, n_repeats, seed)):
        p_oof = np.zeros(len(df))
        for tr, te in folds:
            inner_p = np.zeros(len(tr))
            for a, b in GroupKFold(n_splits=4).split(tr, groups=g[tr]):
                m = make_model("RandomForest", *fs, seed=r); m.fit(X.iloc[tr[a]], y[tr[a]])
                inner_p[b] = m.predict_proba(X.iloc[tr[b]])[:, 1]
            platt = LogisticRegression(C=1e6).fit(_logit(inner_p), y[tr])
            m = make_model("RandomForest", *fs, seed=r); m.fit(X.iloc[tr], y[tr])
            p_oof[te] = platt.predict_proba(_logit(m.predict_proba(X.iloc[te])[:, 1]))[:, 1]
        out[r] = p_oof
    return out
