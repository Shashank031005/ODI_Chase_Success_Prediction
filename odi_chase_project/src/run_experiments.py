"""Runs every experiment end-to-end and writes tables to results/. Figures are made by make_figures.py."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).parent))
from prep import *
from assoc import *
from models import *

R = ROOT / "results"
R.mkdir(exist_ok=True)
t0 = time.time()

# ---------------------------------------------------------------- Phases 2-3: data
dc, m, p, mt, st, rep = build_all()
df = st[st.eligible].reset_index(drop=True)
mt.to_csv(R / "match_table.csv")
df.to_csv(R / "chase_states_eligible.csv", index=False)
pd.Series(exclusion_counts(st)).to_csv(R / "state_exclusion_counts.csv", header=["rows"])
json.dump({k: (v if not isinstance(v, (np.integer,)) else int(v)) for k, v in rep.items()}, open(R / "validation_report.json", "w"), indent=1)
timing = []
for _ in range(7):
    t = time.perf_counter(); build_all(); timing.append(time.perf_counter() - t)
pd.DataFrame({"run": range(7), "pandas_pipeline_sec": timing}).to_csv(R / "pandas_pipeline_timing.csv", index=False)

# ---------------------------------------------------------------- Phase 4: association rules
s_tr, attrs = build_transactions(st)
T = to_item_lists(s_tr, attrs)
comp = compare_algorithms(T)
comp.to_csv(R / "arm_apriori_vs_fpgrowth.csv", index=False)
MIN_SUP, MIN_CONF, MIN_LIFT, MIN_MATCHES = 0.05, 0.60, 1.15, 10
sup_a, cnt_a = apriori(T, MIN_SUP)
sup_f, cnt_f = fpgrowth(T, MIN_SUP)
assert cnt_a == cnt_f
rules = outcome_rules(cnt_f, T, s_tr, min_conf=MIN_CONF, min_lift=MIN_LIFT, min_matches=MIN_MATCHES)
rules = bootstrap_rules(rules, s_tr, n_boot=300)
rules.drop(columns=["_mask"]).to_csv(R / "association_rules.csv", index=False)
s_tr.to_csv(R / "arm_transactions.csv", index=False)
json.dump({"n_transactions": len(T), "n_matches": int(s_tr.match_id.nunique()),
           "min_support": MIN_SUP, "min_confidence": MIN_CONF, "min_lift": MIN_LIFT, "min_matches": MIN_MATCHES,
           "base_rate_win": float((s_tr.OUTCOME == "OUTCOME=Win").mean()), "n_rules": int(len(rules))},
          open(R / "arm_settings.json", "w"), indent=1)
print("ARM done", len(rules), "rules", round(time.time() - t0, 1), flush=True)

# ---------------------------------------------------------------- Phase 5-6: models + ablation
SETS = {**FEATURE_SETS, **EXTRA_SETS}
metrics, oof, part, tp = run_cv(df, MODEL_NAMES, SETS, n_splits=5, n_repeats=5)
metrics.to_csv(R / "cv_metrics_by_repeat.csv", index=False)
part.to_csv(R / "cv_partition_summary.csv", index=False)
tp.to_csv(R / "decision_tree_selected_params.csv", index=False)
pickle.dump({"oof": oof}, open(R / "oof_predictions.pkl", "wb"))
y = df.chase_win.to_numpy()

summ = metrics.groupby(["feature_set", "model"]).agg(["mean", "std"]).drop(columns="repeat")
summ.columns = ["_".join(c) for c in summ.columns]
summ.reset_index().to_csv(R / "cv_metrics_summary.csv", index=False)

# 95% match-cluster bootstrap CIs (repeat-0 out-of-fold predictions)
ci_rows = []
for (fsn, mn, r), pr in oof.items():
    if r != 0:
        continue
    ci = match_cluster_bootstrap(df, pr)
    row = {"feature_set": fsn, "model": mn}
    for k, (lo, hi) in ci.items():
        row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
    row.update(match_weighted_metrics(df, pr))
    ci_rows.append(row)
pd.DataFrame(ci_rows).to_csv(R / "cv_bootstrap_ci_and_match_weighted.csv", index=False)

# confusion matrices (repeat-0 OOF, threshold .5), summed over rows
cm_rows = []
for (fsn, mn, r), pr in oof.items():
    if r == 0:
        tn, fp, fn, tp_ = confusion_matrix(y, (pr >= .5).astype(int)).ravel()
        cm_rows.append({"feature_set": fsn, "model": mn, "TN": tn, "FP": fp, "FN": fn, "TP": tp_})
pd.DataFrame(cm_rows).to_csv(R / "confusion_matrices.csv", index=False)

# paired ablation tests (match-cluster bootstrap) on the real models
abl = []
for mn in ["LogReg", "NaiveBayes", "DecisionTree", "RandomForest"]:
    for new, old in [("B_basic+momentum", "A_basic"), ("C_full(+context)", "B_basic+momentum"),
                     ("C_full(+context)", "A_basic"), ("D_basic+context", "A_basic")]:
        res = paired_bootstrap_diff(df, oof[(new, mn, 0)], oof[(old, mn, 0)])
        abl.append({"model": mn, "new": new, "old": old,
                    "d_auc": res["d_auc"][0], "d_auc_lo": res["d_auc"][1], "d_auc_hi": res["d_auc"][2],
                    "d_brier": res["d_brier"][0], "d_brier_lo": res["d_brier"][1], "d_brier_hi": res["d_brier"][2]})
pd.DataFrame(abl).to_csv(R / "ablation_paired_bootstrap.csv", index=False)
print("CV done", round(time.time() - t0, 1), flush=True)

# checkpoints (repeat-0 OOF): match-level evaluation at fixed innings stages
ck = []
for (fsn, mn, r), pr in oof.items():
    if r == 0 and mn in ("LogReg", "NaiveBayes", "DecisionTree", "RandomForest"):
        t = checkpoint_table(df, pr)
        t.insert(0, "model", mn); t.insert(0, "feature_set", fsn)
        ck.append(t)
pd.concat(ck).to_csv(R / "checkpoint_metrics.csv", index=False)

# stage-wise row-level AUC (by chase progress) for RF
stage_rows = []
for fsn in SETS:
    pr = oof[(fsn, "RandomForest", 0)]
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        mk = ((df.pct_complete >= lo) & (df.pct_complete < hi)).to_numpy()
        if len(np.unique(y[mk])) > 1:
            stage_rows.append({"feature_set": fsn, "stage": f"{int(lo*100)}-{int(min(hi,1)*100)}%", "n_rows": int(mk.sum()),
                               "roc_auc": roc_auc_score(y[mk], pr[mk]), "brier": brier_score_loss(y[mk], pr[mk])})
pd.DataFrame(stage_rows).to_csv(R / "stagewise_rf_metrics.csv", index=False)

# ---------------------------------------------------------------- calibration
fsB = FEATURE_SETS["B_basic+momentum"]
cal = run_calibrated_rf(df, fsB)
cal_rows = []
for r in range(5):
    for lab, pr in [("RF raw", oof[("B_basic+momentum", "RandomForest", r)]), ("RF + Platt (inner-grouped)", cal[r]),
                    ("LogReg", oof[("B_basic+momentum", "LogReg", r)]), ("NaiveBayes", oof[("B_basic+momentum", "NaiveBayes", r)])]:
        mm_ = point_metrics(y, pr); mm_.update({"model": lab, "repeat": r}); cal_rows.append(mm_)
pd.DataFrame(cal_rows).groupby("model")[["brier", "roc_auc", "accuracy", "f1"]].agg(["mean", "std"]).to_csv(R / "calibration_summary.csv")
pickle.dump({"cal": cal}, open(R / "calibrated_rf_oof.pkl", "wb"))
print("calibration done", round(time.time() - t0, 1), flush=True)

# ---------------------------------------------------------------- feature importance
fsC = FEATURE_SETS["C_full(+context)"]
colsC = fsC[0] + fsC[1] + fsC[2]
splits0 = repeated_group_splits(df.match_id.to_numpy(), 5, 1, 42)[0]
imp_perm, imp_gini = [], []
for k, (tr, te) in enumerate(splits0):
    mdl = make_model("RandomForest", *fsC, seed=0)
    mdl.fit(df.iloc[tr][colsC], y[tr])
    pi = permutation_importance(mdl, df.iloc[te][colsC], y[te], scoring="roc_auc", n_repeats=3, random_state=k, n_jobs=1)
    imp_perm.append(pd.Series(pi.importances_mean, index=colsC, name=k))
perm = pd.concat(imp_perm, axis=1)
fsB_cols = fsB[0]
mdl = make_model("RandomForest", *fsB, seed=0).fit(df[fsB_cols], y)
gini = pd.Series(mdl.named_steps["clf"].feature_importances_, index=fsB_cols)
pd.DataFrame({"perm_auc_drop_mean": perm.mean(axis=1), "perm_auc_drop_sd_over_folds": perm.std(axis=1)}).sort_values(
    "perm_auc_drop_mean", ascending=False).to_csv(R / "feature_importance_permutation_setC.csv")
gini.sort_values(ascending=False).to_csv(R / "feature_importance_gini_setB.csv", header=["gini_importance"])

# interpretable tree (descriptive, fitted on all rows of set A: not an evaluated model)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, class_weight="balanced", random_state=0).fit(df[FEATURES_BASIC], y)
open(R / "decision_tree_depth3_rules.txt", "w").write(export_text(tree, feature_names=FEATURES_BASIC))

# ---------------------------------------------------------------- sensitivity: drop shortened match 15
keep = (df.match_id != 15).to_numpy()
df15 = df[keep].reset_index(drop=True)
mm15, _, _, _ = run_cv(df15, ["LogReg", "RandomForest"], {k: SETS[k] for k in ["A_basic", "B_basic+momentum"]}, n_repeats=3, verbose=False)
mm15.groupby(["feature_set", "model"])[["accuracy", "f1", "roc_auc", "brier"]].mean().to_csv(R / "sensitivity_without_match15.csv")

# ---------------------------------------------------------------- label-permutation sanity check (RF, set B)
rng = np.random.default_rng(0)
mt_in = mt[mt.included]
perm_scores = []
for rep_i in range(5):
    flip = dict(zip(mt_in.index, rng.permutation(mt_in.chase_win.to_numpy())))
    dperm = df.copy(); dperm["chase_win"] = dperm.match_id.map(flip).astype(int)
    mmx, _, _, _ = run_cv(dperm, ["LogReg"], {"B_basic+momentum": FEATURE_SETS["B_basic+momentum"]}, n_repeats=1, verbose=False)
    perm_scores.append(mmx[["accuracy", "roc_auc", "brier"]].iloc[0].to_dict())
pd.DataFrame(perm_scores).to_csv(R / "sanity_shuffled_match_labels.csv", index=False)

print("ALL DONE", round(time.time() - t0, 1), flush=True)
