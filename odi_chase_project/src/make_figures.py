"""Phase 7: figures + worked example from the saved results (no model re-training)."""
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from prep import ROOT, FEATURES_BASIC
R, F = ROOT / "results", ROOT / "figures"
F.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
df = pd.read_csv(R / "chase_states_eligible.csv")
mt = pd.read_csv(R / "match_table.csv", index_col=0)
y = df.chase_win.to_numpy()
oof = pickle.load(open(R / "oof_predictions.pkl", "rb"))["oof"]
cal = pickle.load(open(R / "calibrated_rf_oof.pkl", "rb"))["cal"]
summ = pd.read_csv(R / "cv_metrics_summary.csv")
SETS = ["A_basic", "B_basic+momentum", "C_full(+context)"]
MODELS = ["Dummy(majority)", "Dummy(always-win)", "LogReg", "NaiveBayes", "DecisionTree", "RandomForest"]
COL = {"Dummy(majority)": "#bbbbbb", "Dummy(always-win)": "#888888", "LogReg": "#4c78a8", "NaiveBayes": "#f58518",
       "DecisionTree": "#54a24b", "RandomForest": "#b279a2"}

# 1 outcome distribution --------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
inc = mt[mt.included]
ax[0].bar(["Chase won", "Chase lost"], [inc.chase_win.sum(), (1 - inc.chase_win).sum()], color=["#4c78a8", "#e45756"])
ax[0].set_title(f"Matches ({len(inc)} included of {len(mt)})"); ax[0].set_ylabel("matches")
for i, v in enumerate([inc.chase_win.sum(), (1 - inc.chase_win).sum()]): ax[0].text(i, v + .3, int(v), ha="center")
rc = df.chase_win.value_counts().sort_index(ascending=False)
ax[1].bar(["Chase won", "Chase lost"], [rc[1], rc[0]], color=["#4c78a8", "#e45756"])
ax[1].set_title(f"Eligible delivery states ({len(df):,})"); ax[1].set_ylabel("rows")
for i, v in enumerate([rc[1], rc[0]]): ax[1].text(i, v + 80, f"{v:,}", ha="center")
plt.tight_layout(); plt.savefig(F / "fig01_outcome_distribution.png"); plt.close()

# 2 feature distributions ------------------------------------------------------------
feats = [("rrr", "required run rate"), ("wkts_remaining", "wickets remaining"), ("rate_last12", "runs/over, last 12 balls"),
         ("rr_diff", "RRR - CRR"), ("pct_complete", "share of innings completed"), ("boundaries_last12", "boundaries, last 12 balls")]
fig, axs = plt.subplots(2, 3, figsize=(10, 5.2))
for ax, (c, lab) in zip(axs.ravel(), feats):
    lo, hi = df[c].quantile([.005, .995])
    bins = np.linspace(lo, hi, 25) if df[c].nunique() > 12 else np.arange(df[c].min() - .5, df[c].max() + 1.5)
    for k, (cl, nm) in enumerate([(1, "chase won"), (0, "chase lost")]):
        ax.hist(df.loc[df.chase_win == cl, c].clip(lo, hi), bins=bins, alpha=.55, density=True,
                color=["#4c78a8", "#e45756"][k], label=nm)
    ax.set_title(lab); ax.set_yticks([])
axs[0, 0].legend(frameon=False)
plt.tight_layout(); plt.savefig(F / "fig02_feature_distributions.png"); plt.close()

# 3 rules ------------------------------------------------------------------------------
rules = pd.read_csv(R / "association_rules.csv")
if len(rules):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for cons, c in [("OUTCOME=Win", "#4c78a8"), ("OUTCOME=Loss", "#e45756")]:
        r = rules[rules.consequent == cons]
        ax[0].scatter(r.support, r.confidence, s=(r.lift - 1) * 120 + 8, alpha=.55, c=c, label=cons.split("=")[1] + " rules")
    ax[0].set_xlabel("support"); ax[0].set_ylabel("confidence"); ax[0].legend(frameon=False)
    ax[0].set_title("All rules passing the filters (marker size = lift)")
    top = rules.sort_values(["lift", "support"], ascending=False).head(12).iloc[::-1]
    lab = [f"{a.replace('STAGE=','').replace('MOMENTUM=','mom:').replace('PACE_VS_REQ=','pace:').replace('WKTS_LEFT=','wk:').replace('RRR=','rrr:')} -> {c.split('=')[1]}" for a, c in zip(top.antecedent, top.consequent)]
    ax[1].barh(range(len(top)), top.lift, color=["#4c78a8" if c.endswith("Win") else "#e45756" for c in top.consequent])
    ax[1].set_yticks(range(len(top))); ax[1].set_yticklabels([l if len(l) < 70 else l[:67] + "..." for l in lab], fontsize=6.5)
    for i, (l, s_) in enumerate(zip(top.lift, top.support)): ax[1].text(l + .005, i, f"sup {s_:.2f}", va="center", fontsize=6.5)
    ax[1].set_xlabel("lift"); ax[1].set_title("Top-12 rules by lift")
    plt.tight_layout(); plt.savefig(F / "fig03_association_rules.png"); plt.close()

comp = pd.read_csv(R / "arm_apriori_vs_fpgrowth.csv")
fig, ax = plt.subplots(figsize=(4.5, 3))
ax.plot(comp.min_support, comp.apriori_sec * 1e3, "o-", label="Apriori"); ax.plot(comp.min_support, comp.fpgrowth_sec * 1e3, "s-", label="FP-Growth")
ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel("minimum support"); ax.set_ylabel("median runtime (ms)"); ax.legend(frameon=False)
ax.set_title("Apriori vs FP-Growth (1,747 transactions)")
plt.tight_layout(); plt.savefig(F / "fig04_apriori_vs_fpgrowth.png"); plt.close()

# 5 model comparison ------------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, (m, lab) in zip(axs, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1"), ("brier", "Brier (lower better)")]):
    w = .13
    for j, mn in enumerate(MODELS):
        v = [summ[(summ.feature_set == s) & (summ.model == mn)][f"{m}_mean"].iloc[0] for s in SETS]
        e = [summ[(summ.feature_set == s) & (summ.model == mn)][f"{m}_std"].iloc[0] for s in SETS]
        ax.bar(np.arange(3) + (j - 2.5) * w, v, w, yerr=e, color=COL[mn], label=mn, capsize=1.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(["A basic", "B +mom.", "C full"]); ax.set_title(lab)
axs[0].set_ylim(.4, 1); axs[1].set_ylim(.4, 1); axs[2].set_ylim(.3, 1)
axs[0].legend(frameon=False, fontsize=6.5, loc="lower left")
plt.suptitle("Repeated grouped 5-fold CV (5 repeats); error bars = SD across repeats (partition variability only)", fontsize=8)
plt.tight_layout(); plt.savefig(F / "fig05_model_comparison.png"); plt.close()

# 6 confusion matrices ------------------------------------------------------------------
cm = pd.read_csv(R / "confusion_matrices.csv")
fig, axs = plt.subplots(1, 5, figsize=(13, 2.9))
for ax, mn in zip(axs, ["Dummy(majority)", "LogReg", "NaiveBayes", "DecisionTree", "RandomForest"]):
    r = cm[(cm.feature_set == "B_basic+momentum") & (cm.model == mn)].iloc[0]
    M = np.array([[r.TN, r.FP], [r.FN, r.TP]])
    ax.imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2): ax.text(j, i, f"{M[i,j]:,}", ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred loss", "pred win"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["true loss", "true win"]); ax.set_title(mn, fontsize=8)
plt.suptitle("Confusion matrices (set B, out-of-fold, repeat 0, threshold 0.5, delivery rows)", fontsize=8)
plt.tight_layout(); plt.savefig(F / "fig06_confusion_matrices.png"); plt.close()

# 7 ROC + PR ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for mn in ["LogReg", "NaiveBayes", "DecisionTree", "RandomForest"]:
    p = oof[("B_basic+momentum", mn, 0)]
    fpr, tpr, _ = roc_curve(y, p); pr_, rc_, _ = precision_recall_curve(y, p)
    ax[0].plot(fpr, tpr, color=COL[mn], label=f"{mn} ({roc_auc_score(y,p):.3f})")
    ax[1].plot(rc_, pr_, color=COL[mn], label=f"{mn} ({average_precision_score(y,p):.3f})")
ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("ROC (set B, OOF)")
ax[1].axhline(y.mean(), color="k", ls=":", lw=.8); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].legend(frameon=False, fontsize=7); ax[1].set_title("Precision-recall (set B, OOF)")
plt.tight_layout(); plt.savefig(F / "fig07_roc_pr_curves.png"); plt.close()

# 8 ablation -----------------------------------------------------------------------------
ab = pd.read_csv(R / "ablation_paired_bootstrap.csv")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
labs, k = [], 0
for ax, (m, ttl) in zip(axs, [("d_auc", "change in ROC-AUC (positive = better)"), ("d_brier", "change in Brier (negative = better)")]):
    k = 0; labs = []
    for mn in ["LogReg", "NaiveBayes", "DecisionTree", "RandomForest"]:
        for new, old, c in [("B_basic+momentum", "A_basic", "#4c78a8"), ("C_full(+context)", "B_basic+momentum", "#f58518"), ("D_basic+context", "A_basic", "#54a24b")]:
            r = ab[(ab.model == mn) & (ab.new == new) & (ab.old == old)].iloc[0]
            ax.errorbar(r[m], k, xerr=[[r[m] - r[m + "_lo"]], [r[m + "_hi"] - r[m]]], fmt="o", color=c, capsize=2)
            labs.append(f"{mn}: {new[0]}-{old[0]}"); k += 1
    ax.axvline(0, color="k", lw=.8); ax.set_yticks(range(k)); ax.set_yticklabels(labs, fontsize=6.5); ax.set_title(ttl, fontsize=8)
plt.suptitle("Ablation: paired match-cluster bootstrap 95% intervals (B-A: +momentum; C-B: +context; D-A: context only)", fontsize=8)
plt.tight_layout(); plt.savefig(F / "fig08_ablation.png"); plt.close()

# 9 checkpoints ---------------------------------------------------------------------------
ck = pd.read_csv(R / "checkpoint_metrics.csv")
fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
for ax, (m, lab) in zip(axs, [("roc_auc", "ROC-AUC (matches at checkpoint)"), ("brier", "Brier"), ("accuracy", "accuracy")]):
    for mn in ["LogReg", "NaiveBayes", "DecisionTree", "RandomForest"]:
        r = ck[(ck.feature_set == "B_basic+momentum") & (ck.model == mn)]
        ax.plot(r.over, r[m], "o-", color=COL[mn], label=mn)
    ax.set_xlabel("over of the chase"); ax.set_title(lab)
    if m == "accuracy": ax.axhline(.5, color="k", ls=":", lw=.8)
axs[0].legend(frameon=False, fontsize=7)
n_txt = ", ".join(f"{int(a)}:{int(b)}" for a, b in ck[(ck.feature_set == 'B_basic+momentum') & (ck.model == 'RandomForest')][["over", "n_matches"]].values)
plt.suptitle(f"Fixed-checkpoint, one-state-per-match evaluation (set B). n matches by over -> {n_txt}", fontsize=8)
plt.tight_layout(); plt.savefig(F / "fig09_checkpoint_performance.png"); plt.close()

# 10 calibration ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
series = [("RF raw", oof[("B_basic+momentum", "RandomForest", 0)], "#b279a2"), ("RF + Platt", cal[0], "#e45756"),
          ("LogReg", oof[("B_basic+momentum", "LogReg", 0)], "#4c78a8"), ("NaiveBayes", oof[("B_basic+momentum", "NaiveBayes", 0)], "#f58518")]
for lab, p, c in series:
    fp, mp = calibration_curve(y, p, n_bins=10, strategy="uniform")
    axs[0].plot(mp, fp, "o-", color=c, label=lab, ms=3)
    axs[1].hist(p, bins=20, alpha=.35, color=c, label=lab)
axs[0].plot([0, 1], [0, 1], "k:"); axs[0].set_xlabel("mean predicted probability"); axs[0].set_ylabel("observed chase-win fraction")
axs[0].legend(frameon=False, fontsize=7); axs[0].set_title("Reliability (set B, OOF repeat 0)")
axs[1].set_title("Distribution of predicted probabilities"); axs[1].set_xlabel("predicted P(chase win)")
plt.tight_layout(); plt.savefig(F / "fig10_calibration.png"); plt.close()

# 11 importance -----------------------------------------------------------------------------
pm = pd.read_csv(R / "feature_importance_permutation_setC.csv", index_col=0).head(12).iloc[::-1]
gi = pd.read_csv(R / "feature_importance_gini_setB.csv", index_col=0).head(12).iloc[::-1]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].barh(pm.index, pm.perm_auc_drop_mean, xerr=pm.perm_auc_drop_sd_over_folds, color="#b279a2"); axs[0].set_title("Permutation importance, RF set C\n(held-out ROC-AUC drop, mean +/- SD over 5 folds)", fontsize=8)
axs[1].barh(gi.index, gi.gini_importance, color="#54a24b"); axs[1].set_title("Impurity (Gini) importance, RF set B, fitted on all rows\n(descriptive; biased towards continuous features)", fontsize=8)
plt.tight_layout(); plt.savefig(F / "fig11_feature_importance.png"); plt.close()

# 12 trajectories ---------------------------------------------------------------------------
best = ("B_basic+momentum", "RandomForest", 0)
df["p"] = oof[best]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, cl, ttl in [(axs[0], 1, "chases that were won"), (axs[1], 0, "chases that were lost")]:
    for mid, g in df[df.chase_win == cl].groupby("match_id"):
        ax.plot(g.legal_balls / 6, g.p, lw=.7, alpha=.45, color="#4c78a8" if cl else "#e45756")
    ax.axhline(.5, color="k", ls=":", lw=.8); ax.set_xlabel("overs bowled in the chase"); ax.set_title(f"All {ttl} (out-of-fold predictions)")
axs[0].set_ylabel("predicted P(chase win)")
plt.tight_layout(); plt.savefig(F / "fig12a_probability_trajectories_all.png"); plt.close()

# worked example: a won chase whose probability dipped well below 0.4 and a lost chase that was above 0.6 at some point
cand = df.groupby("match_id").agg(win=("chase_win", "first"), pmin=("p", "min"), pmax=("p", "max"), n=("p", "size"))
comeback = cand[(cand.win == 1)].sort_values("pmin").index[0]
collapse = cand[(cand.win == 0)].sort_values("pmax", ascending=False).index[0]
close_win = cand[(cand.win == 1)].index.difference([comeback])
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for mid, c in [(comeback, "#4c78a8"), (collapse, "#e45756")]:
    g = df[df.match_id == mid]; i = mt.loc[mid]
    ax.plot(g.legal_balls / 6, g.p, color=c, label=f"match {mid}: {i.chaser} chasing {int(i.target)} v {i.defender} ({'won' if i.chase_win else 'lost'})")
ax.axhline(.5, color="k", ls=":", lw=.8); ax.set_xlabel("overs bowled"); ax.set_ylabel("predicted P(chase win)"); ax.legend(frameon=False, fontsize=7, loc="best")
ax.set_title("Out-of-fold probability trajectories: largest comeback / largest collapse in the data", fontsize=9)
plt.tight_layout(); plt.savefig(F / "fig12b_example_trajectories.png"); plt.close()

# worked example table (verified out-of-fold model output; model never saw this match when predicting it)
tab = []
for mid in [comeback, collapse]:
    g = df[df.match_id == mid].copy()
    for ov in [10, 20, 30, 40, 45]:
        r = g[g.legal_balls >= ov * 6].head(1)
        if len(r):
            r = r.iloc[0]
            tab.append({"match_id": mid, "chase": f"{mt.loc[mid,'chaser']} v {mt.loc[mid,'defender']}", "over": ov, "score": f"{int(r.runs)}/{int(r.wkts)}",
                        "wickets_remaining": int(r.wkts_remaining), "runs_required": int(r.runs_required), "balls_remaining": int(r.balls_remaining),
                        "required_rr": round(r.rrr, 2), "runs_last_12_balls": int(r.runs_last12), "rate_last12": round(r.rate_last12, 2),
                        "pred_P(chase win)": round(r.p, 3), "actual_result": "chase won" if r.chase_win else "chase lost"})
pd.DataFrame(tab).to_csv(R / "worked_example.csv", index=False)
print("figures done; examples:", comeback, collapse)
