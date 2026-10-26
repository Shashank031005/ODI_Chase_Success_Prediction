"""Phases 1-3: load, validate, clean and build the leakage-safe chase-state dataset.

Every design decision is documented inline. Nothing here uses data outside the three
2023 ICC Men's Cricket World Cup CSV files.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXTRA_COLS = ["wides", "noballs", "byes", "legbyes", "penalty"]
DEFAULT_BALLS = 300          # 50 overs; overridden per match if the data show a shortened innings
WINDOWS = (6, 12, 30)        # momentum windows, in legal balls


def load_raw(data_dir=DATA):
    d = pd.read_csv(Path(data_dir) / "deliveries.csv")
    m = pd.read_csv(Path(data_dir) / "matches.csv")
    p = pd.read_csv(Path(data_dir) / "points_table.csv")
    return d, m, p


# ----------------------------------------------------------------------------- cleaning
def clean_deliveries(d):
    """Add file-order sequence and per-delivery derived columns.

    * The `ball` column is NOT used for ordering or counting: it counts every delivery
      (wides/no-balls get their own number, e.g. 9.7, 9.8, 9.9) and suffers float truncation
      (the 10th delivery of an over, 9.10, is stored as 9.1). File order is chronological.
    * legal ball  = no wide and no no-ball (byes, leg-byes, penalty runs do not change this).
    * wickets on a delivery = [wicket_type present and not 'retired hurt'] + [other_wicket_type present
                    (e.g. 'timed out')]. 'retired hurt' is not a dismissal; a row can hold 2 dismissals.
    * boundary    = 4 or 6 runs off the bat (no explicit boundary flag exists; a run-4 is
                    indistinguishable from a boundary-4 and is treated as one).
    """
    d = d.copy()
    d["orig_order"] = np.arange(len(d))
    for c in EXTRA_COLS:
        d[c] = d[c].fillna(0).astype(int)
    d = d.sort_values(["match_id", "innings", "orig_order"], kind="stable").reset_index(drop=True)
    # innings must be contiguous blocks in the file, otherwise file order would be unreliable
    key = d[["match_id", "innings"]]
    blocks = (key != key.shift()).any(axis=1).cumsum()
    assert blocks.nunique() == d.groupby(["match_id", "innings"]).ngroups, "innings are interleaved"
    d["is_legal"] = ((d.wides == 0) & (d.noballs == 0)).astype(int)
    d["runs_total"] = d.runs_off_bat + d.extras
    # a single delivery can carry TWO dismissals (match 38: 'caught' + a 'timed out' batter on the same row)
    d["is_wicket"] = ((d.wicket_type.notna() & (d.wicket_type != "retired hurt")).astype(int)
                      + d.other_wicket_type.notna().astype(int))
    d["is_boundary"] = d.runs_off_bat.isin([4, 6]).astype(int)
    return d


def validate_deliveries(d):
    """Schema/consistency assertions + a small report dict."""
    rep = {}
    comp = d[EXTRA_COLS].sum(axis=1)
    rep["extras_mismatch_rows"] = int((comp != d.extras).sum())
    rep["innings_per_match_ok"] = bool((d.groupby("match_id").innings.nunique() == 2).all())
    rep["wickets_gt10_innings"] = int((d.groupby(["match_id", "innings"]).is_wicket.sum() > 10).sum())
    rep["rows_ball_not_increasing(float truncation)"] = int((d.groupby(["match_id", "innings"]).ball.diff() <= 0).sum())
    assert rep["extras_mismatch_rows"] == 0 and rep["innings_per_match_ok"] and rep["wickets_gt10_innings"] == 0
    return rep


# ------------------------------------------------------------------ innings & match tables
def innings_summary(d):
    return d.groupby(["match_id", "innings"]).agg(
        runs=("runs_total", "sum"), wkts=("is_wicket", "sum"), legal=("is_legal", "sum"),
        deliveries=("is_legal", "size"), batting_team=("batting_team", "first"),
        bowling_team=("bowling_team", "first")).reset_index()


def match_table(d, m):
    """One row per match: first-innings total, chase totals, overs limit, label, exclusion."""
    s = innings_summary(d)
    i1 = s[s.innings == 1].set_index("match_id")
    i2 = s[s.innings == 2].set_index("match_id")
    mt = m.set_index("match_number").copy()
    out = pd.DataFrame({
        "chaser": i2.batting_team, "defender": i1.batting_team,
        "i1_runs": i1.runs, "i1_wkts": i1.wkts, "i1_legal": i1.legal,
        "i2_runs": i2.runs, "i2_wkts": i2.wkts, "i2_legal": i2.legal,
        "winner": mt.winner, "winner_runs": mt.winner_runs, "winner_wickets": mt.winner_wickets,
        "match_type": mt.match_type, "venue": mt.venue,
        "toss_winner": mt.toss_winner, "toss_decision": mt.toss_decision})
    out["target"] = out.i1_runs + 1
    # Overs limit inferred from the data: a first innings that stopped before 300 legal balls
    # WITHOUT being all out (10 dismissals incl. 'timed out') must have been a shortened innings.
    out["balls_limit"] = np.where((out.i1_legal < DEFAULT_BALLS) & (out.i1_wkts < 10),
                                  out.i1_legal, DEFAULT_BALLS)
    out["chase_win"] = (out.chaser == out.winner).astype(int)
    # Outcome-consistency validation against matches.csv margins
    lost = out.chase_win == 0
    won = out.chase_win == 1
    out["margin_ok"] = True
    out.loc[lost, "margin_ok"] = ((out.i1_runs - out.i2_runs) == out.winner_runs)[lost]
    out.loc[won, "margin_ok"] = (((10 - out.i2_wkts) == out.winner_wickets) & (out.i2_runs >= out.target))[won]
    # exceptional outcomes: tie / no-result / abandoned would show a winner that is not a playing team
    teams_ok = np.array([out.loc[i, "winner"] in (mt.loc[i, "team1"], mt.loc[i, "team2"]) for i in out.index])
    out["exceptional"] = ~teams_ok
    # DLS-type evidence: chasing team 'won' without reaching first-innings total + 1
    out["dls_suspect"] = (out.chase_win == 1) & (out.i2_runs < out.target)
    out["exclude_reason"] = np.select(
        [out.exceptional, out.dls_suspect],
        ["tie/no-result/abandoned", "chase won below first-innings total (DLS-type result; revised target not in data)"], "")
    out["included"] = out.exclude_reason == ""
    return out


# ------------------------------------------------------------------------- chase states
def _past_window(cum_arr, legal_arr, L, W):
    """Cumulative series evaluated at the point where W fewer legal balls had been bowled."""
    target_L = L - W
    idx = np.searchsorted(legal_arr, target_L, side="right") - 1
    return np.where((target_L <= 0) | (idx < 0), 0, cum_arr[np.clip(idx, 0, None)])


def build_chase_states(d, mt):
    """One row per delivery of every included chase; features use only past + current deliveries."""
    parts = []
    for mid, g in d[d.innings == 2].groupby("match_id", sort=True):
        info = mt.loc[mid]
        if not info.included:
            continue
        g = g.sort_values("orig_order")
        L = g.is_legal.cumsum().to_numpy()
        runs = g.runs_total.cumsum().to_numpy()
        wk = g.is_wicket.cumsum().to_numpy()
        bd = g.is_boundary.cumsum().to_numpy()
        dot = ((g.runs_total == 0) & (g.is_legal == 1)).astype(int).cumsum().to_numpy()
        n = len(g)
        f = pd.DataFrame({
            "match_id": mid, "delivery_idx": np.arange(1, n + 1), "ball_label": g.ball.to_numpy(),
            "batting_team": info.chaser, "bowling_team": info.defender, "venue": info.venue,
            "chaser_won_toss": int(info.toss_winner == info.chaser),
            "target": info.target, "balls_limit": info.balls_limit,
            "runs": runs, "wkts": wk, "legal_balls": L, "match_type": info.match_type,
            "is_legal": g.is_legal.to_numpy()})
        f["wkts_remaining"] = 10 - f.wkts
        f["balls_remaining"] = f.balls_limit - f.legal_balls
        f["runs_required"] = f.target - f.runs
        f["crr"] = np.where(L > 0, runs / np.maximum(L, 1) * 6, 0.0)
        f["rrr"] = np.where(f.balls_remaining > 0, f.runs_required / f.balls_remaining.clip(lower=1) * 6, np.nan)
        f["rr_diff"] = f.rrr - f.crr
        f["pct_complete"] = f.legal_balls / f.balls_limit
        # ---- momentum (past + current only) ----
        for W in WINDOWS:
            r_base = _past_window(runs, L, L, W)
            eff = np.minimum(W, L)           # window shorter than W early in the chase ("where available")
            f[f"runs_last{W}"] = runs - r_base
            f[f"rate_last{W}"] = np.where(eff > 0, (runs - r_base) / np.maximum(eff, 1) * 6, 0.0)
            if W in (12, 30):
                f[f"wkts_last{W}"] = wk - _past_window(wk, L, L, W)
                f[f"boundaries_last{W}"] = bd - _past_window(bd, L, L, W)
        eff12 = np.minimum(12, L)
        f["dot_pct_last12"] = np.where(eff12 > 0, (dot - _past_window(dot, L, L, 12)) / np.maximum(eff12, 1), 0.0)
        r12_base = _past_window(runs, L, L, 12)
        earlier_balls = L - eff12
        earlier_rate = np.where(earlier_balls > 0, r12_base / np.maximum(earlier_balls, 1) * 6, f.crr)
        f["rate_change_12_vs_earlier"] = f.rate_last12 - earlier_rate          # acceleration / slowdown
        f["rate_last12_minus_rrr"] = f.rate_last12 - f.rrr
        f["rate_last30_minus_rrr"] = f.rate_last30 - f.rrr
        f["chase_win"] = int(info.chase_win)          # label: used ONLY as y, never as a feature
        # ---- eligibility policy ----
        f["is_last_delivery"] = f.delivery_idx == n
        f["state_won"] = f.runs_required <= 0
        f["state_all_out"] = f.wkts >= 10
        f["state_overs_done"] = f.balls_remaining <= 0
        f["state_impossible"] = (f.runs_required > 6 * f.balls_remaining) & ~f.state_won
        parts.append(f)
    st = pd.concat(parts, ignore_index=True)
    st["terminal"] = st.is_last_delivery | st.state_won | st.state_all_out | st.state_overs_done
    st["eligible"] = ~st.terminal & ~st.state_impossible
    return st


FEATURES_BASIC = ["runs", "wkts", "wkts_remaining", "legal_balls", "balls_remaining", "target",
                  "runs_required", "crr", "rrr", "rr_diff", "pct_complete"]
FEATURES_MOMENTUM = ["runs_last6", "runs_last12", "runs_last30", "rate_last6", "rate_last12", "rate_last30",
                     "wkts_last12", "wkts_last30", "boundaries_last12", "boundaries_last30",
                     "dot_pct_last12", "rate_change_12_vs_earlier", "rate_last12_minus_rrr",
                     "rate_last30_minus_rrr"]
FEATURES_CONTEXT_NUM = ["chaser_won_toss"]
FEATURES_CONTEXT_CAT = ["venue", "batting_team", "bowling_team"]


def exclusion_counts(st):
    return {
        "all chase deliveries in included matches": int(len(st)),
        "terminal (last delivery / target reached / all out / overs done)": int(st.terminal.sum()),
        "   last delivery of innings": int(st.is_last_delivery.sum()),
        "   target reached": int(st.state_won.sum()),
        "   all out": int(st.state_all_out.sum()),
        "   overs exhausted": int(st.state_overs_done.sum()),
        "impossible (required > 6 x balls left), non-terminal": int((st.state_impossible & ~st.terminal).sum()),
        "eligible": int(st.eligible.sum())}


def build_all(data_dir=DATA):
    d, m, p = load_raw(data_dir)
    dc = clean_deliveries(d)
    rep = validate_deliveries(dc)
    mt = match_table(dc, m)
    st = build_chase_states(dc, mt)
    return dc, m, p, mt, st, rep


if __name__ == "__main__":
    import time
    t = time.time()
    dc, m, p, mt, st, rep = build_all()
    print("validation:", rep)
    print(mt[["chaser", "target", "balls_limit", "chase_win", "margin_ok", "included", "exclude_reason"]]
          .query("(not included) or (not margin_ok) or balls_limit != 300"))
    print("margin check failures among included:", int((~mt[mt.included].margin_ok).sum()))
    for k, v in exclusion_counts(st).items():
        print(f"{k}: {v}")
    el = st[st.eligible]
    print("eligible matches:", el.match_id.nunique(), " matches with >=1 eligible row by outcome:",
          el.groupby("chase_win").match_id.nunique().to_dict())
    print("row-level class share:", el.chase_win.value_counts(normalize=True).round(3).to_dict())
    print("NaN in features:", el[FEATURES_BASIC + FEATURES_MOMENTUM].isna().sum().sum())
    print("elapsed", round(time.time() - t, 2))
