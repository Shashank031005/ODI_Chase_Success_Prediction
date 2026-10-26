# Dataset schema and preprocessing documentation

## Files (as inspected)
| File | Shape | Notes |
|---|---|---|
| deliveries.csv | 26,119 x 22 | match_id, season, start_date, venue, innings, ball, batting_team, bowling_team, striker, non_striker, bowler, runs_off_bat, extras, wides, noballs, byes, legbyes, penalty, wicket_type, player_dismissed, other_wicket_type, other_player_dismissed |
| matches.csv | 48 x 18 | season, team1, team2, date, match_number, venue, city, toss_winner, toss_decision, player_of_match, umpire1, umpire2, reserve_umpire, match_referee, winner, winner_runs, winner_wickets, match_type |
| points_table.csv | 10 x 11 | Ranking, Team, Matches, Won, Lost, Tie, No Results, Points, Net Run Rate, For, Against |

Missing values: only optional columns (extras components, dismissal columns, `reserve_umpire` x1, `winner_runs`/`winner_wickets` 24 each, mutually exclusive).
Join: `deliveries.match_id` = `matches.match_number` (1-48, all present). Teams join by name. `points_table` covers group-stage matches only (its `Won` equals group-stage wins in matches.csv) and is **not** used for features or labels.

## Delivery semantics (verified)
* runs on a delivery = `runs_off_bat + extras`; `extras` equals the sum of the five extras columns on every row.
* legal ball = no wide and no no-ball. Byes/leg-byes/penalty do not change this.
* `ball` is not a legal-ball counter (wides/no-balls consume a number, e.g. 9.7-9.9) and the 10th delivery of an over is stored as x.1 (float truncation; 6 occurrences). The single "duplicate row" flagged by a naive duplicate check (match 21) is this artefact, not a duplicate. File order is used instead.
* wickets on a delivery = [wicket_type present and not "retired hurt"] + [other_wicket_type present]. One row (match 38, innings 1) holds two dismissals.
* No target / target-overs / DLS column exists. Target = first-innings total + 1, known when the chase starts. Overs limit = 300 legal balls unless the first innings ended before 300 legal balls without being all out (only match 15, 258 balls = 43 overs).
* boundary = 4 or 6 runs off the bat (no boundary flag exists).

## Validation
* Defended wins (23): first-innings total - chase total = `winner_runs` in 23/23.
* Chasing wins: 10 - wickets lost = `winner_wickets` in 24/24 included matches.
* No ties, no-results or abandoned matches (every match has a winner; points_table Tie = No Results = 0).

## Exclusions
* Match 35 (Pakistan v New Zealand): recorded as a chasing win with `winner_runs` = 21 although the chase ended on 200 against a first-innings 401: a DLS-type result whose revised target is not in the data. Excluded from the binary task -> 47 matches (24 chase wins, 23 losses).
* States removed from the row-level dataset: last delivery of every innings (47; includes all target-reached and all-out states), and 48 non-terminal "impossible" states (runs required > 6 x balls remaining). Eligible rows: 11,473.

## Features (one row per delivery, state after the delivery)
Basic: runs, wkts, wkts_remaining, legal_balls, balls_remaining, target, runs_required, crr, rrr, rr_diff, pct_complete.
Momentum: runs_last{6,12,30}, rate_last{6,12,30} (runs per over; window shortened early in the chase), wkts_last{12,30}, boundaries_last{12,30}, dot_pct_last12, rate_change_12_vs_earlier, rate_last12_minus_rrr, rate_last30_minus_rrr.
Context: chaser_won_toss, venue, batting_team, bowling_team.
Never used as inputs: winner, winner_runs, winner_wickets, player_of_match, final totals, points_table.
