# Data Mining and Classification for Ball-by-Ball ODI Cricket Chase Success Prediction

Single task: after every delivery of a second innings, predict whether the chasing team eventually wins.
Data: only the three 2023 ICC Men's Cricket World Cup files in `data/` (`deliveries.csv`, `matches.csv`, `points_table.csv`).

## Layout
| Path | Content |
|---|---|
| `src/prep.py` | Phases 1-3: loading, validation, cleaning, chase-state + momentum features, eligibility policy |
| `src/assoc.py` | Phase 4: Apriori and FP-Growth (implemented from scratch), rule metrics, match-cluster bootstrap |
| `src/models.py` | Phases 5-6: grouped CV, models, ablation sets, bootstrap, calibration, checkpoints |
| `src/run_experiments.py` | Runs every experiment, writes `results/*.csv` (about 13 min on 1 CPU) |
| `src/make_figures.py` | Phase 7: all figures in `figures/`, worked example `results/worked_example.csv` |
| `src/spark_pipeline.py` | Optional PySpark vs pandas benchmark. **Never executed by the author (PySpark unavailable)** |
| `notebooks/odi_chase_analysis.ipynb` | Walk-through that calls the modules (see note below) |
| `results/` | All numeric outputs used in the report (including `run_log.txt`) |
| `figures/` | PNG figures |
| `report/report.md`, `report/report.docx` | Full project report |
| `DATA_SCHEMA.md` | Schema, cleaning decisions, feature dictionary |

## Reproduce
```bash
pip install -r requirements.txt
cd src
python prep.py                # quick check: validation, exclusion counts
python run_experiments.py     # ~13 min on one CPU core; seeds are fixed
python make_figures.py
```
Random seeds are fixed (CV split seed 42; bootstrap seeds in the code), so reruns on the same library versions should
reproduce the tables. Results may differ in the last digits with other scikit-learn versions.

## Honest notes
* The notebook was assembled without Jupyter/nbformat available; its code cells were executed as a plain Python script to
  check they run, but it has no stored outputs.
* mlxtend was unavailable, so Apriori and FP-Growth are own implementations (cross-checked: identical itemsets).
* All numbers are exploratory estimates from 47 matches of one tournament.
