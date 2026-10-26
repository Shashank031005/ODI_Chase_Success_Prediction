"""OPTIONAL Phase-8 distributed-processing experiment (PySpark) -- NOT EXECUTED in the authoring environment.

PySpark was not installed in the sandbox in which this project was built and there was no network access,
so this script has NOT been run and NO Spark timing is claimed anywhere in the report. It is provided so the
experiment can be run elsewhere (`pip install pyspark`; needs Java). It reproduces the chase-state
construction (cumulative score, legal balls, rolling windows) with Spark SQL window functions and times it
against the pandas implementation in prep.py on the same workload.

Usage:  python src/spark_pipeline.py --replicate 1      # the real 47-match workload
        python src/spark_pipeline.py --replicate 200    # synthetic stress test: matches duplicated with new ids
Replicated data is ONLY for timing the engine; it must never be used for modelling.
"""
import argparse, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def pandas_workload(d_raw, m_raw):
    import pandas as pd
    from prep import clean_deliveries, match_table, build_chase_states
    dc = clean_deliveries(d_raw)
    mt = match_table(dc, m_raw)
    return build_chase_states(dc, mt)


def replicate(d, m, k):
    import pandas as pd
    if k <= 1:
        return d, m
    ds, ms = [], []
    for i in range(k):
        di, mi = d.copy(), m.copy()
        di["match_id"] += 1000 * i
        mi["match_number"] += 1000 * i
        ds.append(di); ms.append(mi)
    return pd.concat(ds, ignore_index=True), pd.concat(ms, ignore_index=True)


def spark_workload(spark, d_pdf):
    from pyspark.sql import functions as F, Window
    sdf = spark.createDataFrame(d_pdf[["match_id", "innings", "orig_order", "runs_off_bat", "extras", "wides", "noballs",
                                       "wicket_type", "other_wicket_type"]].astype({"wicket_type": "object", "other_wicket_type": "object"}))
    c = sdf.filter(F.col("innings") == 2)
    c = (c.withColumn("is_legal", ((F.col("wides") == 0) & (F.col("noballs") == 0)).cast("int"))
          .withColumn("runs_total", F.col("runs_off_bat") + F.col("extras"))
          .withColumn("is_wicket", (F.when(F.col("wicket_type").isNotNull() & (F.col("wicket_type") != "retired hurt"), 1).otherwise(0)
                                    + F.when(F.col("other_wicket_type").isNotNull(), 1).otherwise(0)))
          .withColumn("is_boundary", F.col("runs_off_bat").isin(4, 6).cast("int")))
    w = Window.partitionBy("match_id").orderBy("orig_order").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    c = (c.withColumn("runs", F.sum("runs_total").over(w)).withColumn("wkts", F.sum("is_wicket").over(w))
          .withColumn("legal_balls", F.sum("is_legal").over(w)).withColumn("bnd", F.sum("is_boundary").over(w)))
    # cumulative value at each legal-ball count (last delivery with that count) -> lookup for "W legal balls ago"
    lk = c.groupBy("match_id", "legal_balls").agg(F.max("runs").alias("runs_at"), F.max("wkts").alias("wkts_at"), F.max("bnd").alias("bnd_at"))
    for W in (6, 12, 30):
        past = lk.select("match_id", (F.col("legal_balls") + W).alias("legal_balls"),
                         F.col("runs_at").alias(f"runs_b{W}"), F.col("wkts_at").alias(f"wkts_b{W}"), F.col("bnd_at").alias(f"bnd_b{W}"))
        c = c.join(past, ["match_id", "legal_balls"], "left")
        c = (c.withColumn(f"runs_last{W}", F.col("runs") - F.coalesce(F.col(f"runs_b{W}"), F.lit(0)))
              .withColumn(f"wkts_last{W}", F.col("wkts") - F.coalesce(F.col(f"wkts_b{W}"), F.lit(0)))
              .withColumn(f"bnd_last{W}", F.col("bnd") - F.coalesce(F.col(f"bnd_b{W}"), F.lit(0))))
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--replicate", type=int, default=1); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    import numpy as np
    from prep import load_raw, clean_deliveries
    d_raw, m_raw, _ = load_raw()
    d_raw, m_raw = replicate(d_raw, m_raw, a.replicate)
    tp = []
    for _ in range(a.reps):
        t = time.perf_counter(); out_pd = pandas_workload(d_raw, m_raw); tp.append(time.perf_counter() - t)
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[*]").appName("chase_states").config("spark.sql.shuffle.partitions", "8").getOrCreate()
    dc = clean_deliveries(d_raw)
    sp_times = []
    for _ in range(a.reps):
        t = time.perf_counter(); n = spark_workload(spark, dc).count(); sp_times.append(time.perf_counter() - t)
    print(f"replicate={a.replicate} rows(innings2, all matches)={n}")
    print(f"pandas  median {np.median(tp):.3f}s  runs={np.round(tp,3)}")
    print(f"spark   median {np.median(sp_times):.3f}s  runs={np.round(sp_times,3)}  (first run includes JIT/JVM warm-up)")
    print("Cores / memory / Spark+pandas versions should be recorded alongside these numbers.")
