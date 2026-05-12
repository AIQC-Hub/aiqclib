import polars as pl
from pathlib import Path

src = Path("../tests/data/input/nrt_cora_bo_test.parquet")
df = pl.read_parquet(src)

# Per-profile QC summary: for each (platform, profile), what does its QC look like?
qc_summary = (
    df.group_by(["platform_code", "profile_no"])
      .agg([
          pl.len().alias("n_rows"),
          # Distinct non-1 values in each QC column
          pl.col("temp_qc").filter(pl.col("temp_qc") != 1).unique().alias("temp_nonone"),
          pl.col("psal_qc").filter(pl.col("psal_qc") != 1).unique().alias("psal_nonone"),
          pl.col("pres_qc").filter(pl.col("pres_qc") != 1).unique().alias("pres_nonone"),
          # Count of distinct values across all three QC columns combined
          pl.concat_list(["temp_qc", "psal_qc", "pres_qc"]).list.unique()
            .list.len().alias("n_distinct_qc"),
      ])
)

pl.Config.set_tbl_rows(-1)
print(qc_summary.sort("n_distinct_qc", descending=True).head(30))