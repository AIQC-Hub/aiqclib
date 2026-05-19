"""
Reduce the raw NRT-CORA-BO test input to a chosen subset of platforms/profiles.

Selection rationale (one row per pattern):
    SMHIHANOBUKTEN #328 -- temp_qc=4 and psal_qc=4 (both bad)
    SMHIHANOBUKTEN #333 -- temp_qc=4 and psal_qc=4 and pres_qc=4 (all three)
    SMHIBY4         #353 -- temp_qc=4 and psal_qc=4 (both bad)
    SMHIBCSIII10    #320 -- psal_qc=4 only (salinity-only failure)
    SMHIBCSIII10    #316 -- psal_qc=4 only (salinity-only failure, 2nd example)
    SMFBY29         #92  -- all QC=1 (happy-path baseline)

Total: 6 profiles, ~1,524 rows (down from 132,342), preserving coverage of
the four QC patterns the ML pipeline branches on.

Run once when you want to shrink the fixture. After running, regenerate all
downstream fixtures (select_rows, extract_features, train/test splits, model
joblibs) using the existing scripts plus a fresh pipeline run on the
reduced input.

Usage:
    uv run python scripts/reduce_test_input.py             # writes in place
    uv run python scripts/reduce_test_input.py --dry-run   # report only
"""

import argparse
from pathlib import Path

import polars as pl


# (platform_code, profile_no) tuples to keep.
KEEP_PROFILES: list[tuple[str, int]] = [
    ("SMHIHANOBUKTEN", 328),
    ("SMHIHANOBUKTEN", 333),
    ("SMHIBY4",        353),
    ("SMHIBCSIII10",   320),
    ("SMHIBCSIII10",   316),
    ("SMFBY29",         92),
    ("SMHIBY32",        338),
    ("SMHIBY32",        335),
    ("SMHIBY20",        345),
    ("SMHIBY20",        346),
    ("UMFF3A5",         207),
    ("SYKELL12",        148),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = REPO_ROOT / "tests" / "data" / "input" / "nrt_cora_bo_test.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be written without modifying the file.",
    )
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        raise SystemExit(
            f"{INPUT_FILE} not found. Run scripts/fetch_test_data.sh first."
        )

    df = pl.read_parquet(INPUT_FILE)
    original_rows = df.height
    original_profiles = df.select(["platform_code", "profile_no"]).unique().height

    # Build the keeper frame and inner-join to filter.
    keepers = pl.DataFrame(
        KEEP_PROFILES,
        schema=["platform_code", "profile_no"],
        orient="row",
    ).with_columns(pl.col("profile_no").cast(df["profile_no"].dtype))

    reduced = df.join(keepers, on=["platform_code", "profile_no"], how="inner")

    # Sanity: every requested profile actually existed in the input.
    found = reduced.select(["platform_code", "profile_no"]).unique()
    missing = keepers.join(found, on=["platform_code", "profile_no"], how="anti")
    if missing.height > 0:
        print("WARNING: requested profiles not found in input:")
        print(missing)

    print(f"Original:  {original_rows:>7,} rows, {original_profiles} profiles")
    print(f"Reduced:   {reduced.height:>7,} rows, {found.height} profiles")
    print(f"Shrunk by: {original_rows / reduced.height:.1f}x")
    print()
    print("Per-profile row counts:")
    print(
        reduced.group_by(["platform_code", "profile_no"])
               .agg(pl.len().alias("n_rows"))
               .sort(["platform_code", "profile_no"])
    )

    if args.dry_run:
        print("\n--dry-run set, not writing.")
        return

    reduced.write_parquet(INPUT_FILE)
    print(f"\nWrote {INPUT_FILE.relative_to(REPO_ROOT)}")
    print("\nNext: regenerate downstream fixtures by re-running the pipeline,")
    print("then scripts/regenerate_test_models.py for the joblibs.")


if __name__ == "__main__":
    main()