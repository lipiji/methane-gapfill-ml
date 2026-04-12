"""
Extract FLUXNET-CH4 HH data for the 17 paper sites and save as raw.csv
for each site under data/{SiteID}/raw.csv.

Usage:
    python prepare_data.py
"""

import glob
import io
import os
import zipfile

import pandas as pd

# 17 sites from Irvin et al. 2021 (Table 2)
PAPER_SITES = [
    "US-Uaf", "US-Los", "SE-Deg", "FI-Sii", "US-Twt",
    "FI-Si2", "CA-SCB", "NZ-Kop", "FI-Lom", "JP-Mse",
    "JP-BBY", "BR-Npw", "US-Tw4", "US-WPT", "US-Myb",
    "US-Tw1", "US-OWC",
]

# Columns to DROP (gap-filled FCH4 targets and uncertainty — not valid predictors)
DROP_COLS = [
    "TIMESTAMP_START",
    "FCH4_F", "FCH4_F_ANNOPTLM", "FCH4_F_RANDUNC",
    "FCH4_F_ANNOPTLM_UNC", "FCH4_F_ANNOPTLM_QC",
]

CH4_DATA_DIR = r"E:\微云\data\环境\CH4\CH4"
OUTPUT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def extract_site(site: str) -> None:
    zips = glob.glob(os.path.join(CH4_DATA_DIR, f"FLX_{site}_FLUXNET-CH4_*.zip"))
    if not zips:
        print(f"  [SKIP] {site}: zip not found")
        return

    zip_path = zips[0]
    with zipfile.ZipFile(zip_path) as zf:
        hh_files = [f for f in zf.namelist() if "_HH_" in f and f.endswith(".csv")]
        if not hh_files:
            print(f"  [SKIP] {site}: no HH csv inside zip")
            return

        df = pd.read_csv(io.BytesIO(zf.read(hh_files[0])), na_values=-9999)

    # Validate required columns
    for col in ["TIMESTAMP_END", "FCH4"]:
        if col not in df.columns:
            print(f"  [ERROR] {site}: missing required column '{col}'")
            return

    # Drop unwanted columns (only if present)
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)

    # Ensure TIMESTAMP_END is in YYYYMMDDHHmm format (integer → string zero-padded)
    df["TIMESTAMP_END"] = df["TIMESTAMP_END"].astype(str).str.zfill(12)

    out_dir = os.path.join(OUTPUT_DATA_DIR, site)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "raw.csv")
    df.to_csv(out_path, index=False)

    n_total = len(df)
    n_fch4 = df["FCH4"].notna().sum()
    print(f"  [OK] {site}: {n_total} rows, FCH4 valid={n_fch4} ({n_fch4/n_total*100:.1f}%), "
          f"cols={len(df.columns)} → {out_path}")


def main() -> None:
    print(f"Output directory: {OUTPUT_DATA_DIR}")
    print(f"Processing {len(PAPER_SITES)} sites...\n")
    for site in PAPER_SITES:
        extract_site(site)
    print("\nDone.")


if __name__ == "__main__":
    main()
