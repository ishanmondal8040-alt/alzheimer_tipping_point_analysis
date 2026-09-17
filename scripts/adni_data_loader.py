"""
adni_data_loader.py
--------------------
Production data-loading and preprocessing module for the AD Tipping Point /
Critical Slowing Down (CSD) pipeline.

Responsibilities of this module (and ONLY this module):
    1. Load a longitudinal ADNI-style biomarker CSV
       (columns: RID, VISCODE, ABETA, TAU, PTAU, FDG, Hippocampus).
    2. Parse ADNI's VISCODE strings ('bl', 'm06', 'm12', ...) into a numeric
       MONTHS time axis, since CSD indicators require an ordered time series.
    3. Handle missing values with SUBJECT-LEVEL interpolation (linear by
       default, optional cubic spline), never pooling across subjects.
    4. Apply GLOBAL (across-subject) Z-score normalization per biomarker,
       so indicators computed on different biomarkers (different units/
       scales) are directly comparable.
    5. Reshape the cleaned data into a structure that plugs directly into
       early_warning_utils.compute_csd_indicators() -- one clean, sorted,
       NaN-free 1-D array per (subject, biomarker).

This module does NOT compute CSD indicators itself (variance / AC1 /
Kendall's tau) -- that stays in early_warning_utils.py. Keeping data loading
and indicator computation separate means a bug or change in one does not
silently corrupt the other.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
REQUIRED_COLS = ["RID", "VISCODE", "ABETA", "TAU", "PTAU", "FDG", "Hippocampus"]
BIOMARKER_COLS = ["ABETA", "TAU", "PTAU", "FDG", "Hippocampus"]


# ---------------------------------------------------------------------------
# 1. Raw CSV loading + schema validation
# ---------------------------------------------------------------------------
def load_adni_csv(csv_path, required_cols=None):
    """
    Load a raw ADNI-style longitudinal CSV and validate its schema.

    ADNI biomarker exports sometimes store values as strings with
    inequality prefixes (e.g. ABETA = ">1700" for values above the assay's
    detection ceiling). These are coerced to numeric by stripping the
    '<'/'>' characters; anything still non-numeric becomes NaN and is
    handled later by the interpolation step.

    Parameters
    ----------
    csv_path : str or Path
    required_cols : list[str], optional
        Defaults to REQUIRED_COLS.

    Returns
    -------
    pd.DataFrame with exactly `required_cols`, correctly typed.
    """
    required_cols = required_cols or REQUIRED_COLS
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"ADNI CSV not found at: {path}")

    df = pd.read_csv(path)

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[required_cols].copy()

    for col in BIOMARKER_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[<>]", "", regex=True),
                errors="coerce",
            )

    df["RID"] = df["RID"].astype(int)
    df["VISCODE"] = df["VISCODE"].astype(str).str.lower().str.strip()

    return df


# ---------------------------------------------------------------------------
# 2. VISCODE -> numeric time axis
# ---------------------------------------------------------------------------
def viscode_to_months(viscode_series):
    """
    Convert ADNI VISCODE strings to numeric months-since-baseline.

    Recognized patterns:
        'bl', 'sc', 'scmri'  -> 0   (baseline / screening)
        'm06', 'm12', 'm24'  -> 6, 12, 24, ...

    Anything unrecognized -> NaN (the caller drops these rows, since we
    cannot place them on the time axis needed for CSD analysis).
    """

    def _parse_one(v):
        v = str(v).lower().strip()
        if v in ("bl", "sc", "scmri"):
            return 0
        if v.startswith("m") and v[1:].isdigit():
            return int(v[1:])
        return np.nan

    return viscode_series.map(_parse_one)


def add_time_axis(df):
    """
    Add a numeric 'MONTHS' column derived from VISCODE, drop rows whose
    VISCODE could not be parsed, and sort by (RID, MONTHS).
    """
    df = df.copy()
    df["MONTHS"] = viscode_to_months(df["VISCODE"])

    n_before = len(df)
    df = df.dropna(subset=["MONTHS"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[adni_data_loader] Dropped {n_dropped} row(s) with unrecognized VISCODE.")

    df["MONTHS"] = df["MONTHS"].astype(int)
    df = df.sort_values(["RID", "MONTHS"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Subject-level missing-value interpolation
# ---------------------------------------------------------------------------
def interpolate_subject_trajectory(
    df_subject,
    biomarker_cols=None,
    method="linear",
    spline_order=3,
    min_points_for_spline=4,
):
    """
    Interpolate missing biomarker values within ONE subject's trajectory,
    ordered by MONTHS. Interpolation is always done WITHIN a subject --
    never across subjects, since patients are on different disease
    trajectories and pooling would create spurious "recovery" artifacts.

    Falls back to linear interpolation if a subject does not have enough
    valid (non-NaN) observations for the requested spline order (a cubic
    spline needs at least `order + 1` points; with typical ADNI visit
    counts of 4-8 per subject, several subjects WILL fall back -- this is
    expected and safe, not a bug.

    Edge NaNs (before the first or after the last valid observation) are
    filled via `limit_direction='both'`, which is equivalent to a
    nearest-value carry at the boundary. `interpolated_mask` flags exactly
    which cells were filled in, so this can be excluded later if you want
    CSD windows to only use directly-observed values.

    Returns
    -------
    (filled_df, interpolated_mask) : both indexed the same way as the
        input, `interpolated_mask` is boolean (True = value was imputed).
    """
    biomarker_cols = biomarker_cols or BIOMARKER_COLS

    df_subject = df_subject.sort_values("MONTHS").set_index("MONTHS")
    out = df_subject.copy()
    interpolated_mask = pd.DataFrame(False, index=df_subject.index, columns=biomarker_cols)

    for col in biomarker_cols:
        series = df_subject[col]
        was_na = series.isna()
        n_valid = series.notna().sum()

        if n_valid == 0:
            # Nothing to interpolate from -- leave as NaN. Downstream code
            # (build_subject_series_dict) will simply produce an empty
            # series for this (subject, biomarker) pair.
            continue
        elif method == "spline" and n_valid >= min_points_for_spline:
            out[col] = series.interpolate(
                method="spline", order=spline_order, limit_direction="both"
            )
        else:
            out[col] = series.interpolate(method="linear", limit_direction="both")

        interpolated_mask[col] = was_na & out[col].notna()

    out = out.reset_index()
    return out, interpolated_mask.reset_index(drop=True)


def interpolate_all_subjects(df, biomarker_cols=None, method="linear", spline_order=3):
    """
    Apply `interpolate_subject_trajectory` group-wise across every RID.
    """
    biomarker_cols = biomarker_cols or BIOMARKER_COLS

    filled_frames, mask_frames = [], []
    for rid, g in df.groupby("RID", sort=False):
        filled, mask = interpolate_subject_trajectory(
            g, biomarker_cols, method=method, spline_order=spline_order
        )
        filled["RID"] = rid
        mask["RID"] = rid
        filled_frames.append(filled)
        mask_frames.append(mask)

    filled_df = pd.concat(filled_frames, ignore_index=True)
    mask_df = pd.concat(mask_frames, ignore_index=True)
    return filled_df, mask_df


# ---------------------------------------------------------------------------
# 4. Global Z-score normalization per biomarker
# ---------------------------------------------------------------------------
def zscore_normalize(df, biomarker_cols=None):
    """
    Z-score normalize each biomarker GLOBALLY (across all subjects and
    visits combined):
        z = (x - mean_across_all_subjects) / std_across_all_subjects

    This is deliberately global, not per-subject: CSD indicators (rolling
    variance, AC1) are about a subject's fluctuations relative to the
    population's typical scale for that biomarker, and per-subject
    normalization would remove exactly the amplitude information CSD
    needs.

    Guards against near-constant columns (std ~ 0) to avoid divide-by-zero.

    Returns
    -------
    (normalized_df, stats) where stats = {biomarker: {"mean": ..., "std": ...}}
    -- keep `stats` if you ever need to invert the transform.
    """
    biomarker_cols = biomarker_cols or BIOMARKER_COLS
    df = df.copy()
    stats = {}

    for col in biomarker_cols:
        mu = df[col].mean()
        sigma = df[col].std(ddof=0)
        if not np.isfinite(sigma) or sigma < 1e-8:
            sigma = 1.0  # effectively constant column; avoid divide-by-zero
        df[col] = (df[col] - mu) / sigma
        stats[col] = {"mean": float(mu), "std": float(sigma)}

    return df, stats


# ---------------------------------------------------------------------------
# 5. Reshape into a CSD-ready structure
# ---------------------------------------------------------------------------
def build_subject_series_dict(df, biomarker_cols=None):
    """
    Reshape the long-format DataFrame into a nested dict keyed by subject
    and biomarker, each holding a clean, time-sorted, NaN-free 1-D array --
    exactly what early_warning_utils.compute_csd_indicators() expects as
    its `series` argument.

    Returns
    -------
    dict:
        { rid: { biomarker: {"months": np.ndarray, "values": np.ndarray} } }
    """
    biomarker_cols = biomarker_cols or BIOMARKER_COLS
    subject_dict = {}

    for rid, g in df.groupby("RID", sort=False):
        g = g.sort_values("MONTHS")
        subject_dict[rid] = {}
        for col in biomarker_cols:
            valid = g[["MONTHS", col]].dropna()
            subject_dict[rid][col] = {
                "months": valid["MONTHS"].to_numpy(),
                "values": valid[col].to_numpy(),
            }

    return subject_dict


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------
def load_and_preprocess(csv_path, biomarker_cols=None, interpolation="linear", spline_order=3):
    """
    Full pipeline: raw CSV -> validated -> time-axis added -> per-subject
    interpolation -> global Z-score normalization -> CSD-ready structure.

    Parameters
    ----------
    csv_path : str or Path
    biomarker_cols : list[str], optional (defaults to BIOMARKER_COLS)
    interpolation : {"linear", "spline"}
    spline_order : int, cubic (3) by default; only used if interpolation="spline"

    Returns
    -------
    dict with keys:
        'long_df'           : fully processed long-format DataFrame
        'interpolated_mask' : DataFrame flagging which cells were imputed
        'zscore_stats'      : {biomarker: {"mean": ..., "std": ...}}
        'subject_series'    : nested dict, ready for CSD indicator extraction
    """
    biomarker_cols = biomarker_cols or BIOMARKER_COLS

    raw = load_adni_csv(csv_path)
    dated = add_time_axis(raw)
    filled, mask = interpolate_all_subjects(
        dated, biomarker_cols, method=interpolation, spline_order=spline_order
    )
    normalized, stats = zscore_normalize(filled, biomarker_cols)
    subject_series = build_subject_series_dict(normalized, biomarker_cols)

    return {
        "long_df": normalized,
        "interpolated_mask": mask,
        "zscore_stats": stats,
        "subject_series": subject_series,
    }


# ---------------------------------------------------------------------------
# Synthetic demo -- verifies the whole module with NO real ADNI data needed
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(42)
    viscode_options = ["bl", "m06", "m12", "m18", "m24", "m36"]
    rows = []

    for rid in range(1001, 1011):  # 10 synthetic subjects
        n_visits = rng.integers(4, len(viscode_options) + 1)
        visits = viscode_options[:n_visits]

        base_abeta = rng.normal(900, 100)
        base_tau = rng.normal(250, 40)

        for i, vc in enumerate(visits):
            rows.append(
                {
                    "RID": rid,
                    "VISCODE": vc,
                    # Amyloid tends to decline, tau tends to rise with progression --
                    # loosely mimicking real AD biomarker trends for a realistic demo.
                    "ABETA": base_abeta - i * rng.uniform(5, 20) + rng.normal(0, 15),
                    "TAU": base_tau + i * rng.uniform(3, 10) + rng.normal(0, 10),
                    "PTAU": rng.normal(30, 5) + i * rng.uniform(0.5, 2),
                    "FDG": rng.normal(1.2, 0.1) - i * rng.uniform(0.01, 0.03),
                    "Hippocampus": rng.normal(7000, 500) - i * rng.uniform(20, 60),
                }
            )

    demo_df = pd.DataFrame(rows)

    # Inject missingness (~10% per biomarker) to actually exercise the
    # interpolation logic instead of testing on a suspiciously clean CSV.
    for col in BIOMARKER_COLS:
        drop_mask = rng.random(len(demo_df)) < 0.10
        demo_df.loc[drop_mask, col] = np.nan

    with tempfile.TemporaryDirectory() as tmp_dir:
        demo_csv_path = Path(tmp_dir) / "synthetic_adni_demo.csv"
        demo_df.to_csv(demo_csv_path, index=False)

        print(f"[demo] Synthetic ADNI-style CSV written to: {demo_csv_path}")
        print(f"[demo] Raw missing values per biomarker:\n{demo_df[BIOMARKER_COLS].isna().sum()}\n")

        result = load_and_preprocess(demo_csv_path, interpolation="spline")

        print("[demo] Processed long_df (head):")
        print(result["long_df"].head(8).to_string(index=False))

        remaining_na = result["long_df"][BIOMARKER_COLS].isna().sum()
        print("\n[demo] Remaining NaNs after interpolation (0 unless a subject had ALL visits missing for a biomarker):")
        print(remaining_na)
        assert remaining_na.sum() == 0, "Unexpected NaNs remained after interpolation."

        print("\n[demo] Z-score normalization stats used:")
        for biomarker, s in result["zscore_stats"].items():
            print(f"  {biomarker}: mean={s['mean']:.3f}, std={s['std']:.3f}")
            assert s["std"] > 0, "Std should never be non-positive after the zero-guard."

        example_rid = next(iter(result["subject_series"]))
        example_series = result["subject_series"][example_rid]["TAU"]
        print(f"\n[demo] CSD-ready series for subject {example_rid}, biomarker 'TAU':")
        print("  months:", example_series["months"])
        print("  values:", np.round(example_series["values"], 3))
        assert len(example_series["months"]) == len(example_series["values"])

        print("\n[demo] adni_data_loader.py verified successfully -- no errors raised.")