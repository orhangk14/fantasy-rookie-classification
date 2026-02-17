import nfl_data_py as nfl
import pandas as pd
from src.utils.config import DATA_RAW, DATA_PROCESSED


def fix_college_stats() -> pd.DataFrame:
    """
    The nfl_data_py draft_picks 'college stats' are actually NFL career stats.
    We need to remove them and note this limitation.
    For historical players, we'll use draft capital + combine as primary features.
    The college stats in draft_picks are NFL career totals - NOT college production.
    """
    print("Fixing college stats issue...")
    print("NOTE: nfl_data_py draft_picks stat columns are NFL career stats, not college.")
    print("Historical model will rely on draft capital + combine + archetype features.")

    path = DATA_PROCESSED / "rookie_dataset.csv"
    df = pd.read_csv(path)

    # remove the mislabeled college columns
    college_cols = [c for c in df.columns if c.startswith("college_")]
    print(f"Removing mislabeled columns: {college_cols}")
    df = df.drop(columns=college_cols, errors="ignore")

    df.to_csv(path, index=False)
    print(f"Updated {path}")
    print(f"Remaining columns: {df.columns.tolist()}")

    return df


def run():
    return fix_college_stats()


if __name__ == "__main__":
    run()