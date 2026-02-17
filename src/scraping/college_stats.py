import nfl_data_py as nfl
import pandas as pd
from src.utils.config import DATA_RAW, DATA_PROCESSED


def pull_college_stats() -> pd.DataFrame:
    """Pull college stats and merge with draft data to get pre-draft production."""
    print("Loading draft data...")
    draft = pd.read_csv(DATA_RAW / "draft_history.csv")

    # nfl_data_py doesn't have college stats directly
    # but the draft_picks data has career college stats baked in
    # let's pull the full draft data with those columns
    print("Pulling full draft picks with college stats...")
    full = nfl.import_draft_picks()

    # filter to our players
    full = full[full["season"].isin(range(2015, 2026))]
    full = full[full["position"].isin(["QB", "RB", "WR", "TE"])]

    cols = [
        "season", "pfr_player_id", "pfr_player_name", "position",
        "pass_completions", "pass_attempts", "pass_yards", "pass_tds", "pass_ints",
        "rush_atts", "rush_yards", "rush_tds",
        "receptions", "rec_yards", "rec_tds"
    ]
    full = full[[c for c in cols if c in full.columns]]

    full = full.rename(columns={
        "season": "draft_year",
        "pfr_player_id": "pfr_id",
        "pfr_player_name": "player",
        "position": "pos",
        "pass_completions": "college_pass_cmp",
        "pass_attempts": "college_pass_att",
        "pass_yards": "college_pass_yd",
        "pass_tds": "college_pass_td",
        "pass_ints": "college_int",
        "rush_atts": "college_rush_att",
        "rush_yards": "college_rush_yd",
        "rush_tds": "college_rush_td",
        "receptions": "college_rec",
        "rec_yards": "college_rec_yd",
        "rec_tds": "college_rec_td"
    })

    print(f"College stats shape: {full.shape}")
    print(f"\nSample:")
    print(full.head(10).to_string())

    return full


def merge_college_to_dataset(college: pd.DataFrame) -> pd.DataFrame:
    """Merge college stats into the main rookie dataset."""
    path = DATA_PROCESSED / "rookie_dataset.csv"
    df = pd.read_csv(path)

    # drop any existing college columns to avoid dupes
    college_cols = [c for c in df.columns if c.startswith("college_")]
    df = df.drop(columns=college_cols, errors="ignore")

    # merge on pfr_id and draft_year
    college_slim = college.drop(columns=["player", "pos"], errors="ignore")
    merged = df.merge(college_slim, on=["pfr_id", "draft_year"], how="left")

    # count how many have college stats
    has_college = merged["college_pass_yd"].notna() | merged["college_rush_yd"].notna()
    print(f"\nPlayers with college stats: {has_college.sum()}/{len(merged)}")

    return merged


def save(df: pd.DataFrame) -> None:
    path = DATA_PROCESSED / "rookie_dataset.csv"
    df.to_csv(path, index=False)
    print(f"Saved to {path}")


def run():
    college = pull_college_stats()
    df = merge_college_to_dataset(college)
    save(df)
    return df


if __name__ == "__main__":
    run()