import nfl_data_py as nfl
import pandas as pd
from src.utils.config import YEARS, POSITIONS, DATA_RAW


def pull_combine() -> pd.DataFrame:
    """Pull combine data for our draft classes."""
    print("Pulling combine data...")
    df = nfl.import_combine_data()

    # filter years and positions
    df = df[df["season"].isin(YEARS)]
    df = df[df["pos"].isin(POSITIONS)]
    df = df.reset_index(drop=True)

    cols = [
        "season", "pfr_id", "player_name", "pos", "school",
        "ht", "wt", "forty", "bench", "vertical", "broad_jump",
        "cone", "shuttle"
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.rename(columns={
        "season": "draft_year",
        "player_name": "player",
        "school": "college"
    })

    print(f"Total combine entries: {len(df)}")
    print(df["pos"].value_counts())
    return df


def save(df: pd.DataFrame) -> None:
    path = DATA_RAW / "combine_data.csv"
    df.to_csv(path, index=False)
    print(f"Saved to {path}")


def run():
    df = pull_combine()
    save(df)
    return df


if __name__ == "__main__":
    run()