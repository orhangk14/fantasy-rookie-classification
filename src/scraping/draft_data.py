import nfl_data_py as nfl
import pandas as pd
from src.utils.config import YEARS, POSITIONS, DATA_RAW


def pull_draft_picks() -> pd.DataFrame:
    """Pull draft picks and filter to offensive skill positions 2015-2025."""
    print("Pulling draft picks...")
    df = nfl.import_draft_picks()

    # filter years and positions
    df = df[df["season"].isin(YEARS)]
    df = df[df["position"].isin(POSITIONS)]
    df = df.reset_index(drop=True)

    # keep useful columns
    cols = [
        "season", "round", "pick", "team", "gsis_id", "pfr_player_id",
        "cfb_player_id", "pfr_player_name", "position", "college", "age"
    ]
    df = df[cols]
    df = df.rename(columns={
        "season": "draft_year",
        "pfr_player_name": "player",
        "position": "pos",
        "pfr_player_id": "pfr_id",
        "gsis_id": "player_id"
    })

    print(f"Total offensive skill players: {len(df)}")
    print(df["pos"].value_counts())
    return df


def save(df: pd.DataFrame) -> None:
    path = DATA_RAW / "draft_history.csv"
    df.to_csv(path, index=False)
    print(f"Saved to {path}")


def run():
    df = pull_draft_picks()
    save(df)
    return df


if __name__ == "__main__":
    run()