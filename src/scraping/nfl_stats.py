import nfl_data_py as nfl
import pandas as pd
from src.utils.config import DATA_RAW


# 2025 season hasn't happened yet
STAT_YEARS = list(range(2015, 2025))


def pull_seasonal_stats() -> pd.DataFrame:
    """Pull seasonal NFL stats for all years."""
    print(f"Pulling seasonal stats for {STAT_YEARS[0]}-{STAT_YEARS[-1]}...")
    df = nfl.import_seasonal_data(STAT_YEARS)

    cols = [
        "player_id", "season", "season_type",
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds", "rushing_fumbles_lost",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "receiving_fumbles_lost", "receiving_air_yards", "receiving_yards_after_catch",
        "target_share", "fantasy_points", "fantasy_points_ppr", "games"
    ]
    df = df[[c for c in cols if c in df.columns]]

    # regular season only
    df = df[df["season_type"] == "REG"].reset_index(drop=True)
    df = df.drop(columns=["season_type"])

    print(f"Total player-seasons: {len(df)}")
    return df


def save(df: pd.DataFrame) -> None:
    path = DATA_RAW / "nfl_stats.csv"
    df.to_csv(path, index=False)
    print(f"Saved to {path}")


def run():
    df = pull_seasonal_stats()
    save(df)
    return df


if __name__ == "__main__":
    run()