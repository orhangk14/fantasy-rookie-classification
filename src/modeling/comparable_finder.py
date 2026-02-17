import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.config import DATA_PROCESSED


# features used for comparison by position
COMPARISON_FEATURES = {
    "QB": [
        "round", "pick", "age", "ht_inches", "wt", "forty",
        "completions", "attempts", "passing_yards", "passing_tds",
        "interceptions", "carries", "rushing_yards", "rushing_tds", "games"
    ],
    "RB": [
        "round", "pick", "age", "ht_inches", "wt", "forty",
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds", "games"
    ],
    "WR": [
        "round", "pick", "age", "ht_inches", "wt", "forty",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "receiving_yards_after_catch", "target_share", "games"
    ],
    "TE": [
        "round", "pick", "age", "ht_inches", "wt", "forty",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "receiving_yards_after_catch", "games"
    ]
}


def height_to_inches(ht: str) -> float:
    """Convert '6-4' format to inches."""
    if pd.isna(ht):
        return np.nan
    try:
        parts = str(ht).split("-")
        return int(parts[0]) * 12 + int(parts[1])
    except (ValueError, IndexError):
        return np.nan


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset with height in inches."""
    df = df.copy()
    df["ht_inches"] = df["ht"].apply(height_to_inches)
    return df


def find_comparables(
    player_row: pd.Series,
    dataset: pd.DataFrame,
    pos: str,
    archetype: str = None,
    top_n: int = 5,
    same_archetype_only: bool = True
) -> pd.DataFrame:
    """Find most similar players using cosine similarity."""

    features = COMPARISON_FEATURES.get(pos, [])
    if not features:
        return pd.DataFrame()

    # filter to same position
    pool = dataset[dataset["pos"] == pos].copy()

    # optionally filter to same archetype
    if same_archetype_only and archetype:
        pool = pool[pool["archetype"] == archetype]

    if len(pool) < 2:
        return pd.DataFrame()

    # exclude the player themselves if they're in the dataset
    player_name = player_row.get("player", "")
    pool = pool[pool["player"] != player_name].copy()

    if len(pool) == 0:
        return pd.DataFrame()

    # get available features
    available = [f for f in features if f in pool.columns and f in player_row.index]
    if not available:
        return pd.DataFrame()

    # build feature matrices
    pool_features = pool[available].copy()
    player_features = pd.DataFrame([player_row[available]])

    # fill NaN with column medians from pool
    medians = pool_features.median()
    pool_features = pool_features.fillna(medians)
    player_features = player_features.fillna(medians)

    # scale
    scaler = StandardScaler()
    pool_scaled = scaler.fit_transform(pool_features)
    player_scaled = scaler.transform(player_features)

    # cosine similarity
    similarities = cosine_similarity(player_scaled, pool_scaled)[0]

    pool = pool.copy()
    pool["similarity"] = similarities
    pool = pool.sort_values("similarity", ascending=False).head(top_n)

    pool["similarity"] = pool["similarity"].round(3)

    return pool


def show_comparables(player_name: str, dataset: pd.DataFrame, top_n: int = 5):
    """Find and display comparables for a player in the dataset."""
    player_rows = dataset[dataset["player"] == player_name]

    if player_rows.empty:
        print(f"Player '{player_name}' not found")
        return None

    player = player_rows.iloc[0]
    pos = player["pos"]
    archetype = player.get("archetype", None)

    print(f"\n{'='*60}")
    print(f"Comparables for: {player_name}")
    print(f"Position: {pos} | Archetype: {archetype}")
    print(f"Rookie PPG: {player['ppg']} | Tier: {player['tier']}")
    print(f"{'='*60}\n")

    comps = find_comparables(player, dataset, pos, archetype, top_n)

    if comps.empty:
        print("No comparables found")
        return None

    display_cols = ["player", "draft_year", "archetype", "ppg", "tier", "similarity"]
    print(comps[display_cols].to_string(index=False))

    return comps


def run():
    """Demo the comparable finder."""
    path = DATA_PROCESSED / "rookie_dataset.csv"
    df = pd.read_csv(path)
    df = prepare_data(df)

    # test with some well-known players
    test_players = [
        "Lamar Jackson",
        "Saquon Barkley",
        "Justin Jefferson",
        "Kyle Pitts"
    ]

    for name in test_players:
        show_comparables(name, df)
        print()


if __name__ == "__main__":
    run()