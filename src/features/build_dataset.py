import pandas as pd
from src.utils.config import DATA_RAW, DATA_PROCESSED
from src.utils.scoring import calculate_ppr_points, tier_label


def load_raw_data():
    """Load all three raw datasets."""
    draft = pd.read_csv(DATA_RAW / "draft_history.csv")
    stats = pd.read_csv(DATA_RAW / "nfl_stats.csv")
    combine = pd.read_csv(DATA_RAW / "combine_data.csv")

    print(f"Draft: {draft.shape}")
    print(f"Stats: {stats.shape}")
    print(f"Combine: {combine.shape}")

    return draft, stats, combine


def get_rookie_stats(draft: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Merge draft picks with their rookie year NFL stats."""
    # merge on player_id and rookie season
    merged = draft.merge(
        stats,
        left_on="player_id",
        right_on="player_id",
        how="left"
    )

    # keep only rookie year stats
    merged = merged[merged["season"] == merged["draft_year"]].reset_index(drop=True)

    print(f"\nPlayers with rookie stats: {len(merged)}")
    print(f"Players without rookie stats: {len(draft) - len(merged)}")

    return merged


def add_combine(df: pd.DataFrame, combine: pd.DataFrame) -> pd.DataFrame:
    """Merge combine data onto the dataset."""
    combine_cols = ["pfr_id", "ht", "wt", "forty", "bench", "vertical",
                    "broad_jump", "cone", "shuttle"]
    combine_slim = combine[[c for c in combine_cols if c in combine.columns]]
    combine_slim = combine_slim.drop_duplicates(subset=["pfr_id"])

    merged = df.merge(combine_slim, on="pfr_id", how="left")

    has_combine = merged["forty"].notna().sum()
    print(f"Players with combine data: {has_combine}/{len(merged)}")

    return merged


def add_fantasy_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate PPR fantasy points and per-game averages."""
    # calculate PPR points from raw stats
    df["ppr_calculated"] = df.apply(lambda row: calculate_ppr_points({
        "pass_yd": row.get("passing_yards", 0) or 0,
        "pass_td": row.get("passing_tds", 0) or 0,
        "interception": row.get("interceptions", 0) or 0,
        "rush_yd": row.get("rushing_yards", 0) or 0,
        "rush_td": row.get("rushing_tds", 0) or 0,
        "reception": row.get("receptions", 0) or 0,
        "rec_yd": row.get("receiving_yards", 0) or 0,
        "rec_td": row.get("receiving_tds", 0) or 0,
        "fumble_lost": (row.get("rushing_fumbles_lost", 0) or 0) +
                       (row.get("receiving_fumbles_lost", 0) or 0)
    }), axis=1)

    # use nfl_data_py's PPR if available, otherwise our calculation
    df["ppr_total"] = df["fantasy_points_ppr"].fillna(df["ppr_calculated"])

    # per game
    df["games"] = pd.to_numeric(df["games"], errors="coerce").fillna(0)
    df["ppg"] = (df["ppr_total"] / df["games"]).where(df["games"] > 0, 0).round(2)

    # tier labels
    df["tier"] = df.apply(lambda row: tier_label(row["ppg"], row["pos"]), axis=1)

    print(f"\nFantasy tiers:")
    print(df["tier"].value_counts())

    return df


def clean_final(df: pd.DataFrame) -> pd.DataFrame:
    """Final cleanup and column selection."""
    df = df.copy()

    keep = [
        "player", "pos", "college", "draft_year", "round", "pick", "team",
        "player_id", "pfr_id", "cfb_player_id",
        "ht", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle",
        "age",
        "games",
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "receiving_yards_after_catch", "target_share",
        "ppr_total", "ppg", "tier"
    ]
    df = df[[c for c in keep if c in df.columns]]

    stat_cols = [
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "receiving_yards_after_catch", "target_share",
        "ppr_total", "ppg"
    ]
    for col in stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def save(df: pd.DataFrame) -> None:
    path = DATA_PROCESSED / "rookie_dataset.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved to {path}")


def run():
    draft, stats, combine = load_raw_data()
    df = get_rookie_stats(draft, stats)
    df = add_combine(df, combine)
    df = add_fantasy_scoring(df)
    df = clean_final(df)

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"\nSample:")
    print(df[["player", "pos", "draft_year", "ppg", "tier"]].head(10))

    save(df)
    return df


if __name__ == "__main__":
    run()