import pandas as pd
import numpy as np
from src.utils.config import DATA_RAW, DATA_PROCESSED, POSITIONS
from src.utils.scoring import calculate_ppr_points, tier_label
from src.modeling.fantasy_predictor import predict_player
from src.modeling.comparable_finder import height_to_inches
import nfl_data_py as nfl


def get_actual_2025_stats() -> pd.DataFrame:
    """Pull actual 2025 rookie stats from draft picks data."""
    print("Pulling 2025 draft class actual stats...")
    draft = nfl.import_draft_picks()
    d25 = draft[draft["season"] == 2025]
    d25 = d25[d25["position"].isin(POSITIONS)].copy()

    # combine data
    print("Pulling combine data...")
    combine = nfl.import_combine_data()
    c25 = combine[combine["season"] == 2025]
    c25 = c25[c25["pos"].isin(POSITIONS)]
    combine_cols = ["pfr_id", "ht", "wt", "forty", "bench", "vertical",
                    "broad_jump", "cone", "shuttle"]
    c25 = c25[[c for c in combine_cols if c in c25.columns]]
    c25 = c25.drop_duplicates(subset=["pfr_id"])

    d25 = d25.rename(columns={
        "season": "draft_year",
        "pfr_player_name": "player",
        "pfr_player_id": "pfr_id",
        "position": "pos",
    })

    # merge combine
    d25 = d25.merge(c25, on="pfr_id", how="left")
    d25["ht_inches"] = d25["ht"].apply(height_to_inches)

    # calculate actual PPR points
    d25["actual_ppr"] = d25.apply(lambda row: calculate_ppr_points({
        "pass_yd": row.get("pass_yards", 0) or 0,
        "pass_td": row.get("pass_tds", 0) or 0,
        "interception": row.get("pass_ints", 0) or 0,
        "rush_yd": row.get("rush_yards", 0) or 0,
        "rush_td": row.get("rush_tds", 0) or 0,
        "reception": row.get("receptions", 0) or 0,
        "rec_yd": row.get("rec_yards", 0) or 0,
        "rec_td": row.get("rec_tds", 0) or 0,
        "fumble_lost": 0
    }), axis=1)

    d25["games"] = pd.to_numeric(d25["games"], errors="coerce").fillna(0)
    d25["actual_ppg"] = (d25["actual_ppr"] / d25["games"]).where(d25["games"] > 0, 0).round(2)
    d25["actual_tier"] = d25.apply(lambda row: tier_label(row["actual_ppg"], row["pos"]), axis=1)

    print(f"2025 rookies with stats: {len(d25)}")
    return d25


def get_predictions(d25: pd.DataFrame) -> pd.DataFrame:
    """Load our saved predictions."""
    preds = pd.read_csv(DATA_PROCESSED / "prospect_predictions.csv")
    return preds


def validate() -> pd.DataFrame:
    """Compare predictions vs actual results."""
    actuals = get_actual_2025_stats()
    preds = get_predictions(actuals)

    # merge on player name
    merged = preds.merge(
        actuals[["player", "actual_ppg", "actual_tier", "actual_ppr", "games"]],
        on="player",
        how="inner"
    )

    if merged.empty:
        print("No matches found! Checking names...")
        print("\nPrediction names:", preds["player"].head(10).tolist())
        print("Actual names:", actuals["player"].head(10).tolist())
        return pd.DataFrame()

    # calculate errors
    merged["ppg_error"] = (merged["predicted_ppg"] - merged["actual_ppg"]).round(2)
    merged["ppg_abs_error"] = merged["ppg_error"].abs()
    merged["tier_correct"] = merged["predicted_tier"] == merged["actual_tier"]

    # overall metrics
    print(f"\n{'='*70}")
    print(f"  VALIDATION RESULTS — 2025 Rookie Class")
    print(f"{'='*70}")
    print(f"\n  Matched players: {len(merged)}")
    print(f"  Mean Absolute Error (PPG): {merged['ppg_abs_error'].mean():.2f}")
    print(f"  Median Absolute Error (PPG): {merged['ppg_abs_error'].median():.2f}")
    print(f"  Tier Accuracy: {merged['tier_correct'].mean():.1%}")

    # by position
    for pos in ["QB", "RB", "WR", "TE"]:
        subset = merged[merged["pos"] == pos]
        if subset.empty:
            continue
        print(f"\n  {pos}s ({len(subset)} players):")
        print(f"    MAE: {subset['ppg_abs_error'].mean():.2f}")
        print(f"    Tier Accuracy: {subset['tier_correct'].mean():.1%}")

    # detailed results
    for pos in ["QB", "RB", "WR", "TE"]:
        subset = merged[merged["pos"] == pos].sort_values("actual_ppg", ascending=False)
        if subset.empty:
            continue
        print(f"\n{'='*70}")
        print(f"  {pos}s — Predicted vs Actual")
        print(f"{'='*70}")
        for _, row in subset.iterrows():
            direction = "✅" if row["tier_correct"] else "❌"
            print(f"\n  {row['player']}")
            print(f"    Predicted: {row['predicted_ppg']:.1f} PPG ({row['predicted_tier']})")
            print(f"    Actual:    {row['actual_ppg']:.1f} PPG ({row['actual_tier']}) | {int(row['games'])} games")
            print(f"    Error:     {row['ppg_error']:+.1f} PPG | Tier: {direction}")

    # save
    path = DATA_PROCESSED / "validation_2025.csv"
    merged.to_csv(path, index=False)
    print(f"\nSaved to {path}")

    return merged


def run():
    return validate()


if __name__ == "__main__":
    run()