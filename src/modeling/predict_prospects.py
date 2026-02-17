import pandas as pd
import numpy as np
from src.utils.config import DATA_RAW, DATA_PROCESSED, MODELS_DIR
from src.modeling.fantasy_predictor import predict_player
from src.modeling.comparable_finder import (
    find_comparables, prepare_data, height_to_inches
)


def load_prospects() -> pd.DataFrame:
    """Load 2026 prospects."""
    path = DATA_RAW / "prospects_2026.csv"
    df = pd.read_csv(path)
    if "ht_inches" not in df.columns:
        df["ht_inches"] = df["ht"].apply(height_to_inches)
    return df


def load_historical() -> pd.DataFrame:
    """Load historical dataset for comparables."""
    path = DATA_PROCESSED / "rookie_dataset.csv"
    df = pd.read_csv(path)
    df = prepare_data(df)
    return df


def classify_prospect_archetype(row: pd.Series) -> str:
    """Classify archetype for a prospect based on college stats."""
    pos = row.get("pos", "")

    if pos == "QB":
        rush_att = row.get("college_rush_att", 0) or 0
        pass_att = row.get("college_pass_att", 0) or 0
        if pass_att > 0 and rush_att / pass_att >= 0.15:
            return "Dual Threat"
        return "Pocket Passer"

    elif pos == "RB":
        wt = row.get("wt", 0) or 0
        rec = row.get("college_rec", 0) or 0
        rush = row.get("college_rush_att", 0) or 0
        rec_ratio = rec / rush if rush > 0 else 0

        if rec_ratio >= 0.15 and rush >= 400:
            return "3-Down Back"
        if rec_ratio >= 0.15:
            return "Receiving Back"
        if wt >= 220:
            return "Power Back"
        return "Speed Back"

    elif pos == "WR":
        ht = row.get("ht_inches", 0) or 0
        rec_yd = row.get("college_rec_yd", 0) or 0
        rec = row.get("college_rec", 0) or 0
        ypr = rec_yd / rec if rec > 0 else 0

        if ypr >= 16:
            return "Deep Threat"
        if ht <= 72:
            return "Slot"
        if ht >= 75:
            return "Alpha/X"
        if ypr < 13:
            return "Possession"
        return "Alpha/X"

    elif pos == "TE":
        rec = row.get("college_rec", 0) or 0
        if rec >= 120:
            return "Receiving TE"
        if rec >= 80:
            return "Hybrid TE"
        return "Blocking TE"

    return "Unknown"


def run():
    """Predict fantasy outcomes for all prospects."""
    prospects = load_prospects()
    historical = load_historical()

    print(f"Predicting {len(prospects)} prospects...\n")

    results = []

    for i, row in prospects.iterrows():
        player = row["player"]
        pos = row["pos"]

        if pd.isna(player):
            continue

        archetype = classify_prospect_archetype(row)

        try:
            player_data = row.to_dict()
            prediction = predict_player(player_data, pos)
        except Exception as e:
            print(f"  Error predicting {player}: {e}")
            continue

        comps = find_comparables(
            row, historical, pos, archetype,
            top_n=3, same_archetype_only=True
        )
        if comps.empty or len(comps) < 3:
            comps = find_comparables(
                row, historical, pos, archetype,
                top_n=3, same_archetype_only=False
            )

        comp_names = comps["player"].tolist() if not comps.empty else []
        comp_ppgs = comps["ppg"].tolist() if not comps.empty else []

        result = {
            "player": player,
            "pos": pos,
            "college": row.get("college", ""),
            "archetype": archetype,
            "projected_pick": row.get("pick", 0),
            "predicted_ppg": prediction["predicted_ppg"],
            "predicted_tier": prediction["predicted_tier"],
            "tier_probs": prediction["tier_probabilities"],
            "comp_1": comp_names[0] if len(comp_names) > 0 else "",
            "comp_1_ppg": comp_ppgs[0] if len(comp_ppgs) > 0 else 0,
            "comp_2": comp_names[1] if len(comp_names) > 1 else "",
            "comp_2_ppg": comp_ppgs[1] if len(comp_ppgs) > 1 else 0,
            "comp_3": comp_names[2] if len(comp_names) > 2 else "",
            "comp_3_ppg": comp_ppgs[2] if len(comp_ppgs) > 2 else 0,
        }
        results.append(result)

    results_df = pd.DataFrame(results)

    # display
    for pos in ["QB", "RB", "WR", "TE"]:
        subset = results_df[results_df["pos"] == pos]
        if subset.empty:
            continue
        print(f"\n{'='*70}")
        print(f"  {pos}s")
        print(f"{'='*70}")
        for _, row in subset.iterrows():
            print(f"\n  {row['player']} ({row['college']})")
            print(f"    Archetype: {row['archetype']}")
            print(f"    Projected Pick: {row['projected_pick']}")
            print(f"    Predicted PPG: {row['predicted_ppg']}")
            print(f"    Predicted Tier: {row['predicted_tier']}")
            print(f"    Tier Probs: {row['tier_probs']}")
            print(f"    Comps: {row['comp_1']} ({row['comp_1_ppg']}), "
                  f"{row['comp_2']} ({row['comp_2_ppg']}), "
                  f"{row['comp_3']} ({row['comp_3_ppg']})")

    path = DATA_PROCESSED / "prospect_predictions.csv"
    results_df.to_csv(path, index=False)
    print(f"\nSaved to {path}")

    return results_df


if __name__ == "__main__":
    run()