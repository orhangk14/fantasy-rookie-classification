import pandas as pd
import numpy as np
from src.utils.config import DATA_PROCESSED


def height_to_inches(ht: str) -> float:
    """Convert '6-4' format to inches (76)."""
    if pd.isna(ht):
        return np.nan
    try:
        parts = str(ht).split("-")
        return int(parts[0]) * 12 + int(parts[1])
    except (ValueError, IndexError):
        return np.nan


def classify_qb(row: pd.Series) -> str:
    """Classify QB archetype based on rushing volume."""
    rush_att = row.get("carries", 0) or 0
    rush_yd = row.get("rushing_yards", 0) or 0
    games = row.get("games", 1) or 1

    rush_att_pg = rush_att / games
    rush_yd_pg = rush_yd / games

    if rush_att_pg >= 4 or rush_yd_pg >= 20:
        return "Dual Threat"
    return "Pocket Passer"


def classify_rb(row: pd.Series) -> str:
    """Classify RB archetype based on build, speed, and usage."""
    wt = row.get("wt", None)
    forty = row.get("forty", None)
    receptions = row.get("receptions", 0) or 0
    carries = row.get("carries", 0) or 0
    games = row.get("games", 1) or 1

    rec_pg = receptions / games
    carry_pg = carries / games

    if rec_pg >= 3 and carry_pg < 10:
        return "Receiving Back"

    if rec_pg >= 2 and carry_pg >= 10:
        return "3-Down Back"

    if wt and wt >= 220:
        if forty and forty >= 4.55:
            return "Power Back"
        if carry_pg >= 10:
            return "3-Down Back"
        return "Power Back"

    if forty and forty < 4.45:
        return "Speed Back"

    if wt and wt < 210:
        return "Speed Back"

    return "Power Back"


def classify_wr(row: pd.Series) -> str:
    """Classify WR archetype based on size, route type, and usage."""
    ht = row.get("ht_inches", None)
    wt = row.get("wt", None)
    rec_yd = row.get("receiving_yards", 0) or 0
    receptions = row.get("receptions", 0) or 0
    targets = row.get("targets", 0) or 0
    yac = row.get("receiving_yards_after_catch", 0) or 0
    games = row.get("games", 1) or 1

    ypr = rec_yd / receptions if receptions > 0 else 0
    yac_pct = yac / rec_yd if rec_yd > 0 else 0
    catch_rate = receptions / targets if targets > 0 else 0

    if ypr >= 15:
        return "Deep Threat"

    if ht and ht <= 72:
        return "Slot"

    if catch_rate >= 0.65 and ypr < 13:
        return "Possession"

    if ht and ht >= 73:
        return "Alpha/X"

    if yac_pct >= 0.5:
        return "Slot"

    return "Possession"


def classify_te(row: pd.Series) -> str:
    """Classify TE archetype based on receiving usage."""
    receptions = row.get("receptions", 0) or 0
    targets = row.get("targets", 0) or 0
    games = row.get("games", 1) or 1

    rec_pg = receptions / games
    tgt_pg = targets / games

    if tgt_pg >= 4:
        return "Receiving TE"

    if tgt_pg >= 2 or rec_pg >= 1.5:
        return "Hybrid TE"

    return "Blocking TE"


def classify_archetype(row: pd.Series) -> str:
    """Route to position-specific classifier."""
    pos = row.get("pos", "")

    if pos == "QB":
        return classify_qb(row)
    elif pos == "RB":
        return classify_rb(row)
    elif pos == "WR":
        return classify_wr(row)
    elif pos == "TE":
        return classify_te(row)
    return "Unknown"


def run():
    """Add archetypes to the rookie dataset."""
    path = DATA_PROCESSED / "rookie_dataset.csv"
    df = pd.read_csv(path)

    print(f"Classifying archetypes for {len(df)} players...\n")

    # convert height to inches
    df["ht_inches"] = df["ht"].apply(height_to_inches)

    df["archetype"] = df.apply(classify_archetype, axis=1)

    # drop helper column
    df = df.drop(columns=["ht_inches"])

    # show breakdown by position
    for pos in ["QB", "RB", "WR", "TE"]:
        subset = df[df["pos"] == pos]
        print(f"\n{pos}:")
        print(subset["archetype"].value_counts().to_string())

    # show some examples
    print("\n\nSample players with archetypes:")
    sample_cols = ["player", "pos", "archetype", "ppg", "tier"]
    for pos in ["QB", "RB", "WR", "TE"]:
        subset = df[df["pos"] == pos].head(5)
        print(f"\n{pos}:")
        print(subset[sample_cols].to_string(index=False))

    # save
    df.to_csv(path, index=False)
    print(f"\nUpdated {path}")

    return df


if __name__ == "__main__":
    run()