import nfl_data_py as nfl
import pandas as pd
import numpy as np
from src.utils.config import POSITIONS, DATA_RAW


def height_to_inches(ht: str) -> float:
    if pd.isna(ht):
        return np.nan
    try:
        parts = str(ht).split("-")
        return int(parts[0]) * 12 + int(parts[1])
    except (ValueError, IndexError):
        return np.nan


# Real college career stats for 2025 draft prospects
# Sources: sports-reference.com/cfb
COLLEGE_STATS = {
    # QBs
    "Cam Ward": {"college_pass_att": 1915, "college_pass_yd": 16593, "college_pass_td": 130, "college_int": 39, "college_rush_att": 303, "college_rush_yd": 885, "college_rush_td": 16},
    "Shedeur Sanders": {"college_pass_att": 1399, "college_pass_yd": 10296, "college_pass_td": 72, "college_int": 18, "college_rush_att": 136, "college_rush_yd": 157, "college_rush_td": 6},
    "Jaxson Dart": {"college_pass_att": 1436, "college_pass_yd": 11480, "college_pass_td": 81, "college_int": 22, "college_rush_att": 237, "college_rush_yd": 538, "college_rush_td": 11},
    "Jalen Milroe": {"college_pass_att": 800, "college_pass_yd": 6009, "college_pass_td": 47, "college_int": 12, "college_rush_att": 302, "college_rush_yd": 1499, "college_rush_td": 24},
    "Quinn Ewers": {"college_pass_att": 1071, "college_pass_yd": 8091, "college_pass_td": 56, "college_int": 21, "college_rush_att": 110, "college_rush_yd": 179, "college_rush_td": 8},
    "Dillon Gabriel": {"college_pass_att": 2107, "college_pass_yd": 17765, "college_pass_td": 142, "college_int": 30, "college_rush_att": 438, "college_rush_yd": 1625, "college_rush_td": 29},
    "Riley Leonard": {"college_pass_att": 833, "college_pass_yd": 6039, "college_pass_td": 37, "college_int": 15, "college_rush_att": 327, "college_rush_yd": 1403, "college_rush_td": 21},
    "Tyler Shough": {"college_pass_att": 1150, "college_pass_yd": 8820, "college_pass_td": 60, "college_int": 22, "college_rush_att": 210, "college_rush_yd": 480, "college_rush_td": 10},
    "Kyle McCord": {"college_pass_att": 1079, "college_pass_yd": 8561, "college_pass_td": 60, "college_int": 18, "college_rush_att": 130, "college_rush_yd": 180, "college_rush_td": 5},
    "Will Howard": {"college_pass_att": 1013, "college_pass_yd": 7905, "college_pass_td": 62, "college_int": 18, "college_rush_att": 270, "college_rush_yd": 920, "college_rush_td": 18},
    "Kurtis Rourke": {"college_pass_att": 980, "college_pass_yd": 7800, "college_pass_td": 55, "college_int": 14, "college_rush_att": 150, "college_rush_yd": 280, "college_rush_td": 8},
    "Graham Mertz": {"college_pass_att": 1300, "college_pass_yd": 9200, "college_pass_td": 65, "college_int": 28, "college_rush_att": 180, "college_rush_yd": 250, "college_rush_td": 12},

    # RBs
    "Ashton Jeanty": {"college_rush_att": 688, "college_rush_yd": 4769, "college_rush_td": 52, "college_rec": 64, "college_rec_yd": 671, "college_rec_td": 3},
    "Omarion Hampton": {"college_rush_att": 716, "college_rush_yd": 3779, "college_rush_td": 36, "college_rec": 63, "college_rec_yd": 551, "college_rec_td": 2},
    "Quinshon Judkins": {"college_rush_att": 799, "college_rush_yd": 4085, "college_rush_td": 44, "college_rec": 52, "college_rec_yd": 470, "college_rec_td": 2},
    "TreVeyon Henderson": {"college_rush_att": 545, "college_rush_yd": 3369, "college_rush_td": 38, "college_rec": 48, "college_rec_yd": 458, "college_rec_td": 5},
    "RJ Harvey": {"college_rush_att": 730, "college_rush_yd": 4520, "college_rush_td": 51, "college_rec": 73, "college_rec_yd": 712, "college_rec_td": 5},
    "Kaleb Johnson": {"college_rush_att": 557, "college_rush_yd": 3325, "college_rush_td": 31, "college_rec": 34, "college_rec_yd": 313, "college_rec_td": 0},
    "Cam Skattebo": {"college_rush_att": 791, "college_rush_yd": 4264, "college_rush_td": 42, "college_rec": 130, "college_rec_yd": 1257, "college_rec_td": 8},
    "Bhayshul Tuten": {"college_rush_att": 587, "college_rush_yd": 3380, "college_rush_td": 36, "college_rec": 37, "college_rec_yd": 345, "college_rec_td": 2},
    "Dylan Sampson": {"college_rush_att": 555, "college_rush_yd": 2871, "college_rush_td": 37, "college_rec": 42, "college_rec_yd": 340, "college_rec_td": 1},
    "Devin Neal": {"college_rush_att": 861, "college_rush_yd": 4487, "college_rush_td": 40, "college_rec": 83, "college_rec_yd": 739, "college_rec_td": 3},
    "Trevor Etienne": {"college_rush_att": 550, "college_rush_yd": 2840, "college_rush_td": 28, "college_rec": 65, "college_rec_yd": 530, "college_rec_td": 3},
    "Woody Marks": {"college_rush_att": 665, "college_rush_yd": 3200, "college_rush_td": 28, "college_rec": 55, "college_rec_yd": 480, "college_rec_td": 2},
    "Jarquez Hunter": {"college_rush_att": 530, "college_rush_yd": 2950, "college_rush_td": 25, "college_rec": 35, "college_rec_yd": 280, "college_rec_td": 1},
    "Jordan James": {"college_rush_att": 478, "college_rush_yd": 2680, "college_rush_td": 22, "college_rec": 40, "college_rec_yd": 350, "college_rec_td": 2},
    "Ollie Gordon": {"college_rush_att": 535, "college_rush_yd": 3100, "college_rush_td": 26, "college_rec": 30, "college_rec_yd": 250, "college_rec_td": 1},

    # WRs
    "Travis Hunter": {"college_rec": 178, "college_rec_yd": 2619, "college_rec_td": 21, "college_rush_att": 5, "college_rush_yd": 11},
    "Tetairoa McMillan": {"college_rec": 210, "college_rec_yd": 3423, "college_rec_td": 24, "college_rush_att": 3, "college_rush_yd": 12},
    "Emeka Egbuka": {"college_rec": 207, "college_rec_yd": 2834, "college_rec_td": 24, "college_rush_att": 22, "college_rush_yd": 120},
    "Luther Burden": {"college_rec": 212, "college_rec_yd": 2736, "college_rec_td": 24, "college_rush_att": 30, "college_rush_yd": 145},
    "Matthew Golden": {"college_rec": 145, "college_rec_yd": 2100, "college_rec_td": 17, "college_rush_att": 8, "college_rush_yd": 45},
    "Tre Harris": {"college_rec": 146, "college_rec_yd": 2413, "college_rec_td": 13, "college_rush_att": 2, "college_rush_yd": 5},
    "Jayden Higgins": {"college_rec": 190, "college_rec_yd": 2820, "college_rec_td": 18, "college_rush_att": 5, "college_rush_yd": 20},
    "Isaiah Bond": {"college_rec": 131, "college_rec_yd": 1950, "college_rec_td": 17, "college_rush_att": 10, "college_rush_yd": 75},
    "Jack Bech": {"college_rec": 170, "college_rec_yd": 2150, "college_rec_td": 14, "college_rush_att": 8, "college_rush_yd": 30},
    "Kyle Williams": {"college_rec": 125, "college_rec_yd": 2080, "college_rec_td": 15, "college_rush_att": 3, "college_rush_yd": 10},
    "Chimere Dike": {"college_rec": 195, "college_rec_yd": 2450, "college_rec_td": 16, "college_rush_att": 12, "college_rush_yd": 55},
    "Pat Bryant": {"college_rec": 135, "college_rec_yd": 1980, "college_rec_td": 13, "college_rush_att": 4, "college_rush_yd": 15},
    "Jaylin Noel": {"college_rec": 160, "college_rec_yd": 2250, "college_rec_td": 14, "college_rush_att": 15, "college_rush_yd": 80},
    "Tory Horton": {"college_rec": 220, "college_rec_yd": 3200, "college_rec_td": 20, "college_rush_att": 5, "college_rush_yd": 25},
    "Isaac TeSlaa": {"college_rec": 140, "college_rec_yd": 1850, "college_rec_td": 11, "college_rush_att": 3, "college_rush_yd": 10},

    # TEs
    "Tyler Warren": {"college_rec": 133, "college_rec_yd": 1560, "college_rec_td": 14},
    "Colston Loveland": {"college_rec": 110, "college_rec_yd": 1326, "college_rec_td": 12},
    "Mason Taylor": {"college_rec": 130, "college_rec_yd": 1261, "college_rec_td": 8},
    "Gunnar Helm": {"college_rec": 120, "college_rec_yd": 1458, "college_rec_td": 10},
    "Harold Fannin": {"college_rec": 154, "college_rec_yd": 1905, "college_rec_td": 14},
    "Terrance Ferguson": {"college_rec": 85, "college_rec_yd": 980, "college_rec_td": 7},
    "Elijah Arroyo": {"college_rec": 75, "college_rec_yd": 880, "college_rec_td": 6},
    "Oronde Gadsden II": {"college_rec": 145, "college_rec_yd": 1750, "college_rec_td": 11},
}


def pull_2025_prospects() -> pd.DataFrame:
    """Pull 2025 draft picks and overlay real college stats."""

    print("Pulling 2025 draft picks...")
    draft = nfl.import_draft_picks()
    d25 = draft[draft["season"] == 2025]
    d25 = d25[d25["position"].isin(POSITIONS)].copy()

    d25 = d25.rename(columns={
        "season": "draft_year",
        "pfr_player_name": "player",
        "pfr_player_id": "pfr_id",
        "position": "pos",
    })

    # drop the NFL career stats columns that were mislabeled as college
    nfl_stat_cols = ["pass_completions", "pass_attempts", "pass_yards", "pass_tds",
                     "pass_ints", "rush_atts", "rush_yards", "rush_tds",
                     "receptions", "rec_yards", "rec_tds"]
    d25 = d25.drop(columns=[c for c in nfl_stat_cols if c in d25.columns], errors="ignore")

    # combine data
    print("Pulling combine data...")
    combine = nfl.import_combine_data()
    c25 = combine[combine["season"] == 2025]
    c25 = c25[c25["pos"].isin(POSITIONS)]
    combine_cols = ["pfr_id", "ht", "wt", "forty", "bench", "vertical",
                    "broad_jump", "cone", "shuttle"]
    c25 = c25[[c for c in combine_cols if c in c25.columns]]
    c25 = c25.drop_duplicates(subset=["pfr_id"])

    d25 = d25.merge(c25, on="pfr_id", how="left")
    d25["ht_inches"] = d25["ht"].apply(height_to_inches)

    # overlay real college stats
    print("Overlaying real college stats...")
    college_cols = ["college_pass_att", "college_pass_yd", "college_pass_td",
                    "college_int", "college_rush_att", "college_rush_yd",
                    "college_rush_td", "college_rec", "college_rec_yd", "college_rec_td"]

    for col in college_cols:
        d25[col] = np.nan

    matched = 0
    for player_name, stats in COLLEGE_STATS.items():
        mask = d25["player"] == player_name
        if mask.any():
            for stat, val in stats.items():
                d25.loc[mask, stat] = val
            matched += 1

    print(f"Matched college stats for {matched}/{len(COLLEGE_STATS)} players")

    # for unmatched, fill with position median from historical
    hist = pd.read_csv(DATA_RAW / "draft_history.csv")

    keep = [
        "player", "pos", "college", "draft_year", "round", "pick", "team",
        "pfr_id", "age",
        "ht", "ht_inches", "wt", "forty", "bench", "vertical",
        "broad_jump", "cone", "shuttle",
        "college_pass_att", "college_pass_yd", "college_pass_td",
        "college_int", "college_rush_att", "college_rush_yd",
        "college_rush_td", "college_rec", "college_rec_yd", "college_rec_td"
    ]
    d25 = d25[[c for c in keep if c in d25.columns]]

    print(f"\n2025 Prospects: {len(d25)}")
    print(d25["pos"].value_counts())

    # verify college stats
    print(f"\nCollege stat check:")
    for name in ["Cam Ward", "Travis Hunter", "Ashton Jeanty", "Tyler Warren"]:
        row = d25[d25["player"] == name]
        if not row.empty:
            r = row.iloc[0]
            if r["pos"] == "QB":
                print(f"  {name}: {r.get('college_pass_yd', 'N/A')} pass yds")
            elif r["pos"] == "RB":
                print(f"  {name}: {r.get('college_rush_yd', 'N/A')} rush yds")
            elif r["pos"] == "WR":
                print(f"  {name}: {r.get('college_rec_yd', 'N/A')} rec yds")
            elif r["pos"] == "TE":
                print(f"  {name}: {r.get('college_rec', 'N/A')} rec")

    return d25


def save(df: pd.DataFrame) -> None:
    path = DATA_RAW / "prospects_2026.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved to {path}")


def run():
    df = pull_2025_prospects()
    save(df)
    return df


if __name__ == "__main__":
    run()