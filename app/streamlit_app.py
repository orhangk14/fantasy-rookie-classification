import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import DATA_PROCESSED, DATA_RAW
from src.modeling.comparable_finder import prepare_data, find_comparables, height_to_inches
from src.modeling.fantasy_predictor import predict_player

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Fantasy Rookie Draft Model",
    page_icon="🏈",
    layout="wide"
)

# ── Load Data ──────────────────────────────────────────
@st.cache_data
def load_prospects():
    df = pd.read_csv(DATA_PROCESSED / "prospect_predictions.csv")
    return df

@st.cache_data
def load_historical():
    df = pd.read_csv(DATA_PROCESSED / "rookie_dataset.csv")
    df = prepare_data(df)
    return df

@st.cache_data
def load_raw_prospects():
    df = pd.read_csv(DATA_RAW / "prospects_2026.csv")
    df["ht_inches"] = df["ht"].apply(height_to_inches)
    return df

prospects = load_prospects()
historical = load_historical()
raw_prospects = load_raw_prospects()

# ── Sidebar ────────────────────────────────────────────
st.sidebar.title("🏈 Fantasy Rookie Model")
page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "🔍 Prospect Explorer",
    "📊 Comparables",
    "📈 Historical Data",
    "🎯 Custom Prospect"
])

# ── Color Maps ─────────────────────────────────────────
TIER_COLORS = {
    "Elite": "#2ecc71",
    "Starter": "#3498db",
    "Bench": "#f39c12",
    "Bust": "#e74c3c"
}

ARCHETYPE_COLORS = {
    "Pocket Passer": "#3498db",
    "Dual Threat": "#e74c3c",
    "Power Back": "#e67e22",
    "Speed Back": "#2ecc71",
    "Receiving Back": "#9b59b6",
    "3-Down Back": "#1abc9c",
    "Alpha/X": "#e74c3c",
    "Slot": "#3498db",
    "Deep Threat": "#f39c12",
    "Possession": "#2ecc71",
    "Receiving TE": "#3498db",
    "Hybrid TE": "#9b59b6",
    "Blocking TE": "#95a5a6"
}


# ══════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🏈 2026 Fantasy Rookie Draft Predictions")
    st.markdown("---")

    # summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Prospects", len(prospects))
    col2.metric("Avg Predicted PPG", f"{prospects['predicted_ppg'].mean():.1f}")
    col3.metric("Projected Starters+", len(prospects[prospects["predicted_tier"].isin(["Elite", "Starter"])]))
    col4.metric("Projected Busts", len(prospects[prospects["predicted_tier"] == "Bust"]))

    st.markdown("---")

    # tier overview chart
    st.subheader("Prospect Tier Overview")
    tier_order = ["Elite", "Starter", "Bench", "Bust"]
    fig = px.bar(
        prospects.sort_values("predicted_ppg", ascending=False),
        x="player",
        y="predicted_ppg",
        color="predicted_tier",
        color_discrete_map=TIER_COLORS,
        category_orders={"predicted_tier": tier_order},
        hover_data=["pos", "archetype", "college"],
        title="Predicted PPG by Prospect"
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # by position
    st.subheader("Rankings by Position")

    for pos in ["QB", "RB", "WR", "TE"]:
        subset = prospects[prospects["pos"] == pos].sort_values("predicted_ppg", ascending=False)
        if subset.empty:
            continue

        st.markdown(f"### {pos}s")

        display_cols = ["player", "college", "archetype", "projected_pick",
                        "predicted_ppg", "predicted_tier", "comp_1", "comp_2", "comp_3"]
        available = [c for c in display_cols if c in subset.columns]

        styled = subset[available].reset_index(drop=True)
        styled.index = styled.index + 1
        styled.index.name = "Rank"

        st.dataframe(
            styled,
            use_container_width=True,
            column_config={
                "predicted_ppg": st.column_config.NumberColumn("Pred PPG", format="%.2f"),
                "projected_pick": st.column_config.NumberColumn("Pick", format="%d"),
            }
        )


# ══════════════════════════════════════════════════════════
# PAGE: Prospect Explorer
# ══════════════════════════════════════════════════════════
elif page == "🔍 Prospect Explorer":
    st.title("🔍 Prospect Deep Dive")
    st.markdown("---")

    selected = st.selectbox("Select a Prospect", prospects["player"].tolist())
    player = prospects[prospects["player"] == selected].iloc[0]
    raw = raw_prospects[raw_prospects["player"] == selected].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Position", player["pos"])
    col2.metric("Archetype", player["archetype"])
    col3.metric("College", player["college"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Projected Pick", int(player["projected_pick"]))
    col2.metric("Predicted PPG", f"{player['predicted_ppg']:.2f}")
    col3.metric("Predicted Tier", player["predicted_tier"])

    st.markdown("---")

    # tier probabilities
    st.subheader("Tier Probabilities")
    tier_probs = eval(player["tier_probs"]) if isinstance(player["tier_probs"], str) else player["tier_probs"]

    fig = go.Figure(go.Bar(
        x=list(tier_probs.keys()),
        y=list(tier_probs.values()),
        marker_color=[TIER_COLORS.get(t, "#95a5a6") for t in tier_probs.keys()]
    ))
    fig.update_layout(
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

    # college stats
    st.subheader("College Production")
    pos = player["pos"]

    if pos == "QB":
        stat_cols = {
            "college_pass_att": "Pass Att",
            "college_pass_yd": "Pass Yds",
            "college_pass_td": "Pass TD",
            "college_int": "INTs",
            "college_rush_att": "Rush Att",
            "college_rush_yd": "Rush Yds",
            "college_rush_td": "Rush TD"
        }
    elif pos == "RB":
        stat_cols = {
            "college_rush_att": "Rush Att",
            "college_rush_yd": "Rush Yds",
            "college_rush_td": "Rush TD",
            "college_rec": "Receptions",
            "college_rec_yd": "Rec Yds",
            "college_rec_td": "Rec TD"
        }
    elif pos == "WR":
        stat_cols = {
            "college_rec": "Receptions",
            "college_rec_yd": "Rec Yds",
            "college_rec_td": "Rec TD",
            "college_rush_att": "Rush Att",
            "college_rush_yd": "Rush Yds"
        }
    else:
        stat_cols = {
            "college_rec": "Receptions",
            "college_rec_yd": "Rec Yds",
            "college_rec_td": "Rec TD"
        }

    stats_data = {}
    for col, label in stat_cols.items():
        val = raw.get(col, 0)
        stats_data[label] = [val if pd.notna(val) else 0]

    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # comparables
    st.subheader("Player Comparables")
    comp_data = []
    for i in range(1, 4):
        name = player.get(f"comp_{i}", "")
        ppg = player.get(f"comp_{i}_ppg", 0)
        if name:
            comp_data.append({"Comparable": name, "Rookie PPG": ppg})

    if comp_data:
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# PAGE: Comparables
# ══════════════════════════════════════════════════════════
elif page == "📊 Comparables":
    st.title("📊 Comparable Finder")
    st.markdown("---")

    mode = st.radio("Search Mode", ["2026 Prospect", "Historical Player"])

    if mode == "2026 Prospect":
        selected = st.selectbox("Select Prospect", raw_prospects["player"].tolist())
        player_row = raw_prospects[raw_prospects["player"] == selected].iloc[0]
    else:
        pos_filter = st.selectbox("Position", ["QB", "RB", "WR", "TE"])
        filtered = historical[historical["pos"] == pos_filter]
        selected = st.selectbox("Select Player", filtered["player"].tolist())
        player_row = filtered[filtered["player"] == selected].iloc[0]

    pos = player_row["pos"]

    same_arch = st.checkbox("Same archetype only", value=True)
    top_n = st.slider("Number of comps", 3, 10, 5)

    archetype = player_row.get("archetype", None)
    if archetype is None and "pos" in player_row:
        from src.modeling.predict_prospects import classify_prospect_archetype
        archetype = classify_prospect_archetype(player_row)

    comps = find_comparables(
        player_row, historical, pos, archetype,
        top_n=top_n, same_archetype_only=same_arch
    )

    if not comps.empty:
        display = ["player", "draft_year", "archetype", "ppg", "tier", "similarity"]
        available = [c for c in display if c in comps.columns]

        st.dataframe(
            comps[available].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "ppg": st.column_config.NumberColumn("PPG", format="%.2f"),
                "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
            }
        )

        # chart
        fig = px.bar(
            comps.sort_values("similarity", ascending=True),
            x="similarity",
            y="player",
            orientation="h",
            color="tier",
            color_discrete_map=TIER_COLORS,
            title=f"Top Comps for {selected}"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No comparables found. Try unchecking 'Same archetype only'.")


# ══════════════════════════════════════════════════════════
# PAGE: Historical Data
# ══════════════════════════════════════════════════════════
elif page == "📈 Historical Data":
    st.title("📈 Historical Rookie Data (2015-2024)")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    pos_filter = col1.multiselect("Position", ["QB", "RB", "WR", "TE"], default=["QB", "RB", "WR", "TE"])
    tier_filter = col2.multiselect("Tier", ["Elite", "Starter", "Bench", "Bust"], default=["Elite", "Starter", "Bench", "Bust"])
    year_range = col3.slider("Draft Year", 2015, 2024, (2015, 2024))

    filtered = historical[
        (historical["pos"].isin(pos_filter)) &
        (historical["tier"].isin(tier_filter)) &
        (historical["draft_year"] >= year_range[0]) &
        (historical["draft_year"] <= year_range[1])
    ]

    st.markdown(f"**{len(filtered)} players**")

        # scatter plot
    fig = px.scatter(
        filtered,
        x="pick",
        y="ppg",
        color="pos",
        hover_data=["player", "archetype", "tier", "draft_year"],
        title="Draft Pick vs Rookie PPG"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # archetype breakdown
    st.subheader("Archetype Performance")
    arch_stats = filtered.groupby(["pos", "archetype"]).agg(
        count=("ppg", "count"),
        avg_ppg=("ppg", "mean"),
        median_ppg=("ppg", "median"),
        max_ppg=("ppg", "max")
    ).round(2).reset_index()

    st.dataframe(arch_stats, use_container_width=True, hide_index=True)

    # full table
    st.subheader("All Players")
    display_cols = ["player", "pos", "archetype", "college", "draft_year",
                    "round", "pick", "ppg", "tier"]
    available = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[available].sort_values("ppg", ascending=False).reset_index(drop=True),
        use_container_width=True,
        column_config={
            "ppg": st.column_config.NumberColumn("PPG", format="%.2f"),
        }
    )


# ══════════════════════════════════════════════════════════
# PAGE: Custom Prospect
# ══════════════════════════════════════════════════════════
elif page == "🎯 Custom Prospect":
    st.title("🎯 Custom Prospect Evaluator")
    st.markdown("Enter custom prospect details to get a fantasy prediction.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Player Name", "Custom Player")
        pos = st.selectbox("Position", ["QB", "RB", "WR", "TE"])
        college = st.text_input("College", "")
        round_num = st.number_input("Projected Round", 1, 7, 1)
        pick = st.number_input("Projected Pick", 1, 260, 15)
        age = st.number_input("Age", 19, 28, 22)

    with col2:
        ht_ft = st.number_input("Height (feet)", 5, 6, 6)
        ht_in = st.number_input("Height (inches)", 0, 11, 2)
        wt = st.number_input("Weight", 150, 350, 215)
        forty = st.number_input("40 Time (0 if unknown)", 0.0, 6.0, 0.0, step=0.01)

    st.markdown("### College Stats")

    if pos == "QB":
        c1, c2, c3, c4 = st.columns(4)
        college_pass_att = c1.number_input("Pass Att", 0, 5000, 1000)
        college_pass_yd = c2.number_input("Pass Yds", 0, 30000, 8000)
        college_pass_td = c3.number_input("Pass TD", 0, 200, 50)
        college_int = c4.number_input("INTs", 0, 100, 15)
        c1, c2, c3 = st.columns(3)
        college_rush_att = c1.number_input("Rush Att", 0, 2000, 100)
        college_rush_yd = c2.number_input("Rush Yds", 0, 10000, 300)
        college_rush_td = c3.number_input("Rush TD", 0, 100, 5)
    elif pos == "RB":
        c1, c2, c3 = st.columns(3)
        college_rush_att = c1.number_input("Rush Att", 0, 2000, 600)
        college_rush_yd = c2.number_input("Rush Yds", 0, 15000, 3000)
        college_rush_td = c3.number_input("Rush TD", 0, 100, 30)
        c1, c2, c3 = st.columns(3)
        college_rec = c1.number_input("Receptions", 0, 500, 50)
        college_rec_yd = c2.number_input("Rec Yds", 0, 5000, 400)
        college_rec_td = c3.number_input("Rec TD", 0, 50, 3)
    elif pos == "WR":
        c1, c2, c3 = st.columns(3)
        college_rec = c1.number_input("Receptions", 0, 500, 150)
        college_rec_yd = c2.number_input("Rec Yds", 0, 10000, 2500)
        college_rec_td = c3.number_input("Rec TD", 0, 100, 20)
        c1, c2 = st.columns(2)
        college_rush_att = c1.number_input("Rush Att", 0, 500, 10)
        college_rush_yd = c2.number_input("Rush Yds", 0, 5000, 50)
    else:
        c1, c2, c3 = st.columns(3)
        college_rec = c1.number_input("Receptions", 0, 500, 100)
        college_rec_yd = c2.number_input("Rec Yds", 0, 5000, 1200)
        college_rec_td = c3.number_input("Rec TD", 0, 50, 10)

    if st.button("🔮 Predict", type="primary"):
        ht_inches = ht_ft * 12 + ht_in

        player_data = {
            "player": name,
            "pos": pos,
            "college": college,
            "round": round_num,
            "pick": pick,
            "age": age,
            "ht": f"{ht_ft}-{ht_in}",
            "ht_inches": ht_inches,
            "wt": wt,
            "forty": forty if forty > 0 else None,
        }

        if pos == "QB":
            player_data.update({
                "college_pass_att": college_pass_att,
                "college_pass_yd": college_pass_yd,
                "college_pass_td": college_pass_td,
                "college_int": college_int,
                "college_rush_att": college_rush_att,
                "college_rush_yd": college_rush_yd,
                "college_rush_td": college_rush_td,
            })
        elif pos == "RB":
            player_data.update({
                "college_rush_att": college_rush_att,
                "college_rush_yd": college_rush_yd,
                "college_rush_td": college_rush_td,
                "college_rec": college_rec,
                "college_rec_yd": college_rec_yd,
                "college_rec_td": college_rec_td,
            })
        elif pos == "WR":
            player_data.update({
                "college_rec": college_rec,
                "college_rec_yd": college_rec_yd,
                "college_rec_td": college_rec_td,
                "college_rush_att": college_rush_att,
                "college_rush_yd": college_rush_yd,
            })
        else:
            player_data.update({
                "college_rec": college_rec,
                "college_rec_yd": college_rec_yd,
                "college_rec_td": college_rec_td,
            })

        # predict
        prediction = predict_player(player_data, pos)

        st.markdown("---")
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted PPG", f"{prediction['predicted_ppg']:.2f}")
        col2.metric("Predicted Tier", prediction["predicted_tier"])
        col3.metric("Position", pos)

        # tier probs
        tier_probs = prediction["tier_probabilities"]
        fig = go.Figure(go.Bar(
            x=list(tier_probs.keys()),
            y=list(tier_probs.values()),
            marker_color=[TIER_COLORS.get(t, "#95a5a6") for t in tier_probs.keys()]
        ))
        fig.update_layout(yaxis_title="Probability", yaxis_tickformat=".0%", height=350)
        st.plotly_chart(fig, use_container_width=True)

        # find comps
        from src.modeling.predict_prospects import classify_prospect_archetype
        player_series = pd.Series(player_data)
        archetype = classify_prospect_archetype(player_series)

        st.markdown(f"**Archetype:** {archetype}")

        comps = find_comparables(
            player_series, historical, pos, archetype,
            top_n=5, same_archetype_only=True
        )
        if comps.empty:
            comps = find_comparables(
                player_series, historical, pos, archetype,
                top_n=5, same_archetype_only=False
            )

        if not comps.empty:
            st.subheader("Comparables")
            display = ["player", "draft_year", "archetype", "ppg", "tier", "similarity"]
            available = [c for c in display if c in comps.columns]
            st.dataframe(comps[available].reset_index(drop=True), use_container_width=True, hide_index=True)