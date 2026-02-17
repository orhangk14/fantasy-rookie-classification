cat > README.md << 'EOF'
# 🏈 Fantasy Rookie Draft Prediction Model

A machine learning-powered tool that predicts NFL rookie fantasy performance (PPR scoring) based on draft capital, physical measurables, and college production. Includes player archetype classification, comparable player finder, model validation against the 2025 rookie class, and an interactive Streamlit dashboard.

> **⚠️ Update Coming:** The 2026 NFL Draft prospect class will be added once the draft takes place. The current model was validated against the 2025 rookie class with strong results.

---

## 📸 Features

- **Fantasy Predictions** — Predicted PPG and tier classification (Elite, Starter, Bench, Bust)
- **Archetype Classification** — Players categorized by playstyle (e.g., Dual Threat QB, 3-Down Back, Slot WR)
- **Comparable Player Finder** — Cosine similarity engine finding the most similar historical rookies
- **Model Validation** — Backtested against the 2025 rookie class with 67.4% tier accuracy
- **Interactive Dashboard** — Streamlit app with prospect explorer, comparables, validation, and custom evaluator
- **PPR Scoring** — Full PPR fantasy scoring system

---

## ✅ Model Validation (2025 Rookie Class)

The model was validated against all 86 drafted offensive skill players from the 2025 NFL Draft after their rookie season.

| Metric | Value |
|--------|-------|
| Overall Tier Accuracy | 67.4% |
| Mean Absolute Error (PPG) | 3.13 |
| Median Absolute Error (PPG) | 3.08 |

| Position | Players | MAE | Tier Accuracy |
|----------|---------|-----|---------------|
| QB | 14 | 4.12 | 85.7% |
| RB | 25 | 3.34 | 60.0% |
| WR | 31 | 2.89 | 67.7% |
| TE | 16 | 2.42 | 62.5% |

### Notable Predictions

| Player | Predicted | Actual | Error |
|--------|-----------|--------|-------|
| Cam Ward (QB) | 10.9 PPG | 11.8 PPG | -0.9 |
| Omarion Hampton (RB) | 14.2 PPG | 15.1 PPG | -0.8 |
| TreVeyon Henderson (RB) | 11.9 PPG | 12.2 PPG | -0.4 |
| Quinshon Judkins (RB) | 12.2 PPG | 12.0 PPG | +0.2 |
| Colston Loveland (TE) | 10.4 PPG | 10.3 PPG | +0.1 |
| Mason Taylor (TE) | 6.8 PPG | 6.7 PPG | +0.1 |
| Tyler Warren (TE) | 10.3 PPG | 11.1 PPG | -0.8 |

---

## 🏗️ Architecture

    Pull Data → Clean/Merge → Feature Engineer → Classify Archetypes → Train Models → Predict Prospects → Validate → Streamlit

Data Sources (all free): nfl_data_py for draft picks, NFL seasonal stats, and combine data. College stats manually sourced from Sports Reference.

Models: Gradient Boosting Regressor (PPG), Gradient Boosting Classifier (Tier with class balancing), Cosine Similarity (Comparables)

---

## 📂 Project Structure

    fantasy-rookie-classification/
    ├── app/
    │   └── streamlit_app.py
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── features/
    ├── models/
    ├── src/
    │   ├── scraping/
    │   │   ├── draft_data.py
    │   │   ├── nfl_stats.py
    │   │   ├── college_stats.py
    │   │   ├── combine_data.py
    │   │   └── prospects_2026.py
    │   ├── features/
    │   │   ├── build_dataset.py
    │   │   └── archetype_classifier.py
    │   ├── modeling/
    │   │   ├── fantasy_predictor.py
    │   │   ├── comparable_finder.py
    │   │   ├── predict_prospects.py
    │   │   └── validate_2025.py
    │   └── utils/
    │       ├── config.py
    │       └── scoring.py
    ├── .env
    ├── .gitignore
    ├── requirements.txt
    └── README.md

---

## 🚀 Quick Start

### 1. Clone and setup

    git clone https://github.com/YOUR_USERNAME/fantasy-rookie-classification.git
    cd fantasy-rookie-classification
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### 2. Run the full pipeline

    python -m src.scraping.draft_data
    python -m src.scraping.nfl_stats
    python -m src.scraping.combine_data
    python -m src.scraping.college_stats
    python -m src.features.build_dataset
    python -m src.features.archetype_classifier
    python -m src.modeling.fantasy_predictor
    python -m src.scraping.prospects_2026
    python -m src.modeling.predict_prospects
    python -m src.modeling.validate_2025

### 3. Launch the dashboard

    streamlit run app/streamlit_app.py

---

## 🏷️ Archetypes

| Position | Archetypes |
|----------|-----------|
| QB | Pocket Passer, Dual Threat |
| RB | Power Back, Speed Back, Receiving Back, 3-Down Back |
| WR | Alpha/X, Slot, Deep Threat, Possession |
| TE | Receiving TE, Hybrid TE, Blocking TE |

---

## ⚠️ Known Limitations

### Archetype Classification Defaults
When combine data (weight, 40-yard dash) is missing for a player, the archetype classifier falls back to defaults:
- **RB** defaults to "Speed Back" if weight is unavailable and under 220 lbs threshold can't be evaluated
- **WR** defaults to "Possession" if height data is missing
- **TE** defaults to "Blocking TE" if reception data is below threshold

This affects some late-round picks and players who did not participate in the NFL Combine. For example, Ollie Gordon (2025) was classified as "Speed Back" despite being a 213 lb power runner because his combine data was not available in the dataset.

### College Stats
The nfl_data_py package does not provide college statistics. For the 2025 prospect class, college career stats were manually sourced from Sports Reference for the top ~50 prospects. Players without manual college stats may have less accurate predictions.

### Data Availability
- 2025 NFL season stats are not yet available in nfl_data_py, so the training data covers 2015-2024 seasons
- The model is trained on historical NFL career stats from draft picks data, which may differ from true college production stats for some players

---

## 🏈 PPR Scoring

| Stat | Points |
|------|--------|
| Passing Yard | 0.04 |
| Passing TD | 4 |
| Interception | -2 |
| Rushing Yard | 0.1 |
| Rushing TD | 6 |
| Reception | 1 |
| Receiving Yard | 0.1 |
| Receiving TD | 6 |
| Fumble Lost | -2 |

---

## 🛣️ Roadmap

- Update with 2026 NFL Draft prospect class
- Add 2025 NFL season data to training set once available in nfl_data_py
- Scrape college stats programmatically for future classes
- Add advanced metrics (PFF grades, EPA) if data becomes available
- Improve archetype classification with more data points
- Add dynasty league trade value calculator
- Deploy Streamlit app to cloud

---

## ⚙️ Tech Stack

- Python 3.9+
- pandas — Data manipulation
- scikit-learn — ML models
- nfl_data_py — NFL data source
- Streamlit — Dashboard
- Plotly — Interactive charts

---

## 🤝 Contributing

PRs welcome. Open an issue first for major changes.
EOF