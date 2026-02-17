cat > README.md << 'EOF'
# 🏈 Fantasy Rookie Draft Prediction Model

A machine learning-powered tool that predicts NFL rookie fantasy performance (PPR scoring) based on college production, draft capital, and physical measurables. Includes player archetype classification, comparable player finder, and an interactive Streamlit dashboard.

> **⚠️ Note:** This project currently uses 2025 NFL Draft prospects as a proof of concept. It will be updated with the actual 2026 draft class once the draft takes place and prospect data becomes available.

---

## 📸 Features

- **Fantasy Predictions** — Predicted PPG and tier classification (Elite, Starter, Bench, Bust) for each prospect
- **Archetype Classification** — Players categorized by playstyle (e.g., Dual Threat QB, 3-Down Back, Slot WR)
- **Comparable Player Finder** — Cosine similarity engine that finds the most similar historical rookies
- **Interactive Dashboard** — Streamlit app with prospect explorer, historical data filters, and custom prospect evaluator
- **PPR Scoring** — Full PPR fantasy scoring system with configurable weights

---

## 🏗️ Architecture

Pipeline: Pull Data → Clean/Merge → Feature Engineer → Classify Archetypes → Train Models → Predict Prospects → Streamlit

Data Sources (all free): nfl_data_py for draft picks, NFL seasonal stats, and combine data

Models: Gradient Boosting Regressor (PPG), Gradient Boosting Classifier (Tier), Cosine Similarity (Comparables)

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
    │   │   └── predict_prospects.py
    │   └── utils/
    │       ├── config.py
    │       └── scoring.py
    ├── .env
    ├── .gitignore
    ├── requirements.txt
    └── README.md

---

## 🚀 Quick Start

### 1. Clone the repo

    git clone https://github.com/YOUR_USERNAME/fantasy-rookie-classification.git
    cd fantasy-rookie-classification

### 2. Set up virtual environment

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### 3. Run the full pipeline

    python -m src.scraping.draft_data
    python -m src.scraping.nfl_stats
    python -m src.scraping.combine_data
    python -m src.scraping.college_stats
    python -m src.features.build_dataset
    python -m src.features.archetype_classifier
    python -m src.modeling.fantasy_predictor
    python -m src.scraping.prospects_2026
    python -m src.modeling.predict_prospects

### 4. Launch the dashboard

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

## 📊 Model Performance

| Position | Samples | PPG CV R² | Tier Test Acc | Top Features |
|----------|---------|-----------|---------------|-------------|
| QB | 73 | 0.439 | 0.600 | college_pass_yd, college_int, pick |
| RB | 174 | 0.546 | 0.657 | college_rush_yd, college_rec, pick |
| WR | 259 | 0.321 | 0.538 | college_rec, college_rec_yd, pick |
| TE | 113 | 0.118 | 0.565 | pick, college_rec, college_rec_td |

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

## 🔮 Sample Predictions

| Player | Position | Archetype | Pred PPG | Tier | Top Comp |
|--------|----------|-----------|----------|------|----------|
| Cam Ward | QB | Dual Threat | 13.81 | Bench | Bo Nix |
| Shedeur Sanders | QB | Pocket Passer | 14.80 | Bench | C.J. Stroud |
| Ashton Jeanty | RB | Speed Back | 12.88 | Starter | J.K. Dobbins |
| Tetairoa McMillan | WR | Deep Threat | 10.58 | Starter | DeVante Parker |
| Luther Burden III | WR | Slot | 11.79 | Starter | Malik Nabers |
| Tyler Warren | TE | Receiving TE | 7.89 | Bench | Pat Freiermuth |

---

## 🛣️ Roadmap

- Update with 2026 NFL Draft class
- Add 2025 NFL season data once available in nfl_data_py
- Scrape college stats per season for year-over-year trends
- Add advanced metrics (PFF grades, EPA, etc.) if data becomes available
- Improve model with ensemble methods and hyperparameter tuning
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

## 📄 License

MIT License. Feel free to use, modify, and distribute.

---

## 🤝 Contributing

PRs welcome. Open an issue first for major changes.
EOF