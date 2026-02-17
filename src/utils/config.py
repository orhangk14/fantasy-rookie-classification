import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_FEATURES = ROOT_DIR / "data" / "features"
MODELS_DIR = ROOT_DIR / "models"

# ── Scraping ───────────────────────────────────────────
PFR_BASE = "https://www.pro-football-reference.com"
CFB_BASE = "https://www.sports-reference.com/cfb"
REQUEST_DELAY = 5
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# ── Draft Classes ──────────────────────────────────────
YEARS = list(range(2015, 2026))
POSITIONS = ["QB", "RB", "WR", "TE"]

# ── Archetypes ─────────────────────────────────────────
ARCHETYPES = {
    "QB": ["Pocket Passer", "Dual Threat"],
    "RB": ["Power Back", "Speed Back", "Receiving Back", "3-Down Back"],
    "WR": ["Alpha/X", "Slot", "Deep Threat", "Possession"],
    "TE": ["Receiving TE", "Hybrid TE", "Blocking TE"]
}

# ── PPR Scoring ────────────────────────────────────────
PPR_SCORING = {
    "pass_yd": 0.04,
    "pass_td": 4,
    "interception": -2,
    "rush_yd": 0.1,
    "rush_td": 6,
    "reception": 1,
    "rec_yd": 0.1,
    "rec_td": 6,
    "fumble_lost": -2
}

# ── Model ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2