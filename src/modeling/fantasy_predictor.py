import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils.config import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.modeling.comparable_finder import height_to_inches


FEATURE_COLS = {
    "QB": [
        "round", "pick", "age", "ht_inches", "wt", "forty",
        "college_pass_att", "college_pass_yd", "college_pass_td", "college_int",
        "college_rush_att", "college_rush_yd", "college_rush_td"
    ],
    "RB": [
        "round", "pick", "age", "ht_inches", "wt", "forty",
        "college_rush_att", "college_rush_yd", "college_rush_td",
        "college_rec", "college_rec_yd", "college_rec_td"
    ],
    "WR": [
        "round", "pick", "age", "ht_inches", "wt", "forty",
        "college_rec", "college_rec_yd", "college_rec_td",
        "college_rush_att", "college_rush_yd"
    ],
    "TE": [
        "round", "pick", "age", "ht_inches", "wt", "forty",
        "college_rec", "college_rec_yd", "college_rec_td"
    ]
}


def load_and_prepare() -> pd.DataFrame:
    """Load dataset and prepare features."""
    path = DATA_PROCESSED / "rookie_dataset.csv"
    df = pd.read_csv(path)
    df["ht_inches"] = df["ht"].apply(height_to_inches)
    return df


def get_usable_features(X: pd.DataFrame, features: list) -> list:
    """Only keep features that have enough non-NaN values."""
    usable = []
    for f in features:
        if f in X.columns and X[f].notna().sum() > 5:
            usable.append(f)
    return usable


def train_ppg_model(df: pd.DataFrame, pos: str):
    """Train a PPG regression model for a position."""
    subset = df[df["pos"] == pos].copy()

    features = FEATURE_COLS[pos]
    available = [f for f in features if f in subset.columns]

    subset = subset.dropna(subset=["ppg"])

    X = subset[available].copy()
    y = subset["ppg"]

    usable = get_usable_features(X, available)
    X = X[usable]

    if len(usable) < 2:
        print(f"  Not enough usable features for {pos}")
        return None, None, None, None

    medians = X.median()
    X = X.fillna(medians)

    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) < 10:
        print(f"  Not enough data for {pos} ({len(X)} rows)")
        return None, None, None, None

    print(f"  Using features: {usable}")
    print(f"  Training samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # simplified model to prevent overfitting
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        min_samples_leaf=10,
        min_samples_split=15,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)

    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)

    print(f"  {pos} PPG Model:")
    print(f"    Train R²: {train_score:.3f}")
    print(f"    Test R²:  {test_score:.3f}")
    print(f"    CV R²:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    importance = pd.Series(model.feature_importances_, index=usable)
    importance = importance.sort_values(ascending=False)
    print(f"    Top features: {dict(importance.head(5).round(3))}")

    return model, scaler, medians, usable


def train_tier_model(df: pd.DataFrame, pos: str):
    """Train a tier classification model for a position."""
    subset = df[df["pos"] == pos].copy()

    features = FEATURE_COLS[pos]
    available = [f for f in features if f in subset.columns]

    subset = subset.dropna(subset=["tier"])

    X = subset[available].copy()
    y = subset["tier"]

    usable = get_usable_features(X, available)
    X = X[usable]

    if len(usable) < 2:
        print(f"  Not enough usable features for {pos}")
        return None, None, None, None, None

    medians = X.median()
    X = X.fillna(medians)

    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) < 10:
        print(f"  Not enough data for {pos} ({len(X)} rows)")
        return None, None, None, None, None

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # simplified model to prevent overfitting
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        min_samples_leaf=10,
        min_samples_split=15,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)

    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    print(f"  {pos} Tier Model:")
    print(f"    Train Acc: {train_score:.3f}")
    print(f"    Test Acc:  {test_score:.3f}")

    return model, scaler, medians, usable, le


def predict_player(player_data: dict, pos: str) -> dict:
    """Predict PPG and tier for a single player."""
    ppg_bundle = joblib.load(MODELS_DIR / f"{pos.lower()}_ppg_model.joblib")
    tier_bundle = joblib.load(MODELS_DIR / f"{pos.lower()}_tier_model.joblib")

    ppg_model = ppg_bundle["model"]
    ppg_scaler = ppg_bundle["scaler"]
    ppg_medians = ppg_bundle["medians"]
    ppg_features = ppg_bundle["features"]

    tier_model = tier_bundle["model"]
    tier_scaler = tier_bundle["scaler"]
    tier_medians = tier_bundle["medians"]
    tier_features = tier_bundle["features"]
    tier_le = tier_bundle["label_encoder"]

    ppg_X = pd.DataFrame([player_data])[ppg_features].fillna(ppg_medians)
    tier_X = pd.DataFrame([player_data])[tier_features].fillna(tier_medians)

    ppg_X_scaled = ppg_scaler.transform(ppg_X)
    tier_X_scaled = tier_scaler.transform(tier_X)

    ppg_pred = ppg_model.predict(ppg_X_scaled)[0]
    tier_probs = tier_model.predict_proba(tier_X_scaled)[0]
    tier_pred = tier_le.inverse_transform([tier_model.predict(tier_X_scaled)[0]])[0]

    tier_prob_dict = dict(zip(tier_le.classes_, tier_probs.round(3)))

    return {
        "predicted_ppg": round(ppg_pred, 2),
        "predicted_tier": tier_pred,
        "tier_probabilities": tier_prob_dict
    }


def run():
    """Train and save all models."""
    df = load_and_prepare()

    print("Training models...\n")

    for pos in ["QB", "RB", "WR", "TE"]:
        print(f"\n{'='*50}")
        print(f"Position: {pos} ({len(df[df['pos'] == pos])} players)")
        print(f"{'='*50}")

        ppg_result = train_ppg_model(df, pos)
        if ppg_result[0] is not None:
            model, scaler, medians, features = ppg_result
            joblib.dump({
                "model": model,
                "scaler": scaler,
                "medians": medians,
                "features": features
            }, MODELS_DIR / f"{pos.lower()}_ppg_model.joblib")

        print()

        tier_result = train_tier_model(df, pos)
        if tier_result[0] is not None:
            model, scaler, medians, features, le = tier_result
            joblib.dump({
                "model": model,
                "scaler": scaler,
                "medians": medians,
                "features": features,
                "label_encoder": le
            }, MODELS_DIR / f"{pos.lower()}_tier_model.joblib")

    print("\n\nAll models saved to models/")


if __name__ == "__main__":
    run()