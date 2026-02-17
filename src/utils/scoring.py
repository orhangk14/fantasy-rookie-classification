from src.utils.config import PPR_SCORING


def calculate_ppr_points(stats: dict) -> float:
    """Calculate PPR fantasy points from a stats dictionary."""
    total = 0.0
    for stat, value in stats.items():
        if stat in PPR_SCORING:
            total += value * PPR_SCORING[stat]
    return round(total, 2)


def tier_label(ppg: float, position: str) -> str:
    """Assign fantasy tier based on PPG and position."""
    thresholds = {
        "QB": [(20, "Elite"), (15, "Starter"), (10, "Bench"), (0, "Bust")],
        "RB": [(15, "Elite"), (10, "Starter"), (6, "Bench"), (0, "Bust")],
        "WR": [(15, "Elite"), (10, "Starter"), (6, "Bench"), (0, "Bust")],
        "TE": [(12, "Elite"), (8, "Starter"), (5, "Bench"), (0, "Bust")],
    }
    for threshold, label in thresholds.get(position, []):
        if ppg >= threshold:
            return label
    return "Bust"