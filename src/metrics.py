import numpy as np


def mape_percent(y_true, y_pred, eps: float = 1e-8) -> float:
    """Mean absolute percentage error in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)
