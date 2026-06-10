from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def evaluate_df(
    eval_df: pd.DataFrame,
    y_true_col: str = "y",
    y_pred_col: str = "yhat",
) -> Dict[str, float]:
    if eval_df.empty:
        return {"mae": float("nan"), "mape": float("nan"), "r2": float("nan")}

    y_true = eval_df[y_true_col].to_numpy(dtype=float)
    y_pred = eval_df[y_pred_col].to_numpy(dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mape = _safe_mape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    return {"mae": mae, "mape": mape, "r2": r2}


def plot_actual_vs_pred(eval_df: pd.DataFrame, out_path_prefix: str) -> None:
    if eval_df.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(eval_df["y"], eval_df["yhat"], alpha=0.6)
    mx = max(float(eval_df["y"].max()), float(eval_df["yhat"].max()))
    plt.plot([0, mx], [0, mx], "r--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_scatter.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(eval_df["ds"], eval_df["y"], label="actual")
    plt.plot(eval_df["ds"], eval_df["yhat"], label="predicted")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Harga")
    plt.title("Actual vs Predicted (time series)")
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_line.png")
    plt.close()