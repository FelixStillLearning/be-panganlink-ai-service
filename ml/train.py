from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from prophet import Prophet

from ml.eval import evaluate_df, plot_actual_vs_pred
from ml.preprocess import preprocess_data
from ml.split import time_train_val_test_split
from ml.utils import COMMODITIES, MODEL_DIR, fetch_historical_data

GRID = [
    {"changepoint_prior_scale": 0.01, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.05, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.1, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.05, "seasonality_mode": "multiplicative"},
]

# Per komoditas, tentukan apakah perlu preprocessing
PREPROCESSING_CONFIG = {
    1: None,                    # beras: tidak perlu (sudah stabil)
    2: {"method": "iqr", "enabled": True},      # cabai merah: volatile
    3: {"method": "iqr", "enabled": True},      # bawang merah: volatile
}


def build_model(params: Dict) -> Prophet:
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
    )


def select_best_params(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
    best_params: Optional[Dict] = None
    best_mape = float("inf")

    for params in GRID:
        model = build_model(params)
        model.fit(train_df)

        pred_val = model.predict(val_df[["ds"]])
        eval_val = val_df.merge(pred_val[["ds", "yhat"]], on="ds", how="inner")
        metrics_val = evaluate_df(eval_val)

        mape = metrics_val["mape"]
        if pd.notna(mape) and mape < best_mape:
            best_mape = mape
            best_params = params

    if best_params is None:
        best_params = GRID[0]
    return best_params


def train_one(komoditas_id: int, val_ratio: float = 0.15, test_ratio: float = 0.15):
    name = COMMODITIES[komoditas_id]
    df = fetch_historical_data(komoditas_id)

    if len(df) < 50:
        print(f"[SKIP] {name}: data terlalu sedikit ({len(df)} rows).")
        return None

    # PREPROCESSING: remove outlier jika config-nya enable
    config = PREPROCESSING_CONFIG.get(komoditas_id)
    if config and config.get("enabled"):
        print(f"[{name}] preprocessing dengan method={config['method']}...")
        df = preprocess_data(df, method=config["method"], verbose=True)
        if len(df) < 50:
            print(f"[SKIP] {name}: data terlalu sedikit setelah preprocessing ({len(df)} rows).")
            return None

    train_df, val_df, test_df = time_train_val_test_split(
        df=df,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_train_size=30,
        min_val_size=7,
        min_test_size=7,
    )

    best_params = select_best_params(train_df, val_df)
    print(f"[{name}] best params: {best_params}")

    final_train = (
        pd.concat([train_df, val_df], ignore_index=True)
        .sort_values("ds")
        .reset_index(drop=True)
    )
    final_model = build_model(best_params)
    final_model.fit(final_train)

    pred_test = final_model.predict(test_df[["ds"]])
    eval_test = test_df.merge(pred_test[["ds", "yhat"]], on="ds", how="inner")
    metrics_test = evaluate_df(eval_test)

    model_path = MODEL_DIR / f"prophet_{name}.joblib"
    joblib.dump(final_model, model_path)

    plot_prefix = str((MODEL_DIR / f"prophet_{name}_test").resolve())
    plot_actual_vs_pred(eval_test, plot_prefix)

    print(
        f"[{name}] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"MAE={metrics_test['mae']:.2f} MAPE={metrics_test['mape']:.2f}% R2={metrics_test['r2']:.4f}"
    )
    print(f"[{name}] saved model -> {model_path}\n")

    row = {
        "komoditas_id": komoditas_id,
        "name": name,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "mae_test": metrics_test["mae"],
        "mape_test": metrics_test["mape"],
        "r2_test": metrics_test["r2"],
        "best_params": str(best_params),
        "model_path": str(model_path),
        "preprocessing": "iqr" if (config and config.get("enabled")) else "none",
    }

    metrics_path = MODEL_DIR / "metrics_summary.csv"
    if metrics_path.exists():
        old_df = pd.read_csv(metrics_path)
        old_df = old_df[old_df["komoditas_id"] != komoditas_id]
        new_df = pd.concat([old_df, pd.DataFrame([row])], ignore_index=True)
    else:
        new_df = pd.DataFrame([row])
    new_df.to_csv(metrics_path, index=False)

    return row


def main():
    results = []
    for kid in COMMODITIES:
        res = train_one(kid, val_ratio=0.15, test_ratio=0.15)
        if res:
            results.append(res)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        preprocess_str = f" [preprocess: {r['preprocessing']}]"
        print(
            f"{r['name']}: MAE={r['mae_test']:.2f}, "
            f"MAPE={r['mape_test']:.2f}%, R2={r['r2_test']:.4f}{preprocess_str}"
        )


if __name__ == "__main__":
    main()