import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import datetime

from ml.utils import COMMODITIES, MODEL_DIR, fetch_historical_data

def create_features(df: pd.DataFrame):
    df = df.copy()
    # Ensure ds is sorted
    df = df.sort_values("ds").reset_index(drop=True)
    
    # Lag features (menggunakan harga hari-hari sebelumnya)
    for i in range(1, 8):
        df[f"lag_{i}"] = df["y"].shift(i)
        
    # Rolling features
    df["rolling_mean_7"] = df["y"].shift(1).rolling(window=7).mean()
    df["rolling_std_7"] = df["y"].shift(1).rolling(window=7).std()
    
    # Time features
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    
    # Drop NaNs akibat pergeseran lag
    df = df.dropna().reset_index(drop=True)
    return df

def time_split(df, test_ratio=0.15):
    n = len(df)
    test_size = int(n * test_ratio)
    train_size = n - test_size
    return df.iloc[:train_size].copy(), df.iloc[train_size:].copy()

def main():
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODEL_DIR / f"run_huber_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for kid, name in COMMODITIES.items():
        print(f"\n--- Training Huber Regression untuk {name} ---")
        df_raw = fetch_historical_data(kid)
        if len(df_raw) < 50:
            print(f"Data terlalu sedikit untuk {name}.")
            continue
            
        # Membuat fitur lags
        df = create_features(df_raw)
        
        train_df, test_df = time_split(df, test_ratio=0.15)
        
        features = [col for col in df.columns if col not in ["ds", "y"]]
        
        X_train, y_train = train_df[features], train_df["y"]
        X_test, y_test = test_df[features], test_df["y"]
        
        # Pipeline: Scaling + HuberRegressor
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("huber", HuberRegressor(epsilon=1.35, max_iter=2000))
        ])
        
        model.fit(X_train, y_train)
        
        # Prediksi test set (One-step-ahead menggunakan actual lag)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        print(f"[{name}] MAE: {mae:.2f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")
        
        # Simpan grafik (Full Time Series, Zoomed Test Series, Scatter Plot)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        
        # 1. Line Plot (Full Time Series)
        ax1.plot(train_df["ds"], train_df["y"], label="Train Actual", color="blue", alpha=0.6)
        ax1.plot(test_df["ds"], test_df["y"], label="Test Actual", color="green", alpha=0.6)
        ax1.plot(test_df["ds"], y_pred, label="Test Predicted", color="red", alpha=0.8, linestyle="--")
        ax1.set_title(f"Full Time Series - {name}")
        ax1.set_xlabel("Tanggal")
        ax1.set_ylabel("Harga")
        ax1.legend()
        
        # 2. Line Plot (Zoomed-in Test Set Only)
        ax2.plot(test_df["ds"], test_df["y"], label="Actual (Test)", color="green", alpha=0.8)
        ax2.plot(test_df["ds"], y_pred, label="Predicted (Test)", color="red", alpha=0.8, linestyle="--")
        ax2.set_title(f"Test Set Only: Actual vs Predicted - {name}")
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("Harga")
        ax2.legend()
        
        # 3. Scatter Plot (Actual vs Predicted)
        ax3.scatter(y_test, y_pred, color="purple", alpha=0.6, edgecolors="white")
        
        # Garis y=x (Perfect Fit Line)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", label="Perfect Fit (y=x)")
        
        ax3.set_title(f"Scatter Plot - {name}")
        ax3.set_xlabel("Actual Price")
        ax3.set_ylabel("Predicted Price")
        ax3.legend()
        
        plt.tight_layout()
        plot_path = run_dir / f"huber_{name}_fit.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Grafik tersimpan di: {plot_path}")
        
        # Simpan Model
        komoditas_dir = run_dir / name
        komoditas_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, komoditas_dir / f"huber_{name}.joblib")
        
        results.append({
            "name": name,
            "mae": mae,
            "mape": mape,
            "r2": r2
        })
        
    print("\n" + "=" * 50)
    print("SUMMARY HUBER REGRESSION")
    print("=" * 50)
    for r in results:
        print(f"{r['name']}: MAPE={r['mape']:.2f}%, R2={r['r2']:.4f}")

if __name__ == "__main__":
    main()
