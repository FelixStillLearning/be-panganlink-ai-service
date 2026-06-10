import pandas as pd
import numpy as np

for name in ["beras", "bawang_merah", "cabai_merah"]:
    df = pd.read_csv(f"data/raw/komoditas_{name}_2022_2026.csv")
    df["ds"] = pd.to_datetime(df["Date_Param"])
    df["y"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    df = df.groupby("ds")["y"].mean().reset_index()
    df = df.sort_values("ds")
    df = df.set_index("ds").resample("D").interpolate(method="linear").reset_index()
    
    y = df["y"]
    print(f"--- {name.upper()} ---")
    print(f"Count: {len(y)}")
    print(f"Min: {y.min():.2f}")
    print(f"Max: {y.max():.2f}")
    print(f"Mean: {y.mean():.2f}")
    print(f"Std: {y.std():.2f}")
    print(f"CV (Std/Mean): {y.std()/y.mean():.4f}")
    
    # Calculate daily percentage change
    pct_change = y.pct_change().dropna()
    print(f"Max daily jump: {pct_change.max()*100:.2f}%")
    print(f"Max daily drop: {pct_change.min()*100:.2f}%")
    print()
