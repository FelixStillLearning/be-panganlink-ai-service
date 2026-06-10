from typing import Tuple

import numpy as np
import pandas as pd


def remove_outliers_iqr(df: pd.DataFrame, col: str = "y", multiplier: float = 1.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove outlier menggunakan Interquartile Range (IQR).
    
    Args:
        df: DataFrame dengan kolom 'ds' dan 'y'
        col: kolom yang dicheck outlier-nya (default 'y')
        multiplier: IQR multiplier (1.5 = standard, 3.0 = lebih ketat)
    
    Returns:
        Tuple: (df_clean, df_outliers)
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    mask_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    df_clean = df[~mask_outlier].copy()
    df_outliers = df[mask_outlier].copy()

    return df_clean, df_outliers


def remove_outliers_zscore(df: pd.DataFrame, col: str = "y", threshold: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove outlier menggunakan Z-score.
    
    Args:
        df: DataFrame dengan kolom 'ds' dan 'y'
        col: kolom yang dicheck outlier-nya
        threshold: Z-score threshold (3.0 = standard, 2.5 = lebih ketat)
    
    Returns:
        Tuple: (df_clean, df_outliers)
    """
    mean = df[col].mean()
    std = df[col].std()
    
    z_scores = np.abs((df[col] - mean) / std)
    mask_outlier = z_scores > threshold
    
    df_clean = df[~mask_outlier].copy()
    df_outliers = df[mask_outlier].copy()
    
    return df_clean, df_outliers


def preprocess_data(df: pd.DataFrame, method: str = "iqr", verbose: bool = True) -> pd.DataFrame:
    """
    Main preprocessing function.
    
    Args:
        df: DataFrame dengan kolom 'ds' dan 'y'
        method: 'iqr' atau 'zscore'
        verbose: print info tentang outlier yang dihapus
    
    Returns:
        df_clean: DataFrame yang sudah dibersihkan
    """
    if method == "iqr":
        df_clean, df_outliers = remove_outliers_iqr(df, col="y", multiplier=1.5)
    elif method == "zscore":
        df_clean, df_outliers = remove_outliers_zscore(df, col="y", threshold=3.0)
    else:
        raise ValueError(f"method harus 'iqr' atau 'zscore', dapat {method}")
    
    if verbose and len(df_outliers) > 0:
        print(f"  Outliers dihapus ({method}): {len(df_outliers)} rows")
        for _, row in df_outliers.iterrows():
            print(f"    - {row['ds']}: {row['y']:.0f}")
    
    return df_clean.reset_index(drop=True)