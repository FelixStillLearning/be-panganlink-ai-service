from typing import Tuple

import pandas as pd


def time_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_train_size: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio harus di antara 0 dan 1.")

    df = df.sort_values("ds").reset_index(drop=True)
    n = len(df)

    if n < min_train_size + 1:
        raise ValueError(
            f"Data terlalu sedikit ({n} rows), min_train_size={min_train_size}."
        )

    split = int(n * (1 - test_ratio))
    split = max(split, min_train_size)

    if split >= n:
        raise ValueError("Split tidak valid, test set kosong.")

    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    return train, test


def time_train_val_test_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_train_size: int = 30,
    min_val_size: int = 7,
    min_test_size: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio/test_ratio tidak valid. Pastikan val+test < 1.")

    df = df.sort_values("ds").reset_index(drop=True)
    n = len(df)

    required = min_train_size + min_val_size + min_test_size
    if n < required:
        raise ValueError(f"Data terlalu sedikit ({n} rows), butuh minimal {required}.")

    train_end = int(n * (1 - (val_ratio + test_ratio)))
    val_end = train_end + int(n * val_ratio)

    train_end = max(train_end, min_train_size)
    if (val_end - train_end) < min_val_size:
        val_end = train_end + min_val_size
    if (n - val_end) < min_test_size:
        val_end = n - min_test_size

    if train_end < min_train_size or (val_end - train_end) < min_val_size:
        raise ValueError("Tidak bisa memenuhi minimum size train/val/test.")

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test