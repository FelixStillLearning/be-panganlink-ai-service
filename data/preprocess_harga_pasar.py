from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
DB_DIR = BASE_DIR.parent / "database"
OUTPUT_SQL = DB_DIR / "seed_harga.sql"

FILES = {
    "komoditas_beras_2022_2026.csv": 1,
    "komoditas_cabai_merah_2022_2026.csv": 2,
    "komoditas_bawang_merah_2022_2026.csv": 3,
}


def preprocess_file(csv_filename: str, komoditas_id: int) -> pd.DataFrame:
    csv_path = RAW_DIR / csv_filename
    df = pd.read_csv(csv_path)

    required_columns = {"Date_Param", "Price"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{csv_filename} missing columns: {sorted(missing)}")

    df = df[["Date_Param", "Price"]].copy()
    df["Date_Param"] = pd.to_datetime(df["Date_Param"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(subset=["Date_Param", "Price"])

    df_avg = (
        df.groupby("Date_Param", as_index=False)["Price"]
        .mean()
        .rename(columns={"Date_Param": "tanggal", "Price": "harga"})
    )

    df_avg["komoditas_id"] = komoditas_id
    df_avg["wilayah"] = "Nasional"
    df_avg["harga"] = df_avg["harga"].round(0).astype(int)
    df_avg["tanggal"] = df_avg["tanggal"].dt.strftime("%Y-%m-%d")

    return df_avg[["komoditas_id", "harga", "wilayah", "tanggal"]]


def main() -> None:
    all_rows = []

    for filename, komoditas_id in FILES.items():
        if not (RAW_DIR / filename).exists():
            raise FileNotFoundError(f"File not found: {RAW_DIR / filename}")
        all_rows.append(preprocess_file(filename, komoditas_id))

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df = final_df.sort_values(["komoditas_id", "tanggal"]).reset_index(drop=True)

    DB_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_SQL, "w", encoding="utf-8") as f:
        f.write("INSERT INTO harga_pasar (komoditas_id, harga, wilayah, tanggal) VALUES\n")

        values = []
        for _, row in final_df.iterrows():
            values.append(
                f"  ({row['komoditas_id']}, {row['harga']}, '{row['wilayah']}', '{row['tanggal']}')"
            )

        f.write(",\n".join(values))
        f.write(
            "\nON DUPLICATE KEY UPDATE "
            "harga = VALUES(harga), "
            "wilayah = VALUES(wilayah);"
        )

    print(f"Done. Generated {len(final_df)} rows into {OUTPUT_SQL}")


if __name__ == "__main__":
    main()