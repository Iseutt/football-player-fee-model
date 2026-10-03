"""Download the raw data into data/raw (not stored in the repository).

Sources:
  - transfermarkt-datasets by dcaribou (CC0): players, valuations, transfers, appearances, games
  - football-transfers-data by d2ski: used only for the country of clubs outside the covered leagues
Run from the project root:  python src/download_data.py
"""
from pathlib import Path
from urllib.request import urlretrieve

RAW = Path("data/raw")
TM = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
TABLES = ["transfers", "players", "player_valuations", "clubs", "competitions", "appearances", "games"]
D2SKI = "https://raw.githubusercontent.com/d2ski/football-transfers-data/main/dataset/transfers.csv"

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        print("downloading", t)
        urlretrieve(TM + t + ".csv.gz", RAW / f"{t}.csv.gz")
    print("downloading d2ski transfers")
    urlretrieve(D2SKI, RAW / "d2ski_transfers.csv")
