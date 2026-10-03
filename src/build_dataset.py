"""Build the transfer-level modelling table: one row per paid transfer.

Run from the project root:  python src/build_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from features import END, build_features, load_tables

OUT = Path("data/processed")
START = "2014-01-01"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = load_tables()
    t = T.transfers
    s = t[(t.transfer_fee > 0) & (t.transfer_date >= START)].copy()
    s["tid"] = np.arange(len(s))
    log = [("paid transfers %s to %s" % (START, END), len(s))]

    keep = ["tid", "player_id", "player_name", "transfer_date", "from_club_id", "from_club_name", "to_club_id",
            "to_club_name", "transfer_fee", "loan_back", "is_loan_fee"]
    s = build_features(s[keep], T)
    s["log_fee"] = np.log(s.transfer_fee)

    for rule, ok in [("drop suspected loan fees", lambda x: ~x.is_loan_fee),
                     ("drop missing date of birth / position", lambda x: x.age.notna() & x.position.notna()),
                     ("drop fee below 50k", lambda x: x.transfer_fee >= 50_000)]:
        s = s[ok(s)]
        log.append((rule, len(s)))
    log.append(("  of which with pre-transfer valuation", int(s.mv_pre.notna().sum())))

    s.sort_values("transfer_date").to_parquet(OUT / "transfers_model.parquet", index=False)
    log = pd.DataFrame(log, columns=["step", "n"])
    log.to_csv(OUT / "sample_construction.csv", index=False)
    print(log.to_string(index=False))
    print("unknown country: seller %.1f%%, buyer %.1f%%" % (100 * (s.sell_country == "Unknown").mean(),
                                                           100 * (s.buy_country == "Unknown").mean()))


if __name__ == "__main__":
    main()
