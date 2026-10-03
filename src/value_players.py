"""Value every active player on 1 July of each of the last five years (Model V, walk-forward).

For each date, the model is trained only on transfers that happened before it, so a past
valuation is what the model would have said at the time. Model V uses no buyer information:
it estimates the fee the player would fetch if sold from his current club.
Run from the project root (after train_model.py):  python src/value_players.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features, load_tables
from modeling import FEATS_V, OUT, conformal_q, fit_lgb, prepare

DATES = pd.to_datetime([f"{y}-07-01" for y in range(2022, 2027)])
ACTIVE_DAYS = 548      # a player is active if Transfermarkt valued him in the 18 months before the date
COVERAGE = 0.8
SITE = Path("docs")


def candidates(T, date, offset):
    v = T.val[(T.val.date <= date) & (T.val.date > date - pd.Timedelta(days=ACTIVE_DAYS))].sort_values("date")
    tm = v.groupby("player_id").market_value_in_eur.last()
    q = pd.DataFrame({"player_id": tm.index, "tm_value": tm.values})
    q["tid"] = offset + np.arange(len(q))
    q["transfer_date"], q["from_club_id"], q["to_club_id"] = date, np.nan, np.nan
    return q


def main():
    T = load_tables()
    tr = pd.read_parquet("data/processed/transfers_model.parquet")
    parts = [build_features(candidates(T, d, 10_000_000 * (i + 1)), T) for i, d in enumerate(DATES)]
    allrows = prepare(pd.concat([tr.assign(is_transfer=True)] + parts, ignore_index=True))
    allrows["log_ratio"] = allrows.log_fee - allrows.log_mv_pre
    tr = allrows[(allrows.is_transfer == True) & allrows.log_mv_pre.notna()]  # noqa: E712
    pl = allrows[(allrows.is_transfer != True) & allrows.log_mv_pre.notna() & allrows.age.notna()  # noqa: E712
                 & (allrows.position != "NA")].copy()

    rounds = fit_lgb(tr[tr.split == "train"], tr[tr.split == "valid"], FEATS_V, "log_ratio").best_iteration
    out = []
    for d in DATES:
        past = tr[tr.transfer_date < d]
        cal = past[past.transfer_date >= d - pd.Timedelta(days=365)]
        m = fit_lgb(past[past.transfer_date < d - pd.Timedelta(days=365)], None, FEATS_V, "log_ratio", rounds)
        q = conformal_q((cal.log_ratio - m.predict(cal[FEATS_V])).abs(), COVERAGE)
        m = fit_lgb(past, None, FEATS_V, "log_ratio", rounds)
        p = pl[pl.transfer_date == d].copy()
        p["pred_log"] = m.predict(p[FEATS_V]) + p.log_mv_pre
        p["model_value"], p["model_lo"], p["model_hi"] = np.exp(p.pred_log), np.exp(p.pred_log - q), np.exp(p.pred_log + q)
        out.append(p)
        print(d.date(), "trained on", len(past), "transfers | players valued:", len(p),
              "| 80%% interval: x/÷ %.2f" % np.exp(q))
    out = pd.concat(out)
    out["club"] = out.from_club_id.map(T.names)
    cols = ["player_id", "name", "transfer_date", "club", "sell_league", "sell_country", "sub_position", "position",
            "age", "tm_value", "model_value", "model_lo", "model_hi"]
    out[cols].rename(columns={"transfer_date": "date"}).to_parquet(OUT / "player_values.parquet", index=False)

    # compact file for the website: players valued at the latest date, values in thousands of euros
    k = lambda s: (s / 1000).round().astype("Int64")  # noqa: E731
    wide = out.pivot(index="player_id", columns="transfer_date", values=["tm_value", "model_value"])
    last = out[out.transfer_date == DATES[-1]].set_index("player_id")
    leagues = T.comps.set_index("competition_id").name.str.replace("-", " ").str.title().to_dict()
    rows = []
    for pid, r in last.iterrows():
        series = lambda c: [None if pd.isna(x) else int(round(x / 1000)) for x in wide.loc[pid, c]]  # noqa: E731
        rows.append([int(pid), r["name"], r.club if isinstance(r.club, str) else "",
                     leagues.get(str(r.sell_league), ""), str(r.sell_country), str(r.sub_position) if
                     r.sub_position != "NA" else str(r.position), round(float(r.age), 1),
                     series("tm_value"), series("model_value"), int(k(pd.Series([r.model_lo]))[0]),
                     int(k(pd.Series([r.model_hi]))[0])])
    data = {"dates": [d.strftime("%Y-%m-%d") for d in DATES], "coverage": COVERAGE,
            "fields": ["id", "name", "club", "league", "country", "position", "age", "tm", "model", "lo", "hi"],
            "players": rows}
    SITE.mkdir(exist_ok=True)
    (SITE / "data.js").write_text("window.DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";",
                                  encoding="utf-8")
    print("site data:", len(rows), "players,", round((SITE / "data.js").stat().st_size / 1e6, 1), "MB")
    top = last.sort_values("model_value", ascending=False).head(15)
    print(top[["name", "club", "age", "tm_value", "model_value", "model_lo", "model_hi"]].round(0).to_string())
    r = np.log(last.model_value / last.tm_value)
    print("model vs Transfermarkt, log ratio quantiles:", r.quantile([.05, .25, .5, .75, .95]).round(2).to_dict())


if __name__ == "__main__":
    main()
