"""Value every active player on 1 July of each of the last five years (Model V, walk-forward).

For each date, the model is trained only on transfers that happened before it, so a past
valuation is what the model would have said at the time. Model V uses no buyer information:
it estimates the fee the player would fetch if sold from his current club.
Run from the project root (after train_model.py):  python src/value_players.py
"""
import base64
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features, load_tables, season_of
from modeling import (CAT, FEATS_A, FEATS_V, OUT, RANGE_EDGES, apply_shrink, band_q, conformal_bands, fit_lgb,
                      fit_ratio_model, predict_ratio, prepare)

DATES = pd.to_datetime([f"{y}-07-01" for y in range(2022, 2027)])
ACTIVE_DAYS = 548      # a player is active if Transfermarkt valued him in the 18 months before the date
COVERAGE = 0.8
DECEASED = [340950]    # players who died but still have a recent valuation in the source (Diogo Jota)
SITE = Path("docs")
BUYER_STEP = 0.02      # buyer prices are stored as log(price / Transfermarkt value) in steps of 2%
BUYER_COLS = ["buy_league", "buy_country", "buy_ppg", "log_buy_spend_3y", "buy_n_3y", "buy_youth_team",
              "log_buy_squad_value", "log_buy_league_avg_squad_value", "buy_club_log_ratio_3y"]


def candidates(T, date, offset):
    v = T.val[(T.val.date <= date) & (T.val.date > date - pd.Timedelta(days=ACTIVE_DAYS))].sort_values("date")
    tm = v.groupby("player_id").market_value_in_eur.last()
    q = pd.DataFrame({"player_id": tm.index, "tm_value": tm.values})
    q["tid"] = offset + np.arange(len(q))
    q["transfer_date"], q["from_club_id"], q["to_club_id"] = date, np.nan, np.nan
    return q


def buyer_clubs(T, date):
    """One row per club of the covered leagues in the last completed season: the possible buyers."""
    cs = T.cs[(T.cs.season == season_of(pd.Series([date]))[0] - 1) & T.cs.league.notna()]
    q = pd.DataFrame({"to_club_id": cs.club_id.values, "player_id": T.val.player_id.iloc[0]})
    q["tid"] = 90_000_000 + np.arange(len(q))
    q["transfer_date"], q["from_club_id"] = date, np.nan
    return build_features(q, T).assign(buyer_row=True)


def usual_values(past, date):
    """Per buying league: the range of Transfermarkt values (5th percentile to maximum, in thousands of euros)
    of the players its clubs bought over the previous five years. Outside it, a buyer price is an extrapolation."""
    r = past[past.transfer_date >= date - pd.Timedelta(days=5 * 365)]
    span = lambda s: [int(s.quantile(0.05) / 1000), int(s.max() / 1000)]  # noqa: E731
    by = {str(k): span(g) for k, g in r.groupby("buy_league", observed=True).mv_pre if len(g) >= 30}
    return by, span(r.mv_pre)


def stamp_site(version):
    """Version the data files in index.html, so that a browser never mixes a new page with cached old data."""
    f = SITE / "index.html"
    f.write_text(re.sub(r"((?:data|buyers)\.js)(\?v=\w*)?", r"\1?v=" + version, f.read_text(encoding="utf-8")),
                 encoding="utf-8", newline="\n")


def buyer_prices(p, buyers, tr, rounds, T, leagues, version):
    """Full model (knows the buyer): price of every player for every possible buying club, for the website."""
    d = DATES[-1]
    past = tr[tr.transfer_date < d]
    cal = past[past.transfer_date >= d - pd.Timedelta(days=365)]
    m, shrink = fit_ratio_model(past, FEATS_A, rounds)
    m_cal = fit_lgb(past[past.transfer_date < d - pd.Timedelta(days=365)], None, FEATS_A, "log_ratio", rounds)
    q = conformal_bands((cal.log_ratio - predict_ratio(m_cal, shrink, cal, FEATS_A)).abs(), cal.mv_pre, COVERAGE)
    to_shown = (p.log_mv_pre - np.log(p.tm_value)).values      # the site multiplies by the latest Transfermarkt value
    out = SITE / "buyers"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    usual, usual_all = usual_values(past, d)
    country = T.comps.set_index("competition_id").country_name.to_dict()
    # countries of the eight richest leagues (average squad value), shown first in the club picker
    top = buyers.groupby("buy_league", observed=True).buy_league_avg_squad_value.first().sort_values(ascending=False)
    top = [country[str(k)] for k in top.index[:8]]
    x, clubs, allc = p[FEATS_A].copy(), [], []
    for b in buyers.itertuples():
        for c in BUYER_COLS:
            v = getattr(b, c)
            x[c] = pd.Categorical([v] * len(x), categories=p[c].cat.categories) if c in CAT else v
        x["same_league"] = ((p.sell_league.astype(str) == b.buy_league) & (b.buy_league != "OTHER")).astype(int).values
        x["same_country"] = ((p.sell_country.astype(str) == b.buy_country) & (b.buy_country != "Unknown")).astype(int).values
        c = apply_shrink(m.predict(x), p.log_mv_pre.values, shrink) + to_shown
        c = np.clip(np.round(c / BUYER_STEP), -127, 127).astype(np.int8)
        c[(p.from_club_id == b.to_club_id).values] = -128           # the player is already at this club
        cid = int(b.to_club_id)
        (out / f"{cid}.js").write_text("window.BUYER_DATA=window.BUYER_DATA||{};window.BUYER_DATA[%d]=\"%s\";"
                                       % (cid, base64.b64encode(c.tobytes()).decode()))
        clubs.append([cid, T.names.get(cid, str(cid)), leagues.get(str(b.buy_league), str(b.buy_league)),
                      *usual.get(str(b.buy_league), usual_all), country.get(str(b.buy_league), ""), str(b.buy_league)])
        allc.append(c)
    index = {"version": version, "players": len(p), "step": BUYER_STEP, "range": [float(v) for v in np.exp(q)],
             "range_edges": [int(e / 1000) for e in RANGE_EDGES],
             "top_countries": top,
             "fields": ["id", "name", "league", "usual_lo", "usual_hi", "country", "league_code"],
             "clubs": sorted(clubs, key=lambda r: (r[2], r[1]))}
    (SITE / "buyers.js").write_text("window.BUYERS = " + json.dumps(index, ensure_ascii=False, separators=(",", ":")) + ";",
                                    encoding="utf-8")
    # summary for the method note
    a = np.where(np.array(allc) == -128, np.nan, np.array(allc)) * BUYER_STEP        # clubs x players
    top = int(np.argmax(p.model_value.values))
    order = np.argsort(a[:, top])
    order = order[~np.isnan(a[order, top])]
    ex = [[clubs[i][1], clubs[i][2], float(p.tm_value.values[top] * np.exp(a[i, top]))] for i in list(order[-5:][::-1]) + list(order[:5])]
    summary = {"n_clubs": len(clubs), "n_players": len(p), "range": [float(v) for v in np.exp(q)], "shrink": shrink,
               "rounds": int(rounds),
               "n_train": len(past), "clipped": float((np.abs(np.array(allc)) == 127).mean()),
               "spread_p10_p90": float(np.nanmedian(np.exp(np.nanpercentile(a, 90, axis=0) - np.nanpercentile(a, 10, axis=0)))),
               "median_vs_no_buyer": float(np.nanmedian(np.exp(np.nanmedian(a, axis=0)) * p.tm_value.values / p.model_value.values)),
               "example_player": p["name"].values[top], "example_no_buyer": float(p.model_value.values[top]), "example": ex}
    (OUT / "buyer_option.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("buyer option:", len(clubs), "clubs | 80% interval by value band: x/÷", np.exp(q).round(2),
          "| clipped %.4f | files %.1f MB" % (summary["clipped"], sum(f.stat().st_size for f in out.iterdir()) / 1e6))


def main():
    T = load_tables()
    tr = pd.read_parquet("data/processed/transfers_model.parquet")
    parts = [build_features(candidates(T, d, 10_000_000 * (i + 1)), T) for i, d in enumerate(DATES)]
    parts.append(buyer_clubs(T, DATES[-1]))
    allrows = prepare(pd.concat([tr.assign(is_transfer=True)] + parts, ignore_index=True))
    buyers = allrows[allrows.buyer_row == True]  # noqa: E712
    allrows = allrows[allrows.buyer_row != True]  # noqa: E712
    allrows["log_ratio"] = allrows.log_fee - allrows.log_mv_pre
    tr = allrows[(allrows.is_transfer == True) & allrows.log_mv_pre.notna()]  # noqa: E712
    gone = pd.concat([T.players[T.players.current_club_id.isna()].player_id, pd.Series(DECEASED)])
    pl = allrows[(allrows.is_transfer != True) & allrows.log_mv_pre.notna() & allrows.age.notna()  # noqa: E712
                 & (allrows.position != "NA") & ~allrows.status.isin(["Retired", "Career break"])
                 & ~allrows.player_id.isin(gone)].copy()

    rounds = fit_lgb(tr[tr.split == "train"], tr[tr.split == "valid"], FEATS_V, "log_ratio").best_iteration
    out = []
    for d in DATES:
        past = tr[tr.transfer_date < d]
        cal = past[past.transfer_date >= d - pd.Timedelta(days=365)]
        m, shrink = fit_ratio_model(past, FEATS_V, rounds)       # with the scale-down for expensive players
        m_cal = fit_lgb(past[past.transfer_date < d - pd.Timedelta(days=365)], None, FEATS_V, "log_ratio", rounds)
        q = conformal_bands((cal.log_ratio - predict_ratio(m_cal, shrink, cal, FEATS_V)).abs(), cal.mv_pre, COVERAGE)
        p = pl[pl.transfer_date == d].copy()
        p["pred_log"] = predict_ratio(m, shrink, p, FEATS_V) + p.log_mv_pre
        qp = band_q(q, p.mv_pre)                                  # the range is narrower for expensive players
        p["model_value"], p["model_lo"], p["model_hi"] = np.exp(p.pred_log), np.exp(p.pred_log - qp), np.exp(p.pred_log + qp)
        out.append(p)
        print(d.date(), "trained on", len(past), "transfers | players valued:", len(p),
              "| 80% interval by value band: x/÷", np.exp(q).round(2), "| scale-down", shrink)
    out = pd.concat(out)
    out["club"] = out.from_club_id.map(T.names)
    cols = ["player_id", "name", "transfer_date", "club", "sell_league", "sell_country", "sub_position", "position",
            "age", "tm_value", "model_value", "model_lo", "model_hi"]
    out[cols].rename(columns={"transfer_date": "date"}).to_parquet(OUT / "player_values.parquet", index=False)

    # compact file for the website: players valued at the latest date, values in thousands of euros
    k = lambda s: (s / 1000).round().astype("Int64")  # noqa: E731
    wide = out.pivot(index="player_id", columns="transfer_date", values=["tm_value", "model_value", "from_club_id"])
    last = out[out.transfer_date == DATES[-1]].set_index("player_id")
    lg = T.comps.set_index("competition_id")
    lg["label"] = lg.name.str.replace("-", " ").str.title()
    lg["label"] = lg.label.where(~lg.label.duplicated(keep=False), lg.label + " (" + lg.country_name.fillna("") + ")")
    leagues = lg.label.to_dict()
    # portrait file name on the Transfermarkt image server ("" when the player has no photo)
    img = T.players.set_index("player_id").image_url.str.extract(r"/header/\d+-(\d+\.\w+)")[0].fillna("").to_dict()
    rows = []
    for pid, r in last.iterrows():
        series = lambda c: [None if pd.isna(x) else int(round(x / 1000)) for x in wide.loc[pid, c]]  # noqa: E731
        rows.append([int(pid), r["name"], r.club if isinstance(r.club, str) else "",
                     leagues.get(str(r.sell_league), ""), str(r.sell_country), str(r.sub_position) if
                     r.sub_position != "NA" else str(r.position), round(float(r.age), 1),
                     series("tm_value"), series("model_value"), int(k(pd.Series([r.model_lo]))[0]),
                     int(k(pd.Series([r.model_hi]))[0]), None if pd.isna(r.from_club_id) else int(r.from_club_id),
                     img.get(pid, ""), [None if pd.isna(x) else int(x) for x in wide.loc[pid, "from_club_id"]],
                     int(r.status != "")])
    data = {"dates": [d.strftime("%Y-%m-%d") for d in DATES], "coverage": COVERAGE,
            "fields": ["id", "name", "club", "league", "country", "position", "age", "tm", "model", "lo", "hi", "club_id",
                       "photo", "club_by_year", "free_agent"],
            "club_names": {int(c): T.names[c] for c in pd.unique(wide["from_club_id"].values.ravel()) if c in T.names},
            "players": rows}
    SITE.mkdir(exist_ok=True)
    (SITE / "data.js").write_text("window.DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";",
                                  encoding="utf-8")
    print("free agents:", int((last.status != "").sum()), "| moved on the valuation date or the day itself counted")
    print("site data:", len(rows), "players,", round((SITE / "data.js").stat().st_size / 1e6, 1), "MB")
    rounds_a = fit_lgb(tr[tr.split == "train"], tr[tr.split == "valid"], FEATS_A, "log_ratio").best_iteration
    version = pd.Timestamp.now().strftime("%Y%m%d%H%M")
    buyer_prices(out[out.transfer_date == DATES[-1]], buyers, tr, rounds_a, T, leagues, version)
    stamp_site(version)
    top = last.sort_values("model_value", ascending=False).head(15)
    print(top[["name", "club", "age", "tm_value", "model_value", "model_lo", "model_hi"]].round(0).to_string())
    r = np.log(last.model_value / last.tm_value)
    print("model vs Transfermarkt, log ratio quantiles:", r.quantile([.05, .25, .5, .75, .95]).round(2).to_dict())


if __name__ == "__main__":
    main()
