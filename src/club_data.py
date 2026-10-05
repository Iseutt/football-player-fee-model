"""Club pages of the website: squad value, last results, last transfers and the club's selling and buying record.

Reads what the other scripts produced (site data, price gaps) and writes docs/clubs.js and docs/clubs/<id>.js.
Run from the project root (after value_players.py):  python src/club_data.py
"""
import json
import shutil

import numpy as np
import pandas as pd
import scipy.stats as st

from features import END, read
from modeling import OUT
from value_players import SITE, classified_moves, stamp_site

MIN_DEALS = 8          # below this number of paid deals, no statement is made about a club's habit
SHOWN = [0, 1, 2, 6]   # kinds of move listed on a club page: transfer, free transfer, loan, fee not disclosed


def site_json(name, prefix):
    return json.loads((SITE / name).read_text(encoding="utf-8")[len(prefix):-1])


def record(x, lr_year, gap):
    """How a club's paid deals compare with Transfermarkt values and with the model price.
    `lr_year`: log(fee / value) minus the market median of the same year; `gap`: log(fee / model price)."""
    n = len(x)
    out = {"n": n, "ratio": None if not n else float(np.exp(x.median()))}
    if n >= MIN_DEALS:
        # typical (median) gap, tested with a signed-rank test: a few extreme deals do not decide the answer
        out.update({"vs_market": float(np.exp(lr_year.median()) - 1), "p_market": float(st.wilcoxon(lr_year).pvalue),
                    "vs_model": float(np.exp(gap.median()) - 1), "p_model": float(st.wilcoxon(gap).pvalue)})
    return out


def main():
    data, buyers = site_json("data.js", "window.DATA = "), site_json("buyers.js", "window.BUYERS = ")
    clubs = {c[0]: c for c in buyers["clubs"]}
    k = lambda x: None if pd.isna(x) else int(round(x / 1000))  # noqa: E731

    # squads: the players listed on the site, at their club of today
    squad = {}
    for r in data["players"]:
        if r[11] in clubs and not r[14]:
            squad.setdefault(r[11], []).append([r[0], r[1], r[5], int(r[6]), r[7][-1], r[8][-1], r[12]])

    # paid transfers with the Transfermarkt value before the deal and the model price (cross-fitted full model)
    pg = pd.read_parquet(OUT / "price_gaps.parquet").dropna(subset=["mv_pre", "pred_log_a"])
    pg["lr"] = np.log(pg.transfer_fee / pg.mv_pre)
    pg["lr_year"] = pg.lr - pg.groupby(pg.transfer_date.dt.year).lr.transform("median")
    pg["model"] = np.exp(pg.pred_log_a)
    market = {"ratio": float(np.exp(pg.lr.median())), "n": len(pg)}

    # every move, to list arrivals and departures
    mv = classified_moves()
    mv = mv[mv.kind.isin(SHOWN)].merge(pg[["player_id", "transfer_date", "to_club_id", "model"]],
                                       on=["player_id", "transfer_date", "to_club_id"], how="left")
    listed = {r[0] for r in data["players"]}

    def moves(rows, other):
        rows = rows.sort_values("transfer_date", ascending=False).head(12)
        return [[r.transfer_date.strftime("%Y-%m-%d"), int(r.player_id), r.player_name, k(getattr(r, other + "_club_id") * 1000),
                 getattr(r, other + "_club_name"), k(r.transfer_fee), int(r.kind), k(r.market_value_in_eur), k(r.model),
                 int(r.player_id in listed)] for r in rows.itertuples()]

    # results
    g = read("games", parse_dates=["date"], usecols=["date", "season", "competition_id", "competition_type", "home_club_id",
                                                     "away_club_id", "home_club_goals", "away_club_goals",
                                                     "home_club_position", "away_club_position", "home_club_name",
                                                     "away_club_name"])
    g = g[(g.date <= END) & g.home_club_goals.notna()]
    comp = read("competitions").set_index("competition_id").name.str.replace("-", " ").str.title().to_dict()
    cols = ["date", "season", "competition_id", "competition_type", "club", "opp", "opp_name", "gf", "ga", "pos", "home"]
    side = lambda a, b, home: g[["date", "season", "competition_id", "competition_type", f"{a}_club_id", f"{b}_club_id",  # noqa: E731
                                 f"{b}_club_name", f"{a}_club_goals", f"{b}_club_goals", f"{a}_club_position"]].assign(home=home).set_axis(cols, axis=1)
    games = pd.concat([side("home", "away", 1), side("away", "home", 0)]).sort_values("date")
    games = games[games.club.isin(clubs)]

    out = SITE / "clubs"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    index = []
    for cid, c in clubs.items():
        sq = sorted(squad.get(cid, []), key=lambda r: -r[5])
        sold, bought = pg[pg.from_club_id == cid], pg[pg.to_club_id == cid]
        sell, buy = record(sold.lr, sold.lr_year, sold.gap_a), record(bought.lr, bought.lr_year, bought.gap_a)
        gc = games[games.club == cid]
        lg = gc[gc.competition_type == "domestic_league"]
        season = None
        if len(lg):
            s = lg[lg.season == lg.season.max()]
            w, d, l = int((s.gf > s.ga).sum()), int((s.gf == s.ga).sum()), int((s.gf < s.ga).sum())
            pos = s.pos.dropna()
            season = {"label": "%d/%02d" % (s.season.max(), (s.season.max() + 1) % 100), "played": len(s), "w": w, "d": d, "l": l,
                      "gf": int(s.gf.sum()), "ga": int(s.ga.sum()), "ppg": round((3 * w + d) / len(s), 2),
                      "position": None if pos.empty else int(pos.iloc[-1])}
        results = [[r.date.strftime("%Y-%m-%d"), comp.get(r.competition_id, r.competition_id), int(r.home), int(r.opp),
                    r.opp_name if isinstance(r.opp_name, str) else "", int(r.gf), int(r.ga)]
                   for r in gc.tail(6).iloc[::-1].itertuples()]
        detail = {"squad": sq, "season": season, "results": results, "sell": sell, "buy": buy,
                  "out": moves(mv[mv.from_club_id == cid], "to"), "in": moves(mv[mv.to_club_id == cid], "from")}
        (out / f"{cid}.js").write_text("window.CLUB=window.CLUB||{};window.CLUB[%d]=%s;"
                                       % (cid, json.dumps(detail, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
        index.append([cid, c[1], c[2], c[5], len(sq), sum(r[4] for r in sq), sum(r[5] for r in sq),
                      sell["n"], sell.get("vs_market"), sell.get("p_market"), buy["n"], buy.get("vs_market"), buy.get("p_market"),
                      None if season is None else season["position"]])
    idx = {"version": buyers["version"], "market": market, "min_deals": MIN_DEALS, "top_countries": buyers["top_countries"],
           "fields": ["id", "name", "league", "country", "players", "squad_tm", "squad_model", "sales", "sells_vs_market",
                      "sells_p", "purchases", "buys_vs_market", "buys_p", "position"],
           "clubs": sorted(index, key=lambda r: -r[6])}
    (SITE / "clubs.js").write_text("window.CLUBS = " + json.dumps(idx, ensure_ascii=False, separators=(",", ":")) + ";", encoding="utf-8")
    stamp_site(buyers["version"])
    top = pd.DataFrame(idx["clubs"], columns=idx["fields"])
    print("clubs:", len(index), "| market median fee / value: %.2f | files %.1f MB"
          % (market["ratio"], sum(f.stat().st_size for f in out.iterdir()) / 1e6))
    print(top.head(8)[["name", "players", "squad_tm", "squad_model", "sales", "sells_vs_market", "sells_p"]].round(3).to_string(index=False))
    sig = top[(top.sells_p < 0.05)]
    print("clubs with at least %d paid sales: %d | significantly above the market: %d | below: %d"
          % (MIN_DEALS, top.sells_p.notna().sum(), (sig.sells_vs_market > 0).sum(), (sig.sells_vs_market < 0).sum()))


if __name__ == "__main__":
    main()
