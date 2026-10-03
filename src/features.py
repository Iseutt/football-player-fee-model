"""Feature construction, shared by the transfer table and the player valuation table.

Every feature is measured strictly before the reference date of the row
(the transfer date, or the date at which a player is valued).
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

RAW = Path("data/raw")
END = "2026-07-06"        # last scrape date of the source
MV_LAG_DAYS = 30          # a valuation must be at least this old to be used
TOP_N_SQUAD = 18
YOUTH = r"U\d\d|Youth|Yth| II$| B$| 2$| 2 |Res\.?$"

PERF = ["min_365", "games_365", "goals_365", "assists_365", "min_league_365", "min_uefa_365",
        "min_seller_365", "min_365_730", "career_min", "career_games", "career_goals"]


def season_of(d):
    """Football season start year (2024 for 2024/25) of a date series."""
    return d.dt.year - (d.dt.month < 7).astype(int)


def read(name, **kw):
    return pd.read_csv(RAW / f"{name}.csv.gz", low_memory=False, **kw)


# --------------------------------------------------------------------------- tables
def load_transfers():
    t = read("transfers", parse_dates=["transfer_date"])
    t = t.drop_duplicates(["player_id", "transfer_date", "from_club_id", "to_club_id"])
    t = t[t.transfer_date <= END].sort_values(["player_id", "transfer_date"]).reset_index(drop=True)

    g = t.groupby("player_id")
    nxt = {c: g[c].shift(-1) for c in ["from_club_id", "to_club_id", "transfer_fee", "transfer_date"]}
    gap = (nxt["transfer_date"] - t.transfer_date).dt.days
    back = (nxt["to_club_id"] == t.from_club_id) & (nxt["transfer_fee"].fillna(0) == 0)
    ratio = t.transfer_fee / t.market_value_in_eur
    # bought and immediately loaned back to the seller: a genuine permanent transfer
    t["loan_back"] = back & (gap <= 45)
    # went back to the seller for free later on, at a price far below value: a loan fee
    loan_fee = back & (gap > 45) & (gap <= 800) & (ratio < 0.35)
    # next move starts again from the seller: the player never really left
    chain_break = (nxt["from_club_id"] == t.from_club_id) & (nxt["to_club_id"] != t.to_club_id)
    t["is_loan_fee"] = loan_fee | chain_break

    # running record of the player, including the current row (looked up later with a strict "before")
    paid = t.transfer_fee.where(t.transfer_fee > 0)
    t["cum_n"] = g.cumcount() + 1
    t["cum_last_paid"] = paid.groupby(t.player_id).ffill()
    t["cum_max_paid"] = paid.groupby(t.player_id).cummax()
    return t


def club_season_tables(games, app, val, comps):
    """League, points per game and squad value of every club-season in the covered leagues."""
    leagues = comps[comps.type == "domestic_league"].competition_id
    g = games[games.competition_id.isin(leagues)]
    cols = ["season", "league", "club_id", "gf", "ga"]
    home = g[["season", "competition_id", "home_club_id", "home_club_goals", "away_club_goals"]].set_axis(cols, axis=1)
    away = g[["season", "competition_id", "away_club_id", "away_club_goals", "home_club_goals"]].set_axis(cols, axis=1)
    cg = pd.concat([home, away])
    cg["pts"] = np.select([cg.gf > cg.ga, cg.gf == cg.ga], [3, 1], 0)
    cs = cg.groupby(["club_id", "season"]).agg(league=("league", "first"), ppg=("pts", "mean")).reset_index()

    # squad value: players who appeared for the club in season S, valued on 1 April of S+1
    a = app.assign(season=season_of(app.date))[["player_club_id", "season", "player_id"]].drop_duplicates()
    a["cut"] = pd.to_datetime((a.season + 1).astype(str) + "-04-01")
    v = val[["player_id", "date", "market_value_in_eur"]].sort_values("date")
    a = pd.merge_asof(a.sort_values("cut"), v, left_on="cut", right_on="date", by="player_id")
    sq = (a.dropna(subset=["market_value_in_eur"]).sort_values("market_value_in_eur", ascending=False)
          .groupby(["player_club_id", "season"]).market_value_in_eur
          .apply(lambda x: x.head(TOP_N_SQUAD).sum()).rename("squad_value").reset_index()
          .rename(columns={"player_club_id": "club_id"}))
    cs = cs.merge(sq, on=["club_id", "season"], how="outer")
    lg = cs.groupby(["league", "season"]).squad_value.mean().rename("league_avg_squad_value").reset_index()
    return cs, lg


def club_countries(comps):
    """Country of each club: covered leagues first, then the d2ski open transfer dataset."""
    clubs = read("clubs", usecols=["club_id", "domestic_competition_id"]).merge(
        comps[["competition_id", "country_name"]], left_on="domestic_competition_id", right_on="competition_id")
    out = clubs.set_index("club_id").country_name.dropna()
    d = pd.read_csv(RAW / "d2ski_transfers.csv", usecols=["team_id", "team_country", "counter_team_id",
                                                           "counter_team_country"])
    d["counter_team_id"] = pd.to_numeric(d.counter_team_id, errors="coerce")
    extra = pd.concat([d[["team_id", "team_country"]].set_axis(["club_id", "country"], axis=1),
                       d[["counter_team_id", "counter_team_country"]].set_axis(["club_id", "country"], axis=1)])
    extra = extra.dropna().drop_duplicates("club_id").set_index("club_id").country
    return pd.concat([out, extra[~extra.index.isin(out.index)]]).to_dict()


def load_tables():
    t = load_transfers()
    val = read("player_valuations", parse_dates=["date"])
    app = read("appearances", parse_dates=["date"],
               usecols=["player_id", "player_club_id", "date", "competition_id", "goals", "assists", "minutes_played"])
    games = read("games", usecols=["season", "competition_id", "home_club_id", "away_club_id",
                                   "home_club_goals", "away_club_goals"])
    comps = read("competitions")
    cs, lg = club_season_tables(games, app, val, comps)
    names = pd.concat([t[["from_club_id", "from_club_name"]].set_axis(["id", "name"], axis=1),
                       t[["to_club_id", "to_club_name"]].set_axis(["id", "name"], axis=1)])
    names = names.dropna().drop_duplicates("id").set_index("id").name.to_dict()
    return SimpleNamespace(transfers=t, players=read("players", parse_dates=["date_of_birth"]), val=val, app=app,
                           comps=comps, cs=cs, lg=lg, country=club_countries(comps), names=names)


# --------------------------------------------------------------------------- feature blocks
def rolling_by_key(events, key, value, q_key, q_date, days, how="sum"):
    """For each query (key, date): aggregate `value` of events with the same key in [date-days, date)."""
    ev = events.sort_values("transfer_date")
    groups = {k: (g.transfer_date.values, g[value].values) for k, g in ev.groupby(key)}
    win = np.timedelta64(days, "D")
    agg, cnt = np.full(len(q_key), np.nan), np.zeros(len(q_key))
    for i, (k, d) in enumerate(zip(q_key, q_date)):
        if k not in groups:
            continue
        dates, vals = groups[k]
        lo, hi = np.searchsorted(dates, d - win, "left"), np.searchsorted(dates, d, "left")
        cnt[i] = hi - lo
        if hi > lo:
            agg[i] = vals[lo:hi].sum() if how == "sum" else np.median(vals[lo:hi])
    return agg, cnt


def history_features(s, T):
    """Past transfer record of the player. Also fills a missing current club."""
    h = T.transfers[["player_id", "transfer_date", "to_club_id", "transfer_fee", "cum_n", "cum_last_paid",
                     "cum_max_paid"]].rename(columns={"transfer_date": "h_date", "to_club_id": "h_to",
                                                      "transfer_fee": "h_fee"}).sort_values("h_date")
    m = pd.merge_asof(s[["tid", "player_id", "transfer_date"]].sort_values("transfer_date"), h,
                      left_on="transfer_date", right_on="h_date", by="player_id",
                      allow_exact_matches=False).set_index("tid").reindex(s.tid)
    m.index = s.index
    if s.from_club_id.isna().any():     # valuation rows: current club = last destination, else last club played for
        a = T.app[["player_id", "date", "player_club_id"]].sort_values("date")
        la = pd.merge_asof(s[["tid", "player_id", "transfer_date"]].sort_values("transfer_date"), a,
                           left_on="transfer_date", right_on="date", by="player_id",
                           allow_exact_matches=False).set_index("tid").player_club_id.reindex(s.tid)
        s["from_club_id"] = s.from_club_id.fillna(m.h_to).fillna(pd.Series(la.values, index=s.index))
    s["n_prev_transfers"] = m.cum_n.fillna(0)
    s["prev_fee_paid"], s["max_prev_fee"] = m.cum_last_paid, m.cum_max_paid
    same = m.h_to == s.from_club_id
    s["tenure_days"] = (s.transfer_date - m.h_date).dt.days.where(same)
    s["seller_paid_fee"] = m.h_fee.where(same & (m.h_fee > 0))
    return s


def valuation_features(s, val):
    v = val[["player_id", "date", "market_value_in_eur"]].rename(
        columns={"date": "vdate", "market_value_in_eur": "mv"}).sort_values("vdate")
    out = s[["tid", "player_id", "transfer_date"]].copy()
    for name, lag in [("mv_pre", MV_LAG_DAYS), ("mv_pre_6m", MV_LAG_DAYS + 180), ("mv_pre_12m", MV_LAG_DAYS + 365)]:
        q = out.assign(cut=out.transfer_date - pd.Timedelta(days=lag)).sort_values("cut")
        m = pd.merge_asof(q, v, left_on="cut", right_on="vdate", by="player_id").set_index("tid")
        out[name] = out.tid.map(m.mv)
        if name == "mv_pre":
            out["mv_pre_age_days"] = out.tid.map((m.transfer_date - m.vdate).dt.days)
    h = out[["tid", "player_id", "transfer_date"]].merge(v, on="player_id")
    h = h[h.vdate <= h.transfer_date - pd.Timedelta(days=MV_LAG_DAYS)]
    a = h.groupby("tid").agg(mv_peak=("mv", "max"), n_valuations=("mv", "size"), first_val=("vdate", "min"))
    out = out.join(a, on="tid")
    out["years_tracked"] = (out.transfer_date - out.first_val).dt.days / 365.25
    return out.drop(columns=["player_id", "transfer_date", "first_val"])


def performance_features(s, T):
    a = T.app[T.app.player_id.isin(s.player_id.unique())].merge(
        T.comps[["competition_id", "type"]], on="competition_id", how="left")
    m = s[["tid", "player_id", "transfer_date", "from_club_id"]].merge(a, on="player_id")
    days = (m.transfer_date - m.date).dt.days
    m = m[(days > 0) & (days <= 3650)]
    days = days[m.index]
    y1, y2 = days <= 365, (days > 365) & (days <= 730)
    mins = m.minutes_played
    return pd.DataFrame({
        "tid": m.tid,
        "min_365": mins.where(y1, 0), "games_365": y1.astype(int),
        "goals_365": m.goals.where(y1, 0), "assists_365": m.assists.where(y1, 0),
        "min_league_365": mins.where(y1 & (m.type == "domestic_league"), 0),
        "min_uefa_365": mins.where(y1 & (m.type == "international_cup"), 0),
        "min_seller_365": mins.where(y1 & (m.player_club_id == m.from_club_id), 0),
        "min_365_730": mins.where(y2, 0),
        "career_min": mins, "career_games": 1, "career_goals": m.goals,
    }).groupby("tid").sum().reset_index()


def attach_club(s, T, side, club_col, league_season, prev_season):
    """League from `league_season` (fallback: nearest season), strength from `prev_season`."""
    lk = T.cs.dropna(subset=["league"]).set_index(["club_id", "season"]).league.to_dict()
    league = pd.Series([next((lk[(c, y + k)] for k in (0, -1, 1, -2) if (c, y + k) in lk), "OTHER")
                        for c, y in zip(s[club_col], league_season)], index=s.index)
    s[f"{side}_league"] = league
    s[f"{side}_country"] = s[club_col].map(T.country).fillna("Unknown")
    s[f"{side}_youth_team"] = s[club_col].map(T.names).fillna("").str.contains(YOUTH, regex=True).astype(int)
    key = pd.MultiIndex.from_arrays([s[club_col], prev_season])
    c = T.cs.set_index(["club_id", "season"])
    s[f"{side}_ppg"] = c.ppg.reindex(key).values
    s[f"{side}_squad_value"] = c.squad_value.reindex(key).values
    lkey = pd.MultiIndex.from_arrays([league, prev_season])
    s[f"{side}_league_avg_squad_value"] = T.lg.set_index(["league", "season"]).league_avg_squad_value.reindex(lkey).values
    return s


# --------------------------------------------------------------------------- assembly
def build_features(s, T):
    """s: one row per (tid, player_id, transfer_date, from_club_id, to_club_id); club ids may be missing."""
    s = history_features(s.reset_index(drop=True).copy(), T)

    # static player attributes (date of birth, position, foot, height do not leak)
    s = s.merge(T.players[["player_id", "name", "date_of_birth", "position", "sub_position", "foot", "height_in_cm",
                           "country_of_citizenship"]], on="player_id", how="left")
    s["age"] = (s.transfer_date - s.date_of_birth).dt.days / 365.25
    s["citizenship"] = s.country_of_citizenship

    s = s.merge(valuation_features(s, T.val), on="tid", how="left")
    s = s.merge(performance_features(s, T), on="tid", how="left")
    s["has_appearances"] = s.career_games.notna().astype(int)
    s[PERF] = s[PERF].fillna(0)
    s["ga_p90_365"] = (s.goals_365 + s.assists_365) / s.min_365.clip(lower=450) * 90
    s["min_trend"] = s.min_365 - s.min_365_730

    # clubs: league known for the season the date falls in, strength from the last completed season
    ref = season_of(s.transfer_date + pd.Timedelta(days=45))
    s = attach_club(s, T, "sell", "from_club_id", season_of(s.transfer_date - pd.Timedelta(days=60)), ref - 1)
    s = attach_club(s, T, "buy", "to_club_id", ref, ref - 1)
    s["same_league"] = ((s.sell_league == s.buy_league) & (s.buy_league != "OTHER")).astype(int)
    s["same_country"] = ((s.sell_country == s.buy_country) & (s.buy_country != "Unknown")).astype(int)

    # club and market activity over the past, from all earlier paid transfers
    t = T.transfers
    paid = t[t.transfer_fee > 0].copy()
    paid["log_ratio"] = np.log(paid.transfer_fee / paid.market_value_in_eur.where(paid.market_value_in_eur > 0))
    d = s.transfer_date.values
    s["buy_spend_3y"], s["buy_n_3y"] = rolling_by_key(paid, "to_club_id", "transfer_fee", s.to_club_id, d, 1095)
    s["sell_income_3y"], s["sell_n_3y"] = rolling_by_key(paid, "from_club_id", "transfer_fee", s.from_club_id, d, 1095)
    pr = paid.dropna(subset=["log_ratio"]).assign(all=0)
    s["mkt_log_ratio_1y"], _ = rolling_by_key(pr, "all", "log_ratio", np.zeros(len(s)), d, 365, "median")
    s["buy_club_log_ratio_3y"], _ = rolling_by_key(pr, "to_club_id", "log_ratio", s.to_club_id, d, 1095, "median")
    s["sell_club_log_ratio_3y"], _ = rolling_by_key(pr, "from_club_id", "log_ratio", s.from_club_id, d, 1095, "median")

    # timing
    m, day = s.transfer_date.dt.month, s.transfer_date.dt.day
    s["window"] = np.select([m.isin([6, 7, 8, 9]), m.isin([1, 2])], ["summer", "winter"], "other")
    s["late_window"] = (((m == 8) & (day >= 25)) | ((m == 9) & (day <= 2)) |
                        ((m == 1) & (day >= 25)) | ((m == 2) & (day <= 2))).astype(int)
    s["year"] = s.transfer_date.dt.year
    return s.drop(columns=["date_of_birth", "country_of_citizenship"])
