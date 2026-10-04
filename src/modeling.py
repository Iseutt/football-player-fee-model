"""Feature lists, preparation and LightGBM helpers shared by the modelling scripts."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

OUT = Path("outputs")
VALID_START, TEST_START = "2023-01-01", "2024-01-01"

CAT = ["position", "sub_position", "foot", "citizenship", "sell_league", "buy_league", "sell_country",
       "buy_country", "window"]
PLAYER = ["age", "height_in_cm", "n_prev_transfers", "tenure_days", "log_prev_fee_paid", "log_max_prev_fee",
          "log_seller_paid_fee"]
PERF = ["has_appearances", "min_365", "games_365", "goals_365", "assists_365", "ga_p90_365", "min_league_365",
        "min_uefa_365", "min_seller_365", "min_365_730", "min_trend", "career_min", "career_games", "career_goals"]
# position-specific statistics: what a goalkeeper, a defender, a midfielder or a forward is judged on
POS = ["goals_p90_365", "assists_p90_365", "conceded_p90_365", "clean_sheet_rate_365", "team_scored_p90_365",
       "team_gd_p90_365", "points_per_game_365", "cards_p90_365", "min_per_game_365", "full_game_share_365",
       "career_assists"]
POSITIONS = ["Goalkeeper", "Defender", "Midfield", "Attack"]
CLUB = ["sell_ppg", "buy_ppg", "log_buy_spend_3y", "buy_n_3y", "log_sell_income_3y", "sell_n_3y", "same_league",
        "same_country", "sell_youth_team", "buy_youth_team", "loan_back", "late_window", "year"]
# everything below is derived from Transfermarkt valuations (excluded from Model B)
TM = ["log_mv_pre", "mv_chg_6m", "mv_chg_12m", "mv_vs_peak", "mv_pre_age_days", "n_valuations", "years_tracked",
      "log_sell_squad_value", "log_buy_squad_value", "log_sell_league_avg_squad_value",
      "log_buy_league_avg_squad_value", "mkt_log_ratio_1y", "buy_club_log_ratio_3y", "sell_club_log_ratio_3y"]
# anything that requires knowing the buyer or the deal (excluded from Model V, used to value players not for sale)
DEAL = ["buy_league", "buy_country", "buy_ppg", "log_buy_spend_3y", "buy_n_3y", "same_league", "same_country",
        "buy_youth_team", "loan_back", "late_window", "year", "log_buy_squad_value",
        "log_buy_league_avg_squad_value", "buy_club_log_ratio_3y"]

FEATS_B = CAT + PLAYER + PERF + POS + CLUB
FEATS_A = FEATS_B + TM
FEATS_V = [f for f in FEATS_A if f not in DEAL]
FEATS_V_OLD = [f for f in FEATS_V if f not in POS]      # the 43 parameters used before the position statistics

BASE_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   cat_smooth=20, min_data_per_group=50, verbose=-1, seed=0)


def params(**override):
    """Default parameters, updated with the tuned ones if tune.py has been run."""
    p = dict(BASE_PARAMS)
    f = OUT / "best_params.json"
    if f.exists():
        p.update(json.loads(f.read_text()))
    p.update(override)
    return p


def prepare(df):
    for c in ["mv_pre", "prev_fee_paid", "max_prev_fee", "seller_paid_fee", "buy_spend_3y", "sell_income_3y",
              "sell_squad_value", "buy_squad_value", "sell_league_avg_squad_value", "buy_league_avg_squad_value"]:
        df["log_" + c] = np.log(df[c].where(df[c] > 0))
    df["mv_chg_6m"] = df.log_mv_pre - np.log(df.mv_pre_6m)
    df["mv_chg_12m"] = df.log_mv_pre - np.log(df.mv_pre_12m)
    df["mv_vs_peak"] = df.log_mv_pre - np.log(df.mv_peak)
    df["loan_back"] = df.loan_back.fillna(False).astype(int)
    for c in CAT:
        df[c] = df[c].fillna("NA").astype(str).astype("category")
    df["split"] = np.select([df.transfer_date < VALID_START, df.transfer_date < TEST_START], ["train", "valid"], "test")
    return df


def fit_lgb(tr, va, feats, target, rounds=None, p=None):
    """Fit with early stopping on `va`; if `rounds` is given, fit exactly that many rounds on `tr`."""
    p = p or params()
    dtr = lgb.Dataset(tr[feats], tr[target])
    if rounds is not None:
        return lgb.train(p, dtr, num_boost_round=rounds)
    return lgb.train(p, dtr, num_boost_round=5000, valid_sets=[lgb.Dataset(va[feats], va[target])],
                     callbacks=[lgb.early_stopping(200, verbose=False)])


def metrics(y, pred, label):
    e = pred - y
    ape = np.abs(np.exp(e) - 1)
    return {"model": label, "n": len(y), "rmse_log": np.sqrt((e ** 2).mean()), "mae_log": e.abs().mean(),
            "median_ape": np.median(ape), "r2_log": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "within_25pct": (ape <= 0.25).mean(), "bias_log": e.mean()}


def conformal_q(abs_res, coverage):
    """Split-conformal quantile of absolute calibration residuals."""
    n = len(abs_res)
    return np.quantile(abs_res, min(1, np.ceil((n + 1) * coverage) / n))
