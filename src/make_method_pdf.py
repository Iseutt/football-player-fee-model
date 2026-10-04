"""Write Player_Valuation_Method.pdf, a plain-language explanation of how players are valued.

The weights, errors and position statistics are read from outputs/ and from the transfer table,
so the document follows the model.
Run from the project root (after train_model.py and value_players.py):  python src/make_method_pdf.py
"""
import json

import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from modeling import DEAL, FEATS_A, FEATS_V, FEATS_V_OLD, OUT, POS, POSITIONS, prepare

INK, INK2, GRID, BLUE = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#e1e0d9"), colors.HexColor("#2a78d6")
W = A4[0] - 4 * cm
MIN_MINUTES = 900      # position statistics are computed on players with at least ten full games in the year

# group and plain name of every parameter of the website model
PARAMS = {
    "Transfermarkt value and its history": {
        "log_mv_pre": "Last Transfermarkt value (at least 30 days old)",
        "n_valuations": "Number of Transfermarkt valuations so far",
        "years_tracked": "Years since the first valuation",
        "mv_chg_6m": "Change in value over 6 months",
        "mv_chg_12m": "Change in value over 12 months",
        "mv_vs_peak": "Value compared with the player's peak value",
        "mv_pre_age_days": "Age of the last valuation, in days"},
    "Player profile": {
        "age": "Age", "citizenship": "Nationality", "sub_position": "Detailed position (e.g. left winger)",
        "height_in_cm": "Height", "position": "Position (goalkeeper, defender, midfield, attack)",
        "foot": "Preferred foot"},
    "Current club": {
        "sell_country": "Country of the club", "log_sell_income_3y": "Club's transfer income over 3 years",
        "sell_club_log_ratio_3y": "Club's past sales compared with Transfermarkt values",
        "sell_ppg": "Club's points per game last season", "sell_league": "League of the club",
        "log_sell_squad_value": "Squad value of the club",
        "log_sell_league_avg_squad_value": "Average squad value in the club's league",
        "sell_n_3y": "Number of sales by the club over 3 years", "sell_youth_team": "Youth or reserve team"},
    "Past transfers of the player": {
        "tenure_days": "Time at the current club", "log_max_prev_fee": "Highest fee ever paid for the player",
        "log_seller_paid_fee": "Fee the current club paid", "log_prev_fee_paid": "Last fee paid for the player",
        "n_prev_transfers": "Number of previous transfers"},
    "Playing time and output": {
        "min_seller_365": "Minutes for the current club, last 365 days", "min_trend": "Change in minutes versus the year before",
        "ga_p90_365": "Goals plus assists per 90 minutes", "career_min": "Career minutes",
        "min_365": "Minutes, last 365 days", "career_games": "Career games",
        "min_league_365": "Minutes in the league, last 365 days", "min_365_730": "Minutes in the year before",
        "career_goals": "Career goals", "games_365": "Games, last 365 days", "goals_365": "Goals, last 365 days",
        "assists_365": "Assists, last 365 days", "has_appearances": "Match data available for the player",
        "min_uefa_365": "Minutes in European competitions, last 365 days",
        "goals_p90_365": "Goals per 90 minutes", "assists_p90_365": "Assists per 90 minutes",
        "conceded_p90_365": "Goals conceded by his team per 90 minutes he played",
        "clean_sheet_rate_365": "Share of his games without a goal conceded",
        "team_scored_p90_365": "Goals scored by his team per 90 minutes he played",
        "team_gd_p90_365": "Goal difference of his team per 90 minutes he played",
        "points_per_game_365": "Points per game won by his team when he played",
        "cards_p90_365": "Cards per 90 minutes (a red counts as three yellows)",
        "min_per_game_365": "Minutes per game", "full_game_share_365": "Share of games played in full",
        "career_assists": "Career assists"},
    "Market conditions": {
        "mkt_log_ratio_1y": "Fees compared with Transfermarkt values across the market, last year",
        "window": "Summer or winter window"},
}
LABEL = {f: name for g in PARAMS.values() for f, name in g.items()}
GROUP = {f: g for g, d in PARAMS.items() for f in d}
assert set(LABEL) == set(FEATS_V), set(LABEL) ^ set(FEATS_V)

# the parameters only the full model has: name and why it matters
BUYER = "Buyer and deal"
DEAL_PARAMS = {
    "log_buy_spend_3y": ("Buying club's transfer spending over 3 years", "A club that spends a lot pays more for the same player."),
    "buy_country": ("Country of the buying club", "Clubs in England pay more than clubs elsewhere for the same player."),
    "buy_club_log_ratio_3y": ("Buying club's past purchases compared with Transfermarkt values",
                              "Some clubs systematically pay above or below value."),
    "log_buy_squad_value": ("Squad value of the buying club", "A measure of the buyer's sporting and financial level."),
    "year": ("Year of the transfer", "Lets the model follow the market over time."),
    "buy_ppg": ("Buying club's points per game last season", "Stronger clubs buy at higher prices."),
    "log_buy_league_avg_squad_value": ("Average squad value in the buyer's league", "The wealth of the league the player moves to."),
    "buy_league": ("League of the buying club", "League-specific price levels beyond the country."),
    "buy_n_3y": ("Number of purchases by the buying club over 3 years", "How active the buyer is on the market."),
    "same_country": ("Buyer and seller in the same country", "Domestic deals are priced differently from moves abroad."),
    "buy_youth_team": ("Buyer is a youth or reserve team", "Signings for a second team are cheaper."),
    "late_window": ("Deal in the last week of the window", "Last-minute deals are made under time pressure."),
    "loan_back": ("Player loaned back to the seller after the sale", "The buyer pays for the future, not for immediate use."),
    "same_league": ("Buyer and seller in the same league", "Deals inside one league could be priced differently; the model finds almost no effect."),
}
assert set(DEAL_PARAMS) == set(DEAL), set(DEAL_PARAMS) ^ set(DEAL)
LABEL_A = {**LABEL, **{f: v[0] for f, v in DEAL_PARAMS.items()}}
GROUP_A = {**GROUP, **{f: BUYER for f in DEAL}}

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=10, leading=14.5, textColor=INK, spaceAfter=7)
small = ParagraphStyle("small", parent=body, fontSize=9, leading=12.5, spaceAfter=0)
smallb = ParagraphStyle("smallb", parent=small, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=4)
sub = ParagraphStyle("sub", parent=body, textColor=INK2, spaceAfter=14)
part = ParagraphStyle("part", parent=body, fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=4)
h2 = ParagraphStyle("h2", parent=body, keepWithNext=1, fontName="Helvetica-Bold", fontSize=13, leading=17, spaceBefore=12, spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=0, spaceAfter=4)


def P(text, style=body):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def table(rows, widths, header=True, bold_rows=()):
    data = [[P(str(c), smallb if (header and i == 0) or i in bold_rows else small) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.5, GRID),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2), ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    return t


def bars(items):
    """Horizontal bar chart: share of the model's correction explained by each group of parameters."""
    row, left, d = 22, 170, Drawing(W, 22 * len(items) + 6)
    scale = (W - left - 40) / max(v for _, v in items)
    for i, (label, v) in enumerate(items):
        y = d.height - (i + 1) * row + 5
        d.add(String(0, y + 3, label, fontName="Helvetica", fontSize=9, fillColor=INK))
        d.add(Rect(left, y, v * scale, 12, fillColor=BLUE, strokeColor=None, rx=2, ry=2))
        d.add(String(left + v * scale + 5, y + 3, "%d%%" % round(v * 100), fontName="Helvetica", fontSize=9, fillColor=INK))
    return d


def pc(x, digits=1):
    return "%.*f%%" % (digits, 100 * x)


# ---------------------------------------------------------------- numbers
gain = pd.read_csv(OUT / "feature_importance_v.csv", index_col=0).gain_share
gain_old = pd.read_csv(OUT / "feature_importance_v_43.csv", index_col=0).gain_share
share = pd.read_csv(OUT / "feature_share_by_position.csv", index_col=0)
pos_err = pd.read_csv(OUT / "position_errors_test.csv").set_index("position")
met = pd.read_csv(OUT / "metrics_test.csv").set_index("model")
M_TM, M_V, M_A = met.loc["naive: fee = market value"], met.loc["LightGBM V (no buyer information)"], met.loc["LightGBM A (target: fee / value)"]
M_OLD = met.loc["LightGBM V without position statistics (43 parameters)"]
group_gain = gain.groupby(GROUP).sum().sort_values(ascending=False)
group_share = share.groupby(GROUP).sum().loc[group_gain.index]

vals = pd.read_parquet(OUT / "player_values.parquet")
last = vals[vals.date == vals.date.max()]
lo, hi = (last.model_lo / last.model_value).median(), (last.model_hi / last.model_value).median()

tr = prepare(pd.read_parquet("data/processed/transfers_model.parquet"))
tr = tr[tr.log_mv_pre.notna()]
pl = tr[tr.min_365 >= MIN_MINUTES]
STATS = [("goals_p90_365", "Goals per 90 minutes", "%.2f"), ("assists_p90_365", "Assists per 90 minutes", "%.2f"),
         ("conceded_p90_365", "Goals conceded by his team per 90 minutes", "%.2f"),
         ("clean_sheet_rate_365", "Share of games without a goal conceded", "%.0f%%"),
         ("cards_p90_365", "Cards per 90 minutes", "%.2f"),
         ("points_per_game_365", "Points per game when he played", "%.2f"),
         ("min_per_game_365", "Minutes per game", "%.0f"), ("full_game_share_365", "Share of games played in full", "%.0f%%"),
         ("min_365", "Minutes over the year", "%.0f")]
by_pos = pl.groupby("position", observed=True)


def stat_row(col, label, fmt):
    k = 100 if "%%" in fmt else 1
    return [label, *[fmt % (k * by_pos[col].mean()[p]) for p in POSITIONS]]


def corr_row(col, label, _):
    return [label, *["%+.2f" % pl[pl.position == p][col].corr(pl[pl.position == p].log_fee, method="spearman")
                     for p in POSITIONS]]


def top_for(p, n=3):
    """The position statistics that weigh most for position p."""
    s = share.loc[POS + ["ga_p90_365"], p].sort_values(ascending=False).head(n)
    return ", ".join("%s (%s)" % (LABEL[f].lower(), pc(v)) for f, v in s.items())


param_rows = [["Parameter", "Group", "Before (43)", "Now (%d)" % len(FEATS_V)]] + [
    [LABEL[f], GROUP[f], pc(gain_old[f]) if f in gain_old else "new", pc(gain[f])] for f in gain.index]
pos_rows, bold = [["Parameter", "All", *POSITIONS]], []
for g in group_gain.index:
    bold.append(len(pos_rows))
    pos_rows.append([g, *[pc(group_share.loc[g, c]) for c in ["All"] + POSITIONS]])
    for f in share.loc[list(PARAMS[g])].sort_values("All", ascending=False).index:
        pos_rows.append([LABEL[f], *[pc(share.loc[f, c]) for c in ["All"] + POSITIONS]])

# full model
gain_a = pd.read_csv(OUT / "feature_importance_a.csv", index_col=0).gain_share
group_gain_a = gain_a.groupby(GROUP_A).sum().sort_values(ascending=False)
buy = json.loads((OUT / "buyer_option.json").read_text(encoding="utf-8"))
n_buy_leagues = len({c[2] for c in json.loads(open("docs/buyers.js", encoding="utf-8").read()[len("window.BUYERS = "):-1])["clubs"]})
full_rows = [["Parameter", "Group", "Full model", "Website model"]] + [
    [LABEL_A[f], GROUP_A[f], pc(gain_a[f]), pc(gain[f]) if f in gain else "not used"] for f in gain_a.index]
bold_a = [i for i, r in enumerate(full_rows) if r[1] == BUYER]


def eur(x):
    return "EUR %.0fm" % (x / 1e6) if x >= 1e7 else "EUR %.1fm" % (x / 1e6)

story = [
    P("How the players are valued", h1),
    P("The method behind the model price on the website, the parameters it uses, and why.", sub),

    P("1. What the model price means", h2),
    P("The model price is an estimate of the <b>transfer fee</b> a player would fetch if he were sold from his "
      "current club on a given date. It is not a rating of how good the player is. It answers: given what clubs "
      "actually paid in the past for players like this one, what would a club pay today?"),
    P("The model learns from %s real transfers with a disclosed fee between January 2014 and July 2026. "
      "It then applies what it learned to %s active players who are not necessarily for sale."
      % (format(len(tr), ","), format(len(last), ","))),

    P("2. The core idea: start from the Transfermarkt value, then correct it", h2),
    P("The model does not predict a fee from scratch. It predicts the <b>ratio between the fee and the player's "
      "Transfermarkt value</b>:"),
    P("<b>model price = Transfermarkt value x correction factor</b>"),
    P("The Transfermarkt value is a good first guess, but it is wrong in systematic ways: on its own it misses the "
      "real fee by a median of %s. The model learns when fees tend to come in above or below it. In the data, "
      "players aged 21 or under sold for a median of 1.7 times their Transfermarkt value, and players aged 28 "
      "to 31 for 0.6 times. Clubs in England sold at 1.2 times the value, clubs in the United States at 0.5 "
      "times. Every parameter below is used to work out this correction factor." % pc(M_TM.median_ape, 0)),
    P("Two reasons for this design:"),
    *bullets([
        "<b>It handles inflation.</b> Fees rise over time, but so do Transfermarkt values, so the ratio between "
        "them stays stable. A model that predicted the fee directly would underprice players in later years.",
        "<b>It keeps what Transfermarkt already knows.</b> Its values already reflect performance, reputation and "
        "potential as judged by thousands of users. The model only has to learn what they get wrong.",
    ]),

    P("3. Players by position: what the statistics show", h2),
    P("A striker and a goalkeeper are not judged on the same things. Before changing the model, the match "
      "statistics of the previous 365 days were compared across positions, on the %s transferred players who "
      "played at least %d minutes that year (%s)."
      % (format(len(pl), ","), MIN_MINUTES, ", ".join("%s %s" % (format(int((pl.position == p).sum()), ","), p.lower())
                                                       for p in POSITIONS))),
    table([["Average over the previous 365 days", *POSITIONS], *[stat_row(*s) for s in STATS]],
          [0.40, 0.15, 0.15, 0.15, 0.15]),
    Spacer(1, 8),
    P("The positions differ where expected: forwards score about seven times as often as defenders, goalkeepers "
      "play every minute, defenders and midfielders collect the most cards. What a team concedes is the same for "
      "every position, so on its own it says nothing about position, only about the team."),
    P("What matters is how each statistic goes with the fee <b>within</b> a position. The table gives the rank "
      "correlation with the fee: +1 means the statistic and the fee always rise together, 0 means no link."),
    table([["Link with the fee, within each position", *POSITIONS], *[corr_row(*s) for s in STATS]],
          [0.40, 0.15, 0.15, 0.15, 0.15]),
    Spacer(1, 8),
    *bullets([
        "<b>Goals and assists pay for forwards, less for midfielders, not for defenders or goalkeepers.</b>",
        "<b>Goals conceded and clean sheets matter most for goalkeepers.</b> They also show for the other "
        "positions, because they measure the strength of the team as much as the player.",
        "<b>Playing time and team results pay for everyone.</b> Minutes and points per game are the strongest "
        "links at every position.",
    ]),

    P("4. How the model adapts to the position", h2),
    P("Eleven position statistics were added to the model, which now uses %d parameters instead of 43. The model "
      "knows each player's position and detailed position, so it can give a different weight to each statistic "
      "depending on the position: goals count for a forward, goals conceded for a goalkeeper." % len(FEATS_V)),
    P("Two ways of adapting to position were tested on %s real transfers from 2024-2026, never seen in "
      "training. The table gives the typical error (lower is better)." % format(int(M_V.n), ",")),
    table([["Typical error by position", "Transfers", "Transfermarkt", "Before (43 parameters)",
            "One shared model with position statistics", "One separate model per position"],
           *[[p, format(int(r.n), ","), "%.3f" % r["Transfermarkt"], "%.3f" % r["43 parameters"],
              "%.3f" % r["with position statistics"], "%.3f" % r["one model per position"]]
             for p, r in pos_err.iterrows()]],
          [0.16, 0.12, 0.16, 0.18, 0.20, 0.18], bold_rows=(1,)),
    Spacer(1, 8),
    *bullets([
        "<b>One separate model per position is worse</b> (%.3f against %.3f). Each model learns from far fewer "
        "transfers, %s for goalkeepers, and loses what the positions have in common: age, club, country and "
        "Transfermarkt value work the same way for everyone."
        % (pos_err.loc["All", "one model per position"], pos_err.loc["All", "43 parameters"],
           format(int((tr.position == "Goalkeeper").sum()), ",")),
        "<b>One shared model with position statistics is the one kept.</b> The error goes from %.3f to %.3f. The "
        "gain is small and comes from midfielders and forwards; goalkeepers and defenders do not improve."
        % (pos_err.loc["All", "43 parameters"], pos_err.loc["All", "with position statistics"]),
        "<b>Why the gain is small.</b> The Transfermarkt value already reflects performance, match data exists "
        "for only 14 top divisions (about half of the transfers), and the statistics available for defenders "
        "and goalkeepers are team results, not individual actions.",
        "<b>Not available: challenges won, tackles, saves, expected goals prevented.</b> The data source records "
        "only goals, assists, cards and minutes for each game. Goals conceded, clean sheets and team results while "
        "the player was on the pitch stand in for them. Individual defensive and goalkeeping statistics would "
        "need a second data source.",
    ]),
    P("Position statistics that weigh most for each position (share of the correction):"),
    table([["Position", "Heaviest position statistics"], *[[p, top_for(p)] for p in POSITIONS]], [0.18, 0.82]),

    P("5. The parameters and why each is there", h2),
    P("The website model uses %d parameters in six groups. The chart shows how much each group contributes to the "
      "correction factor (share of the model's total gain)." % len(FEATS_V)),
    Spacer(1, 4),
    bars([(g, v) for g, v in group_gain.items()]),
    Spacer(1, 10),

    table([
        ["Group", "Parameters", "Why they are used"],
        ["Transfermarkt value and its history",
         "Last value at least 30 days old; change over 6 and 12 months; distance from the player's peak value; "
         "number of valuations so far; years since the first valuation; age of the last valuation.",
         "The value is the starting point. Its trend shows whether the player is on the way up or down. The number "
         "of valuations measures how closely the player is followed, and "
         "little-followed players are valued less reliably."],
        ["Player profile",
         "Age; position and detailed position; nationality; height; preferred foot.",
         "Age is the strongest single driver after the value: clubs pay for the years a player has left and for "
         "resale potential. Fees also differ by position and nationality for the same value."],
        ["Current club",
         "Country and league; points per game last season; squad value; average squad value in its league; "
         "transfer income and number of sales over 3 years; how its past sales compared with Transfermarkt "
         "values; whether it is a youth or reserve team.",
         "Who sells matters: the same Transfermarkt value does not fetch the same fee from every club or "
         "country. The club's past sales show whether it tends to sell above or below value. The country is the most important parameter in this group."],
        ["Past transfers of the player",
         "Time at the current club; fee the current club paid; last and highest fee ever paid for him; number "
         "of previous transfers.",
         "The fee a club paid is a reference point when it sells. Time at the club is also the only available stand-in for "
         "contract length: a player who arrived recently usually has a long contract left."],
        ["Playing time and output",
         "Minutes, games, goals and assists over the previous 365 days; goals plus assists per 90 minutes; "
         "minutes in league and in European competitions; minutes for the current club; change versus the year "
         "before; career totals. <b>New, for the positions:</b> goals and assists per 90 minutes separately; "
         "goals conceded, goals scored and goal difference of his team per 90 minutes he played; share of games "
         "without a goal conceded; points per game when he played; cards per 90 minutes; minutes per game; share "
         "of games played in full; career assists.",
         "Clubs pay for players who play and produce, and what counts as producing depends on the position. "
         "These add little (%s) because the Transfermarkt value already reflects performance, and because match "
         "data exists for only 14 top divisions." % pc(group_gain["Playing time and output"], 0)],
        ["Market conditions",
         "How fees compared with Transfermarkt values across the whole market over the previous year; summer "
         "or winter window.",
         "Captures periods when the whole market pays more or less than Transfermarkt values."],
    ], [0.2, 0.4, 0.4]),

    P("6. The weight of each parameter", h2),
    P("Share of the model's total gain carried by each parameter: how much it improves the fit while the model "
      "learns. The column Before is the model with 43 parameters, whose six group percentages were given "
      "earlier; Now is the model with position statistics. Each column adds up to 100%."),
    table(param_rows, [0.50, 0.26, 0.12, 0.12]),

    P("7. The weight of each parameter for each position", h2),
    P("The same %d parameters, for the players of each position. The measure is the share of the correction "
      "factor that comes from the parameter, averaged over the test transfers of that position (mean absolute "
      "SHAP value). It is not the same measure as in section 6, so the column All differs from it: a parameter "
      "can improve the fit a lot while moving each price a little. Each column adds up to 100%%; group totals "
      "are in bold." % len(FEATS_V)),
    table(pos_rows, [0.48, 0.10, 0.12, 0.10, 0.10, 0.10], bold_rows=bold),
    Spacer(1, 8),
    P("The weights are close across positions because most of the correction comes from parameters that work the "
      "same way for everyone. The differences are where expected: height and the minutes per game weigh more "
      "for goalkeepers, goals plus assists and the detailed position more for forwards."),

    P("8. What is deliberately left out", h2),
    table([
        ["Left out", "Why"],
        ["Anything about the buyer (its league, wealth, past spending)",
         "A player who is not being sold has no buyer. The version of the model that does know the buyer is more "
         "accurate (%s median error against %s) and is the one used to judge real transfers in the thesis."
         % (pc(M_A.median_ape, 0), pc(M_V.median_ape, 0))],
        ["Anything dated after the valuation date",
         "Transfermarkt often updates a value after a transfer, using the fee. Using it would let the model see "
         "the answer. Hence the 30-day rule on values, and performance counted only before the date."],
        ["Current international caps, current club, current value",
         "The source only gives today's figures, not those at the time of each past transfer."],
        ["Contract length",
         "Not available historically: Transfermarkt only shows current contracts. This is the most important "
         "missing parameter."],
        ["Challenges won, tackles, saves, expected goals",
         "Not in the data source. Team results while the player was on the pitch are used instead (section 4)."],
    ], [0.35, 0.65]),

    P("9. How the model learns", h2),
    *bullets([
        "<b>Algorithm.</b> Gradient boosting (LightGBM): about a thousand small decision trees, each correcting "
        "the errors of the previous ones. It finds on its own effects that are not straight lines, such as "
        "age, and combinations such as goals together with position.",
        "<b>One model per year, trained on the past only.</b> The price on 1 July 2023 comes from a model trained "
        "only on transfers before that date, and so on for each year from 2022 to 2026. A past price is what the "
        "model would have said at the time.",
        "<b>Sample rules.</b> Permanent transfers with a disclosed fee of at least EUR 50,000. Free transfers, "
        "undisclosed fees and suspected loan fees are excluded because they give no usable price.",
        "<b>Who is valued.</b> Every player given a Transfermarkt value in the 18 months before the date.",
    ]),

    P("10. How far to trust it", h2),
    table([
        ["Check on %s real transfers from 2024-2026, never seen in training" % format(int(M_V.n), ","), "Median error", "R2 (log)"],
        ["Transfermarkt value alone", pc(M_TM.median_ape, 0), "%.2f" % M_TM.r2_log],
        ["Website model before position statistics (43 parameters)", pc(M_OLD.median_ape, 0), "%.2f" % M_OLD.r2_log],
        ["Website model (no buyer information, %d parameters)" % len(FEATS_V), pc(M_V.median_ape, 0), "%.2f" % M_V.r2_log],
        ["Full model (knows the buyer)", pc(M_A.median_ape, 0), "%.2f" % M_A.r2_log],
    ], [0.6, 0.2, 0.2]),
    Spacer(1, 8),
    *bullets([
        "<b>The model range</b> on the website is an 80%% interval: in the year before each date, four real fees in "
        "five fell inside it. It runs from about %.1f to %.1f times the model price, which is wide. The model is "
        "reliable on averages over many players, much less on one player." % (lo, hi),
        "<b>Selection.</b> The model only learns from players who were sold. Players whom clubs refuse to sell may "
        "differ from them in ways the data does not show.",
        "<b>Top players come out above Transfermarkt.</b> This may be the real premium paid for stars, or an "
        "effect of selection. It has not been tested.",
        "<b>Lower leagues.</b> Errors are largest for cheap players and for clubs outside the 14 covered leagues.",
        "<b>Full statistics</b> on the reliability of the prices are in Statistics.pdf.",
    ]),

    PageBreak(),
    P("Part 2. The full model: the price for a given buyer", part),
    P("The model that also knows who is buying. Used on the website when a buying club is chosen.", sub),

    P("11. Two models, two questions", h2),
    table([
        ["", "Website model (Part 1)", "Full model (Part 2)"],
        ["Question it answers", "What fee would this player fetch if he were sold, whoever the buyer?",
         "What fee would this player fetch if this particular club bought him?"],
        ["Parameters", "%d: the player, his current club, the market" % len(FEATS_V),
         "%d: the same %d, plus %d about the buyer and the deal" % (len(FEATS_A), len(FEATS_V), len(DEAL))],
        ["Median error on %s unseen transfers" % format(int(M_V.n), ","), pc(M_V.median_ape, 0), pc(M_A.median_ape, 0)],
        ["Typical error (log scale)", "%.3f" % M_V.rmse_log, "%.3f" % M_A.rmse_log],
        ["R2 (log)", "%.2f" % M_V.r2_log, "%.2f" % M_A.r2_log],
        ["Where it is used", "Website, when no buyer is chosen; the five-year history of each player",
         "Website, when a buying club is chosen; judging real transfers in the thesis"],
    ], [0.26, 0.37, 0.37]),
    Spacer(1, 8),
    P("The full model is more accurate because the buyer explains a large part of the fee: the same player costs "
      "more when a rich club buys him. It cannot be the default on the website, because a player who is not "
      "being sold has no buyer. It works as soon as a buyer is named, which is what the buying club option does."),
    P("Apart from the extra parameters, the full model is built exactly like the website model: same transfers, "
      "same target (fee divided by Transfermarkt value), same algorithm, same position statistics."),

    P("12. The %d parameters about the buyer and the deal" % len(DEAL), h2),
    P("Together they carry %s of the full model's total gain." % pc(group_gain_a[BUYER], 0)),
    table([["Parameter", "Weight", "Why it matters"],
           *[[DEAL_PARAMS[f][0], pc(gain_a[f]), DEAL_PARAMS[f][1]] for f in gain_a.index if f in DEAL]],
          [0.44, 0.10, 0.46]),

    P("13. The weight of each parameter in the full model", h2),
    P("Share of the full model's total gain by group, then for each of the %d parameters. The last column "
      "gives the weight of the same parameter in the website model (section 6) for comparison. Buyer and deal "
      "parameters are in bold. Each column adds up to 100%%." % len(FEATS_A)),
    Spacer(1, 4),
    bars([(g, v) for g, v in group_gain_a.items()]),
    Spacer(1, 10),
    table(full_rows, [0.48, 0.26, 0.13, 0.13], bold_rows=bold_a),
    Spacer(1, 8),
    P("With the buyer known, the weight of the player's own profile and of his current club falls: part of what "
      "age, nationality or the selling country explained in the website model was in fact a guess about the kind "
      "of club that would buy him."),

    P("14. The buying club option on the website", h2),
    *bullets([
        "<b>What it shows.</b> For the chosen club, the fee the full model expects if that club bought each "
        "player on 1 July 2026. The table, the difference with Transfermarkt and the range all switch to that price.",
        "<b>Which clubs.</b> The %d clubs of the %d leagues covered by the data in the 2025/26 season. The buyer's figures "
        "(spending, squad value, points per game, past purchases) are those known on 1 July 2026."
        % (buy["n_clubs"], n_buy_leagues),
        "<b>How it is computed.</b> The full model is trained on the %s transfers before 1 July 2026, then run "
        "once for each club on all %s players: %s prices in total. The deal is assumed to be a normal summer "
        "transfer, not in the last week of the window and without a loan back."
        % (format(buy["n_train"], ","), format(buy["n_players"], ","), format(buy["n_clubs"] * buy["n_players"], ",")),
        "<b>How much the buyer changes the price.</b> For the typical player, the price with a buyer in the top "
        "tenth of clubs is %.1f times the price with a buyer in the bottom tenth."
        % buy["spread_p10_p90"],
        "<b>The range.</b> An 80%% interval, from the price divided by %.2f to the price multiplied by %.2f. It is "
        "narrower than the range of the website model because the buyer is no longer unknown." % (buy["range"], buy["range"]),
    ]),
    P("Example: %s. With no buyer chosen the website model gives %s. With a buyer:"
      % (buy["example_player"], eur(buy["example_no_buyer"]))),
    table([["Buying club", "League", "Price"], *[[n, lg, eur(v)] for n, lg, v in buy["example"][:5]],
           ["...", "", ""], *[[n, lg, eur(v)] for n, lg, v in buy["example"][5:][::-1]]], [0.4, 0.35, 0.25]),
    Spacer(1, 8),
    P("Limits specific to this option:"),
    *bullets([
        "<b>Unlikely pairs are extrapolations.</b> The model learned from transfers that happened, where rich "
        "clubs bought expensive players and small clubs cheap ones. The price of a low-value player for a top "
        "club, or of a star for a small club, is far from anything the model has seen and should not be trusted. "
        "The website marks a price with * when the player's Transfermarkt value is below the cheapest 5% or above "
        "the most expensive of the players bought by clubs of that league over the last five years.",
        "<b>It is a price, not a probability.</b> The option says what the fee would be if the deal happened, not "
        "whether the club could afford it or the player would go.",
        "<b>The seller is not modelled as negotiating.</b> A club that does not want to sell asks for more than "
        "the model price, whoever the buyer.",
        "<b>Only the latest date.</b> Buyer prices are computed for 1 July 2026; the five-year history shown for "
        "each player stays the model without a buyer.",
    ]),
]


def footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(INK2)
    canvas.drawString(2 * cm, 1.2 * cm, "Football player fee model - method note")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, "Page %d" % doc.page)


if __name__ == "__main__":
    SimpleDocTemplate("Player_Valuation_Method.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm, title="How the players are valued",
                      author="Football player fee model").build(story, onFirstPage=footer, onLaterPages=footer)
    print("written")
