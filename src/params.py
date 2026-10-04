"""Plain name and group of every parameter of the website model, shared by the method note and the website."""
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
