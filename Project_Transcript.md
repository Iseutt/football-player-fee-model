# Project transcript

A running log of the football player fee model project, kept as a follow-up for the thesis.
Each entry says what was asked, what was done, the main numbers, the files touched, and what
was decided or left open. Newest entries are at the bottom. Every later change to the project
is added here.

Conventions used throughout:

- **Test set**: the 5,252 paid transfers from 2024 to mid-2026, never used in training
  (train before 2023, tune on 2023).
- **Typical error**: root mean squared error of ln(fee / price). 0.69 means a factor of two.
- **Website model** (Model V): no information about the buyer. **Full model** (Model A): also
  knows the buyer and the deal. **Model B**: no Transfermarkt values at all.

---

## Step 0 — Starting point (3 October 2026)

This log was started on 4 October 2026. What existed before it, as recorded in the repository
(first commit `25ec6c1`, 3 October 2026) and its README; the detailed reasoning of this first
phase is not recorded here.

- **Goal**: a model that predicts the transfer fee of a player from information available before
  the transfer, as the arm's-length price benchmark for a Master's thesis on transfer pricing in
  multi-club ownership networks (`Thesis_Roadmap_MCO_Transfer_Pricing.pdf`).
- **Data**: transfermarkt-datasets (dcaribou), snapshot of 6 July 2026; d2ski
  football-transfers-data for club countries only.
- **Sample**: 16,345 paid transfers from 2014 to July 2026; after removing suspected loan fees,
  missing birth date or position, and fees under EUR 50,000: 15,936, of which 15,138 have a
  Transfermarkt value dated at least 30 days before the transfer.
- **Design**: the target is ln(fee / last Transfermarkt value). Gradient boosting (LightGBM),
  tuned with Optuna. All features measured before the transfer date.
- **Models**: A (all features), B (no Transfermarkt values), V (no buyer information, used to
  value players who are not for sale), plus a naive benchmark (fee = Transfermarkt value), an OLS
  and a ridge regression.
- **Results at that point** (test set): Transfermarkt 0.942, ridge 0.695, Model A 0.676,
  Model V 0.763, Model B 0.750. 80% and 90% conformal intervals reached their target coverage.
- **Outputs**: cross-fitted price gaps for every transfer (`outputs/price_gaps.parquet`), SHAP
  figure, values of every active player on 1 July of 2022 to 2026 with one model per year trained
  on the past only (`outputs/player_values.parquet`), and a website (`docs/`) published with
  GitHub Pages.
- **Known limits recorded then**: no contract length at the time of transfer; loan fees removed
  with a heuristic; match data for 14 top divisions only (44% of transferred players have none);
  multi-club-ownership transfers not yet excluded from training.
- A first version of `Player_Valuation_Method.pdf` existed (method, six groups of parameters).

## Step 1 — How far can a price be trusted? (4 October 2026)

**Asked**: a statistics PDF, updated continuously, with the variance, standard deviation and
standard error of Transfermarkt prices and of the model, and the real range around a price
(example: is a player priced at 10m really worth 5 to 15?).

**Done**: `src/make_stats_pdf.py` writes `Statistics.pdf`; every number is recomputed from the
data each time it runs. The error is e = ln(real fee / stated price) on the test set.

**Findings** (figures as first computed, later steps changed the model slightly):

- Transfermarkt: mean -0.133 (fees about 12% below its values), standard deviation 0.933
  (a factor of 2.54), standard error of the mean 0.013.
- Website model: mean +0.012 (no measurable bias), standard deviation 0.763 (a factor of 2.14).
  It removes 33% of the variance of Transfermarkt's error.
- A player priced at 10m (transfers priced 5m to 20m): 80% of real fees fall between 3.8m and
  18.0m for Transfermarkt, 5.3m and 20.0m for the model. The fee is inside 5m to 15m for 65% of
  players with both sources.
- The spread shrinks as the price rises. The standard deviation in euros is unusable (a few fees
  at several times the price inflate it); percentile ranges are used instead.
- Limit: a fee is not the true value. Part of the spread is deal noise, so these are upper
  bounds on the pricing error.

**Files**: `src/make_stats_pdf.py`, `Statistics.pdf`, `outputs/statistics_summary.csv`,
`outputs/figures/stats_ranges_10m.png`, `outputs/figures/stats_error_distribution.png`.

## Step 2 — Skill or luck? Significance tests (4 October 2026)

**Asked**: prove with statistics whether Transfermarkt prices and model values are significant
or mainly luck; the aim is a model that is "statistically right".

**Done**: sections 7 to 9 of `Statistics.pdf`.

- **Against luck**: the prices were shuffled among the players 5,000 times. Shuffled prices
  explain less than nothing (R2 about -68% at best); Transfermarkt explains 64% of fee
  differences, the website model 76%. p < 0.001 for both.
- **Is the model's correction to Transfermarkt luck?** No. The fee lands on the side of the
  Transfermarkt value the model predicted for 71% of transfers (50% for a coin). The model beats
  Transfermarkt in 2024, 2025 and 2026 separately (t of at least 6.7).
- **Definition of "statistically right"** (proposed, not yet confirmed by the author): passing
  eight tests at the 5% level: (1) skill, not luck; (2) no overall bias; (3) fee rises one for
  one with price; (4) no bias at any price level; (5) beats Transfermarkt; (6) beats it every
  year; (7) the 80% range holds 80% of fees; (8) the 90% range holds 90%.
- **Scorecard then**: Transfermarkt 1 of 4, website model 7 of 8 (its 90% range held 88.9%).
- Correction made to Step 1: the apparent underpricing of the model around 10m is not
  significant on the mean (+4%, p = 0.24).
- p-values treat transfers as independent, which matters only for borderline results.

**Open questions put to the author**: is this the right definition of "statistically right" or
is there a precision target; which model should be optimised; is the real fee the right
reference for the truth.

## Step 3 — Adapting the model to the position (4 October 2026)

**Asked**: analyse statistics by position and adapt the model so that each position is judged on
what matters for it (goals for a striker, duels won or xG prevented for defenders and
goalkeepers); give the percentage of each of the 43 parameters, and of any new ones by position.

**Done**:

- **Position analysis** on 5,145 transferred players with at least 900 minutes in the previous
  year. Rank correlation with the fee within each position: goals per 90 minutes +0.26 for
  forwards, +0.09 for defenders, -0.07 for goalkeepers; goals conceded by the team -0.28 for
  goalkeepers; minutes and points per game matter at every position.
- **11 position statistics added** (the website model goes from 43 to 54 parameters): goals per
  90, assists per 90, goals conceded / scored / goal difference of his team per 90 minutes he
  played, clean-sheet share, points per game when he played, cards per 90, minutes per game,
  share of games played in full, career assists.
- **Two designs tested** on the test set: one shared model that knows the position (kept), and
  one separate model per position (rejected).

| Typical error | Transfermarkt | Before (43) | Shared + position statistics | One model per position |
|---|---|---|---|---|
| All | 0.942 | 0.763 | 0.760 | 0.796 |
| Goalkeeper | 1.019 | 0.831 | 0.833 | 0.923 |
| Defender | 0.990 | 0.781 | 0.780 | 0.820 |
| Midfield | 0.938 | 0.759 | 0.755 | 0.792 |
| Attack | 0.893 | 0.739 | 0.734 | 0.757 |

- The gain is small and borderline significant (t = 2.0), and comes from midfielders and
  forwards. A position-specific correction stage on top of the shared model was also tried and
  added nothing (t = 0.5).
- **Not available in the data**: challenges won, tackles, saves, xG prevented. The source only
  records goals, assists, cards and minutes per game, so defenders and goalkeepers are judged on
  team results while they were on the pitch.
- The position statistics were added to all three models (A: 0.676 to 0.674; B: 0.750 to 0.749).

**Files**: `src/features.py`, `src/modeling.py`, `src/train_model.py`, `src/make_method_pdf.py`
(rewritten to read its numbers from `outputs/`), `Player_Valuation_Method.pdf` (sections 3, 4,
6, 7), `outputs/feature_importance_v.csv`, `outputs/feature_importance_v_43.csv`,
`outputs/feature_share_by_position.csv`, `outputs/position_errors_test.csv`.

## Step 4 — Would StatsBomb data make the model much better? (4 October 2026)

**Asked**: an opinion, no change to the project.

**Answer given**: better, probably not much better. Performance is already mostly contained in
the Transfermarkt value (all playing-time and output parameters carry about 9% of the model);
the remaining error is mostly about the deal (contract length, buyer, urgency); coverage would
limit the gain. It would help most for defenders and goalkeepers, for Model B, and for spotting
players Transfermarkt misjudges. Contract length at the time of transfer would likely help more.

**Open**: which leagues and seasons of StatsBomb data the author can access.

## Step 5 — Which model is used, and the full model on the website (4 October 2026)

**Asked**: why the full model is not used; where its percentages are; then a separate part on the
full model in the method PDF and a website option to see the price if a given club buys.

**Done**:

- **Explanation**: the full model needs a buyer, and a player who is not for sale has none. Its
  14 extra parameters describe the buying club and the deal, and carry about 30% of its weight
  (buyer's spending over 3 years 9.3%, buyer's country 7.5%, buyer's past purchases compared
  with Transfermarkt values 6.3%).
- **`Player_Valuation_Method.pdf`, Part 2** (sections 11 to 14): the two models side by side,
  the 14 buyer and deal parameters with their weight, the weight of all 68 parameters, and how
  the buying club option works.
- **Website**: a buying club can be chosen among 508 clubs of the leagues covered in 2025/26.
  The full model is trained on all transfers before 1 July 2026 and run once per club on all
  19,435 players (about 9.9 million prices), stored one file per club in `docs/buyers/`.
- For the typical player, a buyer in the top tenth of clubs gives a price 3.8 times that of a
  buyer in the bottom tenth.
- **Limit**: unlikely pairs (a cheap player for a top club, a star for a small club) are
  extrapolations. The model learned from deals that happened.

**Files**: `src/value_players.py`, `src/make_method_pdf.py`, `docs/index.html`,
`docs/buyers.js`, `docs/buyers/`, `outputs/buyer_option.json`.

## Step 6 — Publication and website fixes (4 October 2026)

**Asked**: update GitHub; then, because the site looked unchanged, go through the code and test.

**Done**: everything pushed to `main` (GitHub Pages serves `docs/`). The site looked unchanged
because GitHub took several minutes to publish 508 new files. Testing in Chrome found and fixed:

- A browser could mix the new page with cached old data. Data files now carry a version stamp.
- Absurd prices for unlikely deals were shown without warning. A buyer price is now marked as a
  rough guess when the player's Transfermarkt value is below the cheapest 5% or above the most
  expensive of the players bought by clubs of that league over five years.
- Leagues sharing a name (Bundesliga, Superliga, Premier Liga) now show the country.
- `.gitattributes` added so that PDFs are treated as binary.

## Step 7 — Website redesign (4 October 2026)

**Asked**: a modern, clean, easy-to-use website with club logos and colour.

**Done**: `docs/index.html` rewritten. Search bar, position tabs, league filter, a searchable
buying-club picker with crests, player cards with photo, position tag and difference chip, a
player panel with the five-year chart and the price for the chosen club, phone layout, light and
dark themes. Site data now carries each player's club id and photo file name.

**To keep in mind**: crests and photos are loaded from Transfermarkt's image servers, not stored
in the repository. They are Transfermarkt's images.

## Step 8 — Why was Estêvão valued at EUR 166m? Fix for expensive players (4 October 2026)

**Asked**: explain the price of Estêvão; then measure how widespread the problem is and fix it.

**Diagnosis**: his Transfermarkt value is EUR 80m and the model multiplied it by 2.07. No single
parameter was extreme: about a dozen small bonuses multiplied together (age 19: x1.30; sold from
England: x1.17; fee his club paid: x1.14; Brazilian: x1.12; and others), against x0.53 for the
high value itself. Comparable real deals do not support this: the 34 transfers of players aged
20 or under valued at 30m or more went for a median of 0.97 times the value, and Chelsea bought
him in 2025 for 45m on a value of 60m.

**How widespread**: the model's correction is reliable for cheap players and too strong for
expensive ones. On the test set, the share of the predicted correction that shows up in the real
fee was 0.98 under 5m, 0.74 from 5m to 20m, 0.57 from 20m to 40m and 0.43 above 40m. Where the
model announced at least 1.3 times the value for a player worth 20m or more (36 transfers), it
announced 1.55 times and the real fees came in at 1.30 times.

**Fix (kept)**: the correction is scaled down as the Transfermarkt value rises: untouched below
EUR 5m, multiplied by 0.65 above EUR 40m, in between in proportion to the log value. The three
numbers are estimated on training transfers only, from out-of-sample (cross-fitted) predictions,
and again for each yearly model and for the full model (0.72 above EUR 80m).

**Result on the test set**:

| Transfermarkt value | Transfers | Typical error before | after | Transfermarkt alone |
|---|---|---|---|---|
| under 5m | 3,983 | 0.803 | 0.803 | 1.013 |
| 5m to 20m | 994 | 0.630 | 0.626 | 0.713 |
| 20m to 40m | 205 | 0.516 | 0.498 | 0.538 |
| over 40m | 70 | 0.402 | 0.374 | 0.388 |

- Gain in squared error over the whole test set: t = 4.5. Website model 0.760 to 0.758, full
  model 0.674 to 0.673.
- Before the fix the model was worse than Transfermarkt above 40m; after it, slightly better.
- Estêvão: EUR 166m before, EUR 126m after (range 88m to 179m).

**Also changed in this step**:

- **Ranges by price level**: the width of the 80% range is now set separately for four levels of
  Transfermarkt value, because errors are much smaller for expensive players. For 1 July 2026
  the price is divided and multiplied by 3.0 under 1m, 2.4 from 1m to 5m, 2.0 from 5m to 20m and
  1.4 above 20m (it was 2.4 for everyone).
- **Bug fixed in the buying club option**: buyer prices were computed from the Transfermarkt
  value at least 30 days old but displayed as a multiple of the latest value. Players revalued
  in June 2026 were shown at the wrong price. Both now use the same base.
- **Scorecard now**: website model 8 of 8; full model 6 of 8 (a small bias under 1m, and its 80%
  range holds 81.2% of fees, slightly too wide); Transfermarkt 1 of 4.
- Current statistics (test set): website model mean +0.007, standard deviation 0.758; full model
  mean -0.010, standard deviation 0.673; Transfermarkt mean -0.133, standard deviation 0.933.

**Files**: `src/modeling.py` (scale-down, ranges by level), `src/train_model.py`,
`src/value_players.py`, `src/make_stats_pdf.py` (section 10), `src/make_method_pdf.py`,
`docs/index.html`, `outputs/expensive_player_scaling.csv`, this file.

## Step 9 — Club picker by country, clubs on the player chart, wrong clubs fixed (4 October 2026)

**Asked**: (1) a more attractive club search, by country, the eight biggest leagues first, the
other countries in alphabetical order, each country opening onto its divisions; (2) on the
player chart, the club the player was at each year, with logo and name; (3) check the club of
every listed player, because some were wrong (Balerdi shown at Marseille instead of Roma,
Jacquet at Rennes instead of Liverpool).

**Done**:

- **Club picker**: countries with their flag; the countries of the eight richest leagues first
  (by average squad value: England, Spain, Germany, Italy, France, Brazil, Saudi Arabia,
  Portugal), then the others alphabetically. A country opens onto its league (with its logo)
  and its clubs (with crests). Typing searches clubs, leagues and countries.
- **Only first divisions exist in the data** (30 leagues in 30 countries), so each country shows
  one league. Ligue 2 or the Championship cannot be buyers: the source has no club data for them.
- **Player chart**: under each year, the crest and name of the club the player was registered
  with on 1 July of that year.
- **Wrong clubs, cause found**: a player's club was taken from his last transfer strictly
  before the valuation date, so the 1,195 transfers dated exactly 1 July 2026 were ignored
  (Jacquet to Liverpool, Gordon to Barcelona, Højlund to Napoli, Hincapié to Arsenal). For a
  valuation, a move dated the same day now counts. Training rows are unchanged (verified).
- **Free agents**: 858 players whose last move is to "Without Club" on the valuation date
  (for example Konaté) are labelled as free agents, with their last real club. Players whose
  last move is "Retired" or "Career break" are no longer listed, nor is Diogo Jota (deceased).
  The list goes from 19,435 to 19,326 players.
- **Checked against the players table of the source**: it is less reliable than the transfer
  table (it still shows Xavi Simons at Leipzig and Lookman at Atalanta), so it is not used.
- **Not fixable from the current data**: Balerdi. The snapshot of 6 July 2026 has no transfer of
  his after 2021 and still lists him at Marseille. Any move made or recorded after 6 July 2026
  needs a new download of the data, which would change every number of the project.
- A player on loan on 1 July is shown at the club he was loaned to; a loan that ended on 30 June
  counts as ended.
- The model prices of the players who moved on 1 July changed, because their selling club is now
  the new one.

**Files**: `src/features.py` (club on the valuation date), `src/value_players.py`,
`docs/index.html`, `docs/data.js`, `docs/buyers.js`, `docs/buyers/`.

**Open then**: refresh the data to a later snapshot or not (answered in Step 10: no newer snapshot exists).

## Step 10 — Clubs after the data snapshot, league picker (4 October 2026)

**Asked**: check every player again, free agents included, because a free agent may already
have signed elsewhere (Konaté with Real Madrid); give the league filter the same look as the
buying-club picker.

**What the check found**:

- The source has no newer data. Its transfers file, downloaded again on 4 October 2026, is
  identical to the local one and stops on 6 July 2026. So every move made after that date is
  missing: the whole summer 2026 window after 6 July, including Konaté to Real Madrid and
  Balerdi to Roma (a loan, 30 August 2026).
- None of the 858 free agents has a later move recorded in the data.

**Done**:

- **Manual updates file** `data/manual_transfers.csv`: moves made after the snapshot, each with
  its date, fee, kind (free, loan, permanent) and source. They only tell where a player is
  today. They are never used as training transfers, and the training table is unchanged apart
  from one row (see below).
- **24 moves entered**, each checked on the web:
  - free agents: Konaté and Bernardo Silva to Real Madrid, Stones to Inter, Salah to
    Trabzonspor, Lewandowski to Chicago Fire, Goretzka to Aston Villa;
  - Balerdi, on loan from Marseille to Roma;
  - Cucurella to Real Madrid;
  - the summer's biggest transfers that two published top-20 lists agree on: Enzo Fernández,
    Anderson, Bouaddi and Ndiaye to Manchester City; Barcola to Liverpool; Rogers to Chelsea;
    Diomande to Real Madrid; Tonali, Mateus Fernandes and Savinho to Tottenham; Bruno Guimarães
    to Arsenal; Baleba to Manchester United; Gonçalo Ramos to AC Milan; Martinelli and
    Summerville to Al-Hilal; Reijnders to Al-Qadsiah.
- **Left out because confirmed by one source only**: Vlahović to Beşiktaş, Brandt to Ajax, Rodri
  to Barcelona, Jackson to Aston Villa, van Hecke to Tottenham, Lacroix to Chelsea.
- **The latest list now shows today's club**: for 1 July 2026, moves recorded after that date
  count for the player's club, and his price is computed as if he were sold from that club.
  Earlier years are unchanged.
- Fees given in pounds were converted at 1.155 (the rate implied by the Gordon deal). Most
  dates are approximate (end of the window). Barcola's fee differs between the two lists.
- **League filter**: same picker as the buying club, one row per league with the flag of its
  country, the eight biggest first, then by country in alphabetical order, with the number of
  players and a search box.
- **One training row changed**: transfers on the same day for the same player are now kept in
  file order (needed for the manual file). One transfer out of 15,936 (Fer López, 1 July 2025)
  gets a different "time at the club". All models and both PDFs were rerun on this table.

**Limit to keep in mind**: outside these 24 players, any move made after 6 July 2026 is still
missing, and there are hundreds. A complete fix needs a source that covers the full summer 2026
window. New rows can be added to `data/manual_transfers.csv` at any time.

**Files**: `data/manual_transfers.csv` (new), `src/features.py`, `src/build_dataset.py`,
`src/value_players.py`, `docs/index.html`, site data.

## Step 11 — Transfers of each player on the website (4 October 2026)

**Asked**: show somewhere the value of the transfer when a player moved to another club, or
say if it was a loan.

**Done**: each player has a "Transfers" list (first in the panel under the chart, then on his own
page, see Step 12): his last moves, newest first, each with the date, the two clubs with their crests, and the fee or the
kind of move.

- **The source has no field for the kind of move**, so it is worked out from the moves
  themselves: a move followed by a free return to the same club is a loan; the return is the
  end of the loan; a paid move followed by a free return more than 45 days later, at a price
  far below the player's value, is a loan with a fee; a move to "Without Club" is a contract
  that ended; a move with no fee and no return is a free transfer; a move with no fee published
  is shown as "fee not disclosed"; a move from or to a youth or reserve team without a fee is
  shown as such.
- A loan still running is recognised because the source already holds its scheduled return.
- The 24 moves entered by hand (Step 10) show their own description, for example Balerdi:
  "Loan (fee EUR 1m, option to buy EUR 16m)".
- 117,025 moves for 19,293 players: 13,941 transfers with a fee, 18,175 free transfers,
  22,072 loans, 20,427 ends of loan, 30,031 youth or reserve moves, 9,613 undisclosed fees,
  2,577 contracts ended.
- The moves are stored with the player profiles (`docs/players/`, Step 12).

**Limit**: these labels are deductions, not facts from the source. A free transfer and a loan
whose return is not recorded cannot be told apart.

**Files**: `src/value_players.py` (`player_moves`), `docs/index.html`.

## Step 12 — A profile page for each player (4 October 2026)

**Asked**: clicking a player should open a separate page, in the spirit of a Transfermarkt
player profile, with all the statistics the dataset has on him and the chart, so that it is
clear how his value is estimated.

**Done**: `docs/player.html`. A click on a player in the list opens his page (the chosen
buying club is carried along, and kept when going back). The page shows:

- **Header**: photo, club with crest, position, age and date of birth, height, foot,
  nationality, time at the club, fee the club paid; the model price with its range, the
  Transfermarkt value, and the price for the chosen buying club.
- **How the model gets to this price**: the Transfermarkt value, then one line per group of
  parameters with what it multiplies the price by and the running price, down to the model
  price. Each group opens onto its parameters, with the player's own figure and its effect.
  Below, the four figures that raise his price most and the four that lower it most.
- **Value over time**: the five-year chart with the club of each year, and the yearly table.
- **Playing time and output**: the last 365 days (games, minutes, goals, assists, per 90),
  the team's results with him on the pitch, and career totals. A player without match data
  gets a sentence saying so.
- **Transfers**: his moves with fees and kinds (Step 11).

**How the effects are computed**: SHAP values of the website model trained on all transfers
before 1 July 2026, one per parameter and per player, multiplied by the scale-down for expensive
players (Step 8). They add up exactly to the model's correction, so the lines of the page
multiply to the model price. They describe what the model does for this player; they are not
proof of what clubs pay for.

Example, Estêvão (Transfermarkt value EUR 80m, model price EUR 124m): Transfermarkt value and
its history -28%, player profile +29% (age 19: +18%), current club +22%, past transfers +21%,
playing time and output +16%, market conditions -2%.

**Technical notes**: parameter names and groups moved to `src/params.py`, shared by the method
note and the website. Profiles are stored in 256 files (`docs/players/`, 17.7 MB in total),
one of which is loaded when a player is opened. The inline panel of the list was removed.

**Files**: `docs/player.html` (new), `docs/index.html`, `docs/players/` (new, replaces
`docs/moves/`), `src/params.py` (new), `src/value_players.py`, `src/make_method_pdf.py`.

## Step 13 — Lighter player profile, in the style of Transfermarkt (4 October 2026)

**Asked**: the profile page had too much information, not all of it understandable; base it on
the Transfermarkt interface and make it lighter.

**Done**: `docs/player.html` redone.

- **Header**, as on a Transfermarkt profile: photo, name, club, a short list of facts (date of
  birth and age, position, height, foot, citizenship, date joined, fee paid) and a value box on
  the right with the model price, its range, the Transfermarkt value and the price for the
  chosen buying club.
- **Market value**: the five-year chart with the club of each year.
- **Transfer history**: a table with date, from, to and fee or kind of move. Youth and reserve
  team moves are hidden.
- **Why this price**: only the five figures that move the price most, in plain words (for
  example "Age: 19 years old, +18%"). The full calculation by group of parameters is folded
  away under one line.
- **Last 365 days**: one short table, adapted to the position: goals and assists for forwards
  and midfielders, games without conceding and goals conceded for defenders and goalkeepers.

**Removed from the page**: the list of all 54 parameters with their effect, the 17 statistic
tiles, the bars, the yearly table and the long explanations. The underlying data is unchanged.

**Files**: `docs/player.html`.

## Step 14 — Club pages (5 October 2026)

**Asked**: a part of the site reserved to clubs, with the valuation of their team, their last
results, their last transfers according to the model and to Transfermarkt, and a statistic
that says whether the club usually sells below or above the set prices.

**Done**:

- **Clubs list** (`docs/clubs.html`): the 508 clubs of the leagues covered in 2025/26, with the
  number of listed players, the squad value according to the model and to Transfermarkt, the
  difference, and the club's selling habit in a few words. Sortable and searchable.
- **Club page** (`docs/club.html`): crest, league, league position and record of the last
  season, squad value; the squad with each player's Transfermarkt value and model price; the
  last 12 departures and 12 arrivals with the fee, the player's Transfermarkt value before the
  move and the fee the model expected; the last six results; and two boxes, "When it sells"
  and "When it buys".
- Navigation: Players / Clubs at the top of the list pages; the club of a player links to its
  club page; players in a club page link to their profile.

**How the selling and buying record is measured**:

- Sample: the club's paid transfers since 2014 with a Transfermarkt value before the deal (the
  transfers of the model's dataset). At least 8 are required to say anything; 364 clubs of the
  508 have that many sales.
- **Compared with other clubs**: for each deal, ln(fee / Transfermarkt value) minus the market
  median of the same year, so that years when the whole market paid more or less do not count
  for or against a club. The figure shown is the club's typical (median) gap. It is tested with
  a Wilcoxon signed-rank test at 5%: a few extreme deals do not decide the answer.
- **Compared with the model price**: the same with ln(fee / model price), where the model price
  is the cross-fitted full model (it never saw the deal it prices). The full model already
  knows how the club sold over the previous three years, so this gap is what is left once that
  habit is taken into account. It is smaller by construction.
- Result: 21 clubs sell significantly above what other clubs get and 48 significantly below;
  the others have no clear habit or too few sales. Example: Arsenal, 37 paid sales, typical fee
  0.77 times the Transfermarkt value, 34% below other clubs in the same years (significant),
  8% below the model price; when it buys, 21% above other clubs.
- With 364 clubs tested at 5%, about 18 would come out "significant" by chance alone. The 69
  found are well above that, but an individual club's verdict is not certain.

**Squad value**: the sum of the model prices (no buyer) of the players listed on the site at
their club of today. Players on loan elsewhere on that date are counted at the club they are
loaned to.

**Files**: `src/club_data.py` (new, run after `value_players.py`), `docs/clubs.html`,
`docs/club.html`, `docs/site.css`, `docs/clubs.js`, `docs/clubs/` (new); `docs/index.html`
and `docs/player.html` (navigation).

---

## Open points

- Confirm or change the definition of "statistically right" (Step 2).
- Contract length at the time of transfer is the most important missing parameter.
- Individual defensive and goalkeeping statistics need a second data source (StatsBomb, FBref).
- Multi-club-ownership transfers are not yet excluded from training.
- Only sold players are observed: prices for players who are not for sale cannot be tested.
- The buying club option extrapolates for unlikely pairs of club and player.
