# Football player fee model

A model that predicts the transfer fee of a football player from information available
before the transfer. It is the price benchmark for a Master's thesis on transfer pricing
in multi-club ownership networks (see `Thesis_Roadmap_MCO_Transfer_Pricing.pdf`).

## Results

Test set: 5,252 paid transfers from 2024 to mid-2026, never seen in training (train before 2023, tune on 2023).

| Model | RMSE (log) | Median % error | R² (log) |
|---|---|---|---|
| Naive: fee = Transfermarkt value | 0.942 | 50% | 0.64 |
| Hedonic linear (ridge) | 0.695 | 39% | 0.80 |
| **Model A**: LightGBM, all features | 0.676 | 37% | 0.81 |
| Model V: LightGBM, no buyer information | 0.763 | 42% | 0.76 |
| Model B: LightGBM, no Transfermarkt values | 0.750 | 43% | 0.77 |

- Model A is the arm's-length price for an observed transfer.
- Model B is the robustness check that does not rely on crowd-sourced values.
- Model V prices players who are not being sold, so it cannot use the buyer. The website uses it.
- 80% and 90% conformal prediction intervals reach their target coverage on the test set
  (80.5% and 90.0%), but they are wide: the 80% interval spans a factor of about 5.

Full tables are in `outputs/`, figures in `outputs/figures/`.

## Website

`docs/index.html` lists 19,435 active players with the model price next to the Transfermarkt
value, and both prices on 1 July of 2022 to 2026. Each yearly price comes from a model trained
only on transfers before that date. Open the file in a browser; no server is needed.

## Reproduce

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python src/download_data.py    # raw data into data/raw (about 75 MB)
.venv/Scripts/python src/build_dataset.py    # one row per paid transfer
.venv/Scripts/python src/tune.py             # optional: Optuna search
.venv/Scripts/python src/train_model.py      # benchmarks, models, intervals, SHAP, price gaps
.venv/Scripts/python src/value_players.py    # player values 2022-2026 and website data
```

## Method in brief

- Sample: permanent transfers with a disclosed fee of at least EUR 50k, 2014 to July 2026.
- Target: log(fee / last Transfermarkt value), with the value at least 30 days old.
- Features: age, position, past fees, time at the club, minutes and goals over the previous
  365 days, league and country of both clubs, squad values, each club's past spending.
  Everything is measured before the transfer date.
- `outputs/price_gaps.parquet` holds the cross-fitted gap (actual minus predicted log fee)
  for every transfer, with folds grouped by player.

## Known limits

- No contract length at the time of transfer; the source only has current contracts.
- No transfer-type field, so loan fees are removed with a heuristic.
- Match data covers 14 top divisions; 44% of transferred players have no performance data.
- Multi-club-ownership transfers are not yet excluded from training: the ownership registry
  does not exist yet.

## Data

- [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) (dcaribou), snapshot of 6 July 2026.
- [football-transfers-data](https://github.com/d2ski/football-transfers-data) (d2ski), for club countries only.
