"""Tune LightGBM on the 2023 validation year (train < 2023). Writes outputs/best_params.json.

Run from the project root:  python src/tune.py
"""
import json

import numpy as np
import optuna
import pandas as pd

from modeling import BASE_PARAMS, FEATS_A, OUT, fit_lgb, prepare

N_TRIALS = 40


def main():
    OUT.mkdir(exist_ok=True)
    df = prepare(pd.read_parquet("data/processed/transfers_model.parquet"))
    df = df[df.log_mv_pre.notna()].copy()
    df["log_ratio"] = df.log_fee - df.log_mv_pre
    tr, va = df[df.split == "train"], df[df.split == "valid"]

    def objective(trial):
        p = dict(BASE_PARAMS,
                 num_leaves=trial.suggest_int("num_leaves", 6, 64, log=True),
                 min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 150, log=True),
                 feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
                 bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                 lambda_l2=trial.suggest_float("lambda_l2", 1e-2, 30, log=True),
                 cat_smooth=trial.suggest_float("cat_smooth", 5, 100, log=True))
        m = fit_lgb(tr, va, FEATS_A, "log_ratio", p=p)
        return np.sqrt(((m.predict(va[FEATS_A]) - va.log_ratio) ** 2).mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial({k: BASE_PARAMS[k] for k in ["num_leaves", "min_data_in_leaf", "feature_fraction",
                                                     "bagging_fraction", "lambda_l2", "cat_smooth"]})
    study.optimize(objective, n_trials=N_TRIALS)
    (OUT / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    print("default RMSE on 2023: %.4f | tuned: %.4f" % (study.trials[0].value, study.best_value))
    print(study.best_params)


if __name__ == "__main__":
    main()
