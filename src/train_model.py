"""Train and evaluate the arm's-length fee model.

Temporal split: train < 2023, tune on 2023, test from 2024.
Run from the project root (after build_dataset.py, optionally tune.py):  python src/train_model.py
"""
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modeling import (CAT, FEATS_A, FEATS_B, FEATS_V, FEATS_V_OLD, OUT, POSITIONS, conformal_q, fit_lgb, metrics,
                      params, prepare)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = OUT / "figures"


def lgb_predict(df, feats, target, offset):
    """Tune rounds on 2023, refit on train+valid, predict the test set."""
    tr, va, te = df[df.split == "train"], df[df.split == "valid"], df[df.split == "test"]
    rounds = fit_lgb(tr, va, feats, target).best_iteration
    model = fit_lgb(pd.concat([tr, va]), None, feats, target, rounds)
    return model, rounds, model.predict(te[feats]) + (te[offset] if offset else 0)


def hedonic(df, feats):
    num = [f for f in feats if f not in CAT]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT)])
    fit = df[df.split != "test"]
    m = make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(fit[feats], fit.log_fee)
    return m.predict(df.loc[df.split == "test", feats])


def segments(te, pred, naive):
    te = te.assign(err=pred - te.log_fee, err_naive=naive - te.log_fee)
    te["fee_band"] = pd.cut(te.transfer_fee, [0, 5e5, 2e6, 1e7, 3e7, 1e10], labels=["<0.5m", "0.5-2m", "2-10m", "10-30m", ">30m"])
    te["value_band"] = pd.cut(te.mv_pre, [0, 5e5, 2e6, 1e7, 3e7, 1e10], labels=["<0.5m", "0.5-2m", "2-10m", "10-30m", ">30m"])
    te["age_band"] = pd.cut(te.age, [0, 21, 24, 28, 50], labels=["<=21", "21-24", "24-28", "28+"])
    big5 = ["GB1", "ES1", "IT1", "L1", "FR1"]
    for side in ["buy", "sell"]:
        lg = te[f"{side}_league"].astype(str)
        te[f"{side}_group"] = np.select([lg.isin(big5), lg == "OTHER"], ["big5", "not covered"], "other covered")
    rows = []
    for col in ["fee_band", "value_band", "age_band", "buy_group", "sell_group", "position"]:
        for k, g in te.groupby(col, observed=True):
            rows.append({"segment": col, "value": k, "n": len(g), "rmse_log": np.sqrt((g.err ** 2).mean()),
                         "rmse_naive": np.sqrt((g.err_naive ** 2).mean()), "bias_log": g.err.mean(),
                         "median_ape": np.median(np.abs(np.exp(g.err) - 1))})
    return pd.DataFrame(rows)


def intervals(a, rounds):
    """Prediction intervals, calibrated on 2023 and checked on the test set (model trained on < 2023 only)."""
    tr, va, te = a[a.split == "train"], a[a.split == "valid"], a[a.split == "test"]
    m = fit_lgb(tr, None, FEATS_A, "log_ratio", rounds)
    res_va, res_te = (va.log_ratio - m.predict(va[FEATS_A])).abs(), (te.log_ratio - m.predict(te[FEATS_A])).abs()
    rows = []
    for cov in (0.8, 0.9):
        q = conformal_q(res_va, cov)
        rows.append({"method": "split conformal", "target": cov, "coverage_test": (res_te <= q).mean(),
                     "median_width_factor": np.exp(2 * q)})
        # conformalised quantile regression: width adapts to the transfer
        lo, hi = (fit_lgb(tr, None, FEATS_A, "log_ratio", rounds, params(objective="quantile", alpha=al))
                  for al in ((1 - cov) / 2, 1 - (1 - cov) / 2))
        score = np.maximum(lo.predict(va[FEATS_A]) - va.log_ratio, va.log_ratio - hi.predict(va[FEATS_A]))
        q = conformal_q(score, cov)
        l, h = lo.predict(te[FEATS_A]) - q, hi.predict(te[FEATS_A]) + q
        rows.append({"method": "conformal quantile regression", "target": cov,
                     "coverage_test": ((te.log_ratio >= l) & (te.log_ratio <= h)).mean(),
                     "median_width_factor": np.exp(np.median(h - l))})
    return pd.DataFrame(rows)


def figures(model, te, pred):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred / np.log(10), te.log_fee / np.log(10), s=4, alpha=0.3)
    lim = [4.5, 8.5]
    ax.plot(lim, lim, color="black", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="Predicted fee (log10 EUR)", ylabel="Actual fee (log10 EUR)",
           title="Model A on the 2024-2026 test set")
    fig.savefig(FIG / "calibration_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    import shap
    x = te[FEATS_A].sample(min(2000, len(te)), random_state=0)
    sv = model.predict(x, pred_contrib=True)[:, :-1]
    xn = x.copy()
    for c in CAT:
        xn[c] = xn[c].cat.codes
    shap.summary_plot(sv, xn, max_display=20, show=False)
    plt.title("SHAP values, Model A (effect on log fee / value)")
    plt.savefig(FIG / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def gain_share(model, feats):
    imp = pd.Series(model.feature_importance("gain"), feats).sort_values(ascending=False)
    return (imp / imp.sum()).rename("gain_share")


def position_report(a, te, model_v, preds):
    """Model V by position: error of each variant, and the weight of each parameter for each position."""
    tr, va = a[a.split == "train"], a[a.split == "valid"]
    feats = [f for f in FEATS_V if f != "position"]
    sep = pd.Series(np.nan, index=te.index)             # one separate model per position, for comparison
    for p in POSITIONS:
        rounds = max(fit_lgb(tr[tr.position == p], va[va.position == p], feats, "log_ratio").best_iteration, 30)
        m = fit_lgb(pd.concat([tr, va])[lambda d: d.position == p], None, feats, "log_ratio", rounds)
        sep[te.position == p] = m.predict(te.loc[te.position == p, feats]) + te.log_mv_pre[te.position == p]
    preds = {**preds, "one model per position": sep}
    rows = [{"position": p, "n": int(k.sum()), **{name: np.sqrt(((pr[k] - te.log_fee[k]) ** 2).mean())
                                                   for name, pr in preds.items()}}
            for p, k in [("All", te.position.isin(POSITIONS))] + [(p, te.position == p) for p in POSITIONS]]
    # share of the correction (mean absolute SHAP contribution) carried by each parameter, per position
    sv = np.abs(model_v.predict(te[FEATS_V], pred_contrib=True)[:, :-1])
    share = pd.DataFrame({p: sv[(te.position == p).values].mean(0) for p in POSITIONS}, index=FEATS_V)
    share.insert(0, "All", sv.mean(0))
    return pd.DataFrame(rows), share / share.sum()


def cross_fit(df, feats, target, offset, rounds, k=5):
    """Out-of-fold predictions, folds grouped by player, so every gap is out-of-sample."""
    pred = pd.Series(np.nan, index=df.index)
    for tr, te in GroupKFold(n_splits=k).split(df, groups=df.player_id):
        m = fit_lgb(df.iloc[tr], None, feats, target, rounds)
        pred.iloc[te] = m.predict(df.iloc[te][feats])
    return pred + (df[offset] if offset else 0)


def main():
    OUT.mkdir(exist_ok=True)
    df = prepare(pd.read_parquet("data/processed/transfers_model.parquet"))
    df["log_ratio"] = df.log_fee - df.log_mv_pre
    a = df[df.log_mv_pre.notna()].copy()          # Models A and V need a pre-transfer valuation
    te = a[a.split == "test"]
    print(a.split.value_counts().to_dict(), "| Model B sample:", df.split.value_counts().to_dict())

    res = [metrics(te.log_fee, te.log_mv_pre, "naive: fee = market value")]
    fit = a[a.split != "test"]
    b1, b0 = np.polyfit(fit.log_mv_pre, fit.log_fee, 1)
    res.append(metrics(te.log_fee, b0 + b1 * te.log_mv_pre, "OLS on log market value only"))
    res.append(metrics(te.log_fee, hedonic(a, [f for f in FEATS_A if f != "year"]), "hedonic ridge, all features"))

    model_a, rounds_a, pred_a = lgb_predict(a, FEATS_A, "log_ratio", "log_mv_pre")
    res.append(metrics(te.log_fee, pred_a, "LightGBM A (target: fee / value)"))
    _, _, pred_ad = lgb_predict(a, FEATS_A, "log_fee", None)
    res.append(metrics(te.log_fee, pred_ad, "LightGBM A-direct (target: fee)"))
    model_v, _, pred_v = lgb_predict(a, FEATS_V, "log_ratio", "log_mv_pre")
    res.append(metrics(te.log_fee, pred_v, "LightGBM V (no buyer information)"))
    model_v_old, _, pred_v_old = lgb_predict(a, FEATS_V_OLD, "log_ratio", "log_mv_pre")
    res.append(metrics(te.log_fee, pred_v_old, "LightGBM V without position statistics (43 parameters)"))
    d = (pred_v_old - te.log_fee) ** 2 - (pred_v - te.log_fee) ** 2
    print("position statistics, gain in squared error: t = %.2f" % (d.mean() / (d.std() / np.sqrt(len(d)))))
    _, rounds_b, pred_b = lgb_predict(df, FEATS_B, "log_fee", None)
    teb = df[df.split == "test"]
    res.append(metrics(teb.log_fee, pred_b, "LightGBM B (no Transfermarkt values), all rows"))
    keep = teb.log_mv_pre.notna().values
    res.append(metrics(teb.log_fee[keep], pred_b[keep], "LightGBM B, same rows as A"))

    res = pd.DataFrame(res)
    seg = segments(te, pred_a, te.log_mv_pre)
    itv = intervals(a, rounds_a)
    imp = gain_share(model_a, FEATS_A)
    pos_err, pos_share = position_report(a, te, model_v, {
        "Transfermarkt": te.log_mv_pre, "43 parameters": pred_v_old, "with position statistics": pred_v})
    print(pos_err.round(3).to_string(index=False))
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False), "\nrounds A/B:", rounds_a, rounds_b)
    print(seg.round(3).to_string(index=False))
    print(itv.round(3).to_string(index=False))
    print(imp.head(15).round(3).to_string())
    figures(model_a, te, pred_a)

    # cross-fitted price gaps for every transfer (Model A and Model B)
    a["pred_log_a"] = cross_fit(a, FEATS_A, "log_ratio", "log_mv_pre", rounds_a)
    df["pred_log_b"] = cross_fit(df, FEATS_B, "log_fee", None, rounds_b)
    df["pred_log_a"] = a.pred_log_a
    df["gap_a"], df["gap_b"] = df.log_fee - df.pred_log_a, df.log_fee - df.pred_log_b
    print("cross-fitted RMSE  A: %.3f  B: %.3f" % (np.sqrt((df.gap_a ** 2).mean()), np.sqrt((df.gap_b ** 2).mean())))

    res.to_csv(OUT / "metrics_test.csv", index=False)
    seg.to_csv(OUT / "segment_errors_test.csv", index=False)
    itv.to_csv(OUT / "intervals_test.csv", index=False)
    imp.to_csv(OUT / "feature_importance_a.csv")
    gain_share(model_v, FEATS_V).to_csv(OUT / "feature_importance_v.csv")
    gain_share(model_v_old, FEATS_V_OLD).to_csv(OUT / "feature_importance_v_43.csv")
    pos_err.to_csv(OUT / "position_errors_test.csv", index=False)
    pos_share.to_csv(OUT / "feature_share_by_position.csv")
    model_a.save_model(str(OUT / "model_a.txt"))
    cols = ["tid", "player_id", "player_name", "transfer_date", "from_club_id", "from_club_name", "to_club_id",
            "to_club_name", "transfer_fee", "mv_pre", "split", "pred_log_a", "pred_log_b", "gap_a", "gap_b"]
    df[cols].to_parquet(OUT / "price_gaps.parquet", index=False)


if __name__ == "__main__":
    main()
