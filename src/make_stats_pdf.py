"""Write Statistics.pdf: how far the Transfermarkt value and the model price can be trusted.

Every number in the PDF is recomputed from the data each time the script runs, so the
document is updated by adding a section here and running it again.
Run from the project root (after build_dataset.py):  python src/make_stats_pdf.py
"""
import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import scipy.stats as st
import statsmodels.api as sm

from modeling import FEATS_A, FEATS_V, OUT, conformal_q, fit_lgb, prepare

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = OUT / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
ORANGE, BLUE = "#eb6834", "#2a78d6"       # Transfermarkt, model
EXAMPLE = 10e6                            # the worked example: a player priced at EUR 10m
BANDS, BAND_LABELS = [0, 1e6, 5e6, 20e6, 1e10], ["under 1m", "1m to 5m", "5m to 20m", "over 20m"]
EXAMPLE_BAND = "5m to 20m"
N_BOOT = 2000
N_PERM = 5000                             # random reshuffles of the prices, the "luck" benchmark
ALPHA = 0.05

# one line per step of the work, newest last
UPDATES = [
    ("4 October 2026", "Step 1. Spread of real fees around the Transfermarkt value and around the model price: "
                       "variance, standard deviation, standard error, ranges, worked example at EUR 10m."),
    ("4 October 2026", "Step 2. Significance tests: skill against luck, bias, calibration, stability by year, "
                       "coverage of the ranges, and a scorecard defining what statistically right means."),
    ("4 October 2026", "Step 3. Model changed: eleven position statistics added (54 parameters instead of 43 for the "
                       "website model). All figures in this document recomputed with the new model."),
]


# ---------------------------------------------------------------- calculations

def test_errors():
    """Log of real fee / stated price on the 2024-2026 test set, for each price source."""
    df = prepare(pd.read_parquet("data/processed/transfers_model.parquet"))
    df["log_ratio"] = df.log_fee - df.log_mv_pre
    a = df[df.log_mv_pre.notna()]
    tr, va, te = a[a.split == "train"], a[a.split == "valid"], a[a.split == "test"]
    price, cover = {"Transfermarkt": te.log_mv_pre}, {}
    for name, feats in [("Website model", FEATS_V), ("Full model", FEATS_A)]:
        rounds = fit_lgb(tr, va, feats, "log_ratio").best_iteration
        price[name] = fit_lgb(pd.concat([tr, va]), None, feats, "log_ratio", rounds).predict(te[feats]) + te.log_mv_pre
        # ranges: model trained before 2023, width set on 2023, checked on the test set
        m = fit_lgb(tr, None, feats, "log_ratio", rounds)
        res_va, res_te = (va.log_ratio - m.predict(va[feats])).abs(), (te.log_ratio - m.predict(te[feats])).abs()
        cover[name] = {c: (int((res_te <= conformal_q(res_va, c)).sum()), len(te)) for c in (0.8, 0.9)}
    price = pd.DataFrame(price)
    return price.rsub(te.log_fee, axis=0), np.exp(price), len(tr) + len(va), te, cover


def boot_ci(x, f, rng):
    b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)]
    return np.percentile(b, [2.5, 97.5])


def stats(e, rng):
    """Spread statistics of e = log(real fee / stated price)."""
    e = np.asarray(e)
    n, mean, sd = len(e), e.mean(), e.std(ddof=1)
    z, r = (e - mean) / sd, np.exp(e)
    q = {p: np.percentile(r, p) for p in (2.5, 10, 16, 25, 50, 75, 84, 90, 97.5)}
    return {"n": n, "mean": mean, "var": sd ** 2, "sd": sd, "se": sd / np.sqrt(n),
            "ci_mean": (mean - 1.96 * sd / np.sqrt(n), mean + 1.96 * sd / np.sqrt(n)),
            "ci_sd": boot_ci(e, lambda v: v.std(ddof=1), rng), "rmse": np.sqrt((e ** 2).mean()),
            "skew": (z ** 3).mean(), "kurt": (z ** 4).mean() - 3, "q": q,
            "r_mean": r.mean(), "r_var": r.var(ddof=1), "r_sd": r.std(ddof=1), "r_se": r.std(ddof=1) / np.sqrt(n),
            "in50": ((r >= 0.5) & (r <= 1.5)).mean(), "in25": ((r >= 0.75) & (r <= 1.25)).mean(),
            "below50": (r < 0.5).mean(), "above50": (r > 1.5).mean(),
            "med_ape": np.median(np.abs(r - 1))}


def paired(e_tm, e_model, rng):
    """Is the model's absolute log error smaller than Transfermarkt's on the same transfers?"""
    d = np.abs(np.asarray(e_tm)) - np.abs(np.asarray(e_model))
    return {"diff": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)), "ci": boot_ci(d, np.mean, rng),
            "share": (d > 0).mean()}


def holm(p):
    """Holm correction for several tests at once."""
    p, order = np.asarray(p, float), np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty(len(p))
    out[order] = adj
    return out


def significance(te, err, price, band, rng):
    """Tests of skill against luck, bias, calibration and stability for each price source."""
    y, year = te.log_fee.values, te.transfer_date.dt.year.values
    sst, log_tm = ((y - y.mean()) ** 2).sum(), np.log(price["Transfermarkt"].values)
    out = {}
    for k in price.columns:
        p, e = np.log(price[k].values), err[k].values
        r2 = 1 - (e ** 2).sum() / sst
        perm = np.array([1 - ((y - rng.permutation(p)) ** 2).sum() / sst for _ in range(N_PERM)])
        mz = sm.OLS(y, sm.add_constant(p)).fit(cov_type="HC3")
        bands = [e[(band[k] == b).values] for b in BAND_LABELS]
        s = {"r2": r2, "perm_mean": perm.mean(), "perm_max": perm.max(), "z": (r2 - perm.mean()) / perm.std(),
             "p_perm": (1 + (perm >= r2).sum()) / (N_PERM + 1), "rmse": np.sqrt((e ** 2).mean()),
             "rmse_luck": np.sqrt(((y - rng.permutation(p)) ** 2).mean()), "rmse_flat": y.std(),
             "pearson": st.pearsonr(p, y)[0], "spearman": st.spearmanr(p, y)[0],
             "t_bias": st.ttest_1samp(e, 0).statistic, "p_bias": st.ttest_1samp(e, 0).pvalue,
             "above": (e > 0).sum() / (e != 0).sum(),
             "p_sign": st.binomtest(int((e > 0).sum()), int((e != 0).sum()), 0.5).pvalue,
             "mz_a": mz.params[0], "mz_b": mz.params[1], "mz_b_se": mz.bse[1],
             "p_mz": float(mz.wald_test((np.eye(2), [0, 1]), scalar=True).pvalue),
             "band_mean": [b.mean() for b in bands],
             "band_p": holm([st.ttest_1samp(b, 0).pvalue for b in bands]),
             "year": {yr: {"n": int((year == yr).sum()), "rmse": np.sqrt((e[year == yr] ** 2).mean()),
                           "r": st.pearsonr(p[year == yr], y[year == yr])[0]} for yr in np.unique(year)}}
        if k != "Transfermarkt":
            # is the correction applied to the Transfermarkt value informative, or noise?
            c_pred, c_act, e_tm = p - log_tm, y - log_tm, err["Transfermarkt"].values
            reg = sm.OLS(c_act, sm.add_constant(c_pred)).fit(cov_type="HC3")
            ok = (c_pred != 0) & (c_act != 0)
            hits = int((np.sign(c_pred[ok]) == np.sign(c_act[ok])).sum())
            d = e_tm ** 2 - e ** 2
            s.update({"c_b": reg.params[1], "c_se": reg.bse[1], "c_t": reg.tvalues[1], "c_p": reg.pvalues[1],
                      "c_r": st.pearsonr(c_pred, c_act)[0], "dir": hits / ok.sum(), "dir_n": int(ok.sum()),
                      "dir_p": st.binomtest(hits, int(ok.sum()), 0.5).pvalue,
                      "dm_t": d.mean() / (d.std(ddof=1) / np.sqrt(len(d))),
                      "dm_p": st.ttest_1samp(d, 0).pvalue})
            for yr in np.unique(year):
                dy = np.abs(e_tm[year == yr]) - np.abs(e[year == yr])
                s["year"][yr].update({"t": dy.mean() / (dy.std(ddof=1) / np.sqrt(len(dy))),
                                      "p": st.ttest_1samp(dy, 0).pvalue})
        out[k] = s
    return out


# ---------------------------------------------------------------- figures

def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def fig_ranges(rows, path):
    """Where the real fee falls for a player priced at EUR 10m: 50%, 80% and 95% ranges."""
    fig, ax = plt.subplots(figsize=(7.2, 0.75 * len(rows) + 0.9))
    for i, (label, s, col) in enumerate(rows):
        y, q = len(rows) - 1 - i, s["q"]
        for (lo, hi), h, alpha in [((2.5, 97.5), 0.10, 0.35), ((10, 90), 0.26, 0.6), ((25, 75), 0.42, 1)]:
            ax.barh(y, (q[hi] - q[lo]) * 10, left=q[lo] * 10, height=h, color=col, alpha=alpha, lw=0)
        ax.plot([q[50] * 10] * 2, [y - 0.21, y + 0.21], color="white", lw=2)
        for p in (10, 90):
            ax.text(q[p] * 10, y + 0.27, "%.1fm" % (q[p] * 10), ha="center", va="bottom", fontsize=8, color=INK,
                    bbox=dict(fc="white", ec="none", pad=1))
    for x, ls in [(5, ":"), (10, "-"), (15, ":")]:
        ax.axvline(x, color=INK2, lw=0.8, ls=ls, zorder=0)
    ax.set_xscale("log")
    ticks = [1, 2, 5, 10, 15, 20, 40, 80]
    ax.set_xticks(ticks, ["%dm" % t for t in ticks])
    ax.set_xticks([], minor=True)
    ax.set_xlim(1, 80)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=9, color=INK)
    ax.set_xlabel("Real fee when the stated price is EUR 10m (thick: half of transfers, medium: 80%, thin: 95%; "
                  "labels: 80% range)", fontsize=8, color=INK2)
    style(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_hist(err, cols, path):
    """Distribution of real fee / stated price, one panel per price source, same axis."""
    fig, axes = plt.subplots(len(cols), 1, figsize=(7.2, 1.9 * len(cols)), sharex=True, sharey=True)
    edges = np.linspace(-4, 4, 65)
    for ax, (name, col) in zip(axes, cols):
        e = np.log2(np.exp(err[name])).clip(-4, 4)
        ax.hist(e, bins=edges, color=col, lw=0.4, ec="white", weights=np.full(len(e), 100 / len(e)))
        lo, hi = np.percentile(e, [10, 90])
        ax.axvspan(lo, hi, color=col, alpha=0.10, lw=0)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title("%s: 80%% of real fees between x%.2f and x%.2f of the price" % (name, 2 ** lo, 2 ** hi),
                     fontsize=9, color=INK, loc="left")
        ax.set_ylabel("% of transfers", fontsize=8, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        style(ax)
    axes[-1].set_xticks(range(-4, 5), ["x1/16", "x1/8", "x1/4", "x1/2", "x1", "x2", "x4", "x8", "x16"])
    axes[-1].set_xlabel("Real fee divided by the stated price (log scale)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- PDF

W = A4[0] - 4 * cm
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=10, leading=14.5,
                      textColor=colors.HexColor(INK), spaceAfter=7)
small = ParagraphStyle("small", parent=body, fontSize=9, leading=12.5, spaceAfter=0)
smallb = ParagraphStyle("smallb", parent=small, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=4)
sub = ParagraphStyle("sub", parent=body, textColor=colors.HexColor(INK2), spaceAfter=14)
h2 = ParagraphStyle("h2", parent=body, keepWithNext=1, fontName="Helvetica-Bold", fontSize=13, leading=17,
                    spaceBefore=12, spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=0, spaceAfter=4)


def P(text, style=body):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def table(rows, widths):
    data = [[P(str(c), smallb if i == 0 else small) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(GRID)),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK2)), ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    return t


def image(path, width=W):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def eur(x):
    return "%.1fm" % (x / 1e6)


def rng_eur(s, lo, hi):
    return "%s to %s" % (eur(EXAMPLE * s["q"][lo]), eur(EXAMPLE * s["q"][hi]))


def pct(x):
    return "%.0f%%" % (100 * x)


def pv(p):
    return "p < 0.001" if p < 0.001 else "p = %.3f" % p


def verdict(ok, detail):
    return "<b>%s</b><br/>%s" % ("Pass" if ok else "Fail", detail)


def significance_sections(names, S, G, cover, n):
    tm, mv, models = G["Transfermarkt"], G["Website model"], names[1:]
    years = sorted(tm["year"])

    def col(f, ks=names):
        return [f(G[k]) for k in ks]

    def worst_band(g):
        i = int(np.argmin(g["band_p"]))
        return g["band_p"][i] > ALPHA, "%s: fees %+.0f%%, %s" % (BAND_LABELS[i], 100 * (np.exp(g["band_mean"][i]) - 1),
                                                                 pv(g["band_p"][i]))

    def cov_cell(k, c):
        hit, m = cover[k][c]
        p = st.binomtest(hit, m, c).pvalue
        return verdict(p > ALPHA, "holds %.1f%%, %s" % (100 * hit / m, pv(p)))

    every_year = {k: all(G[k]["year"][y]["t"] > 0 and G[k]["year"][y]["p"] < ALPHA for y in years) for k in models}
    card = [
        ["Criterion", "Test", *names],
        ["1. Skill, not luck", "Prices explain fees better than the same prices shuffled",
         *col(lambda g: verdict(g["p_perm"] < ALPHA, pv(g["p_perm"])))],
        ["2. No overall bias", "Mean log error is 0 (t-test)",
         *col(lambda g: verdict(g["p_bias"] > ALPHA, "fees %+.0f%%, %s" % (100 * (np.exp(g["t_bias"] * 0 + g["mean"]) - 1), pv(g["p_bias"]))))],
        ["3. Right scale", "Fee rises one for one with price (slope 1, intercept 0)",
         *col(lambda g: verdict(g["p_mz"] > ALPHA, "slope %.2f, %s" % (g["mz_b"], pv(g["p_mz"]))))],
        ["4. No bias at any price level", "Mean log error is 0 in each of the four price bands (worst band shown)",
         *col(lambda g: verdict(*worst_band(g)))],
        ["5. Beats Transfermarkt", "Squared error lower on the same transfers", "reference",
         *col(lambda g: verdict(g["dm_p"] < ALPHA and g["dm_t"] > 0, "t = %.1f, %s" % (g["dm_t"], pv(g["dm_p"]))), models)],
        ["6. Beats it every year", "Same test in %s" % ", ".join(map(str, years)), "reference",
         *[verdict(every_year[k], "weakest year: t = %.1f" % min(G[k]["year"][y]["t"] for y in years)) for k in models]],
        ["7. 80% range is honest", "Range set on 2023 holds 80% of 2024-2026 fees", "no range published",
         *[cov_cell(k, 0.8) for k in models]],
        ["8. 90% range is honest", "Same at 90%", "no range published", *[cov_cell(k, 0.9) for k in models]],
    ]
    n_pass = {k: sum(str(r[2 + i]).startswith("<b>Pass") for r in card[1:]) for i, k in enumerate(names)}
    n_test = {k: sum(str(r[2 + i]).startswith("<b>") for r in card[1:]) for i, k in enumerate(names)}

    return [
        P("7. Skill or luck?", h2),
        P("The test of luck: keep the same %s prices and the same %s fees, but hand the prices to the players at "
          "random, %s times. If the real prices do no better than the shuffled ones, they are luck."
          % (format(n, ","), format(n, ","), format(N_PERM, ","))),
        table([["Against luck", *names],
               ["Share of fee differences explained (R2)", *col(lambda g: pct(g["r2"]))],
               ["Same, prices shuffled: average", *col(lambda g: pct(g["perm_mean"]))],
               ["Same, prices shuffled: best of %s" % format(N_PERM, ","), *col(lambda g: pct(g["perm_max"]))],
               ["Distance from luck, in standard deviations", *col(lambda g: "%.0f" % g["z"])],
               ["Probability of doing this well by luck", *col(lambda g: pv(g["p_perm"]))],
               ["Typical error (RMSE, log scale)", *col(lambda g: "%.2f" % g["rmse"])],
               ["Typical error, prices shuffled", *col(lambda g: "%.2f" % g["rmse_luck"])],
               ["Typical error, same price for every player", *col(lambda g: "%.2f" % g["rmse_flat"])],
               ["Correlation of price and fee (logs)", *col(lambda g: "%.2f" % g["pearson"])],
               ["Rank correlation", *col(lambda g: "%.2f" % g["spearman"])]],
              [0.40, 0.20, 0.20, 0.20]),
        Spacer(1, 6),
        P("Neither source is luck. In %s reshuffles, none came close to the real prices: a shuffled price explains "
          "less than nothing (it is worse than giving every player the same price), Transfermarkt explains %s of "
          "the differences in fees and the website model %s. What is left unexplained, %s and %s, is the part "
          "that is not yet under control." % (format(N_PERM, ","), pct(tm["r2"]), pct(mv["r2"]), pct(1 - tm["r2"]),
                                              pct(1 - mv["r2"]))),

        P("8. Is the model's correction skill or luck?", h2),
        P("Beating luck is easy for the model, because it starts from the Transfermarkt value. The harder question "
          "is whether the correction it applies to that value is information or noise. Three tests on the same "
          "%s transfers." % format(n, ",")),
        table([["The correction to the Transfermarkt value", *models],
               ["Direction right (fee on the side of Transfermarkt the model said)", *col(lambda g: pct(g["dir"]), models)],
               ["Probability of that by coin flip", *col(lambda g: pv(g["dir_p"]), models)],
               ["Real correction per unit of predicted correction (1 = right size)",
                *col(lambda g: "%.2f (std error %.2f)" % (g["c_b"], g["c_se"]), models)],
               ["t-statistic against no information (0)", *col(lambda g: "%.1f, %s" % (g["c_t"], pv(g["c_p"])), models)],
               ["Correlation of predicted and real correction", *col(lambda g: "%.2f" % g["c_r"], models)],
               ["Squared error lower than Transfermarkt: t-statistic", *col(lambda g: "%.1f, %s" % (g["dm_t"], pv(g["dm_p"])), models)]],
              [0.46, 0.27, 0.27]),
        Spacer(1, 10),
        P("Luck does not repeat. The same comparison year by year, website model against Transfermarkt:"),
        table([["Year", "Transfers", "Error, Transfermarkt", "Error, website model", "t-statistic", "Probability by luck"],
               *[[y, format(tm["year"][y]["n"], ","), "%.2f" % tm["year"][y]["rmse"], "%.2f" % mv["year"][y]["rmse"],
                  "%.1f" % mv["year"][y]["t"], pv(mv["year"][y]["p"])] for y in years]],
              [0.10, 0.14, 0.20, 0.20, 0.14, 0.22]),
        Spacer(1, 6),
        P("The correction points the right way on %s of transfers against 50%% for a coin, and the model beats "
          "Transfermarkt in %s. It is skill. It is also far from complete: the correlation between predicted "
          "and real correction is %.2f, so most of the gap between fee and Transfermarkt value is still "
          "unexplained." % (pct(mv["dir"]), "each of the %d years" % len(years) if every_year["Website model"]
                            else "some years only", mv["c_r"])),

        P("9. What statistically right means, and where each source stands", h2),
        P("Significant is not the same as right. A price source is statistically right when it passes all of the "
          "tests below at the 5% level. Tests 2, 3, 4, 7 and 8 are passed when no flaw can be detected; tests 1, "
          "5 and 6 when the advantage cannot be luck. This scorecard is the target for the next steps."),
        table(card, [0.17, 0.26, 0.19, 0.19, 0.19]),
        Spacer(1, 6),
        P("Score: Transfermarkt %d of %d, website model %d of %d, full model %d of %d."
          % tuple(x for k in names for x in (n_pass[k], n_test[k]))),
        P("Bias by price band (mean error as fees above or below the price; p-values corrected for testing four "
          "bands at once):"),
        table([["Stated price (EUR)", *names],
               *[[b, *col(lambda g, i=i: "%+.0f%%, %s" % (100 * (np.exp(g["band_mean"][i]) - 1), pv(g["band_p"][i])))]
                 for i, b in enumerate(BAND_LABELS)]],
              [0.25, 0.25, 0.25, 0.25]),
        Spacer(1, 6),
        *bullets([
            "<b>Transfermarkt</b> has real skill but is not statistically right: its slope is %.2f instead of 1 and "
            "fees differ from its values by %+.0f%% on average." % (tm["mz_b"], 100 * (np.exp(tm["mean"]) - 1)),
            "<b>The website model</b> has skill beyond Transfermarkt, %s overall bias and a slope of %.2f (%s). "
            "Its 80%% range holds %.1f%% of fees and its 90%% range %.1f%%."
            % ("no detectable" if mv["p_bias"] > ALPHA else "a detectable", mv["mz_b"],
               "not distinguishable from 1" if mv["p_mz"] > ALPHA else "significantly different from 1",
               100 * cover["Website model"][0.8][0] / n, 100 * cover["Website model"][0.9][0] / n),
            "<b>Mean and median disagree for the model.</b> The tests above are on the mean of the log error. "
            "The median fee is above the model price (%.0f%% of fees are above it, sign test %s): the model is "
            "right on average because a minority of large overpricings offsets a majority of small underpricings."
            % (100 * mv["above"], pv(mv["p_sign"])),
            "<b>Being right on average is not being precise.</b> A model can pass every test here and still have "
            "wide ranges. The tests say whether the prices and ranges can be believed; the standard deviation of "
            "section 3 says how useful they are. Both must improve.",
        ]),
    ]


def build(err, price, n_fit, te, cover):
    rng = np.random.default_rng(0)
    names = ["Transfermarkt", "Website model", "Full model"]
    S = {k: stats(err[k], rng) for k in names}
    band = {k: pd.cut(price[k], BANDS, labels=BAND_LABELS) for k in names}
    G = significance(te, err, price, band, rng)
    for k in names:
        G[k]["mean"] = S[k]["mean"]
    SB = {k: {b: stats(err[k][band[k] == b], rng) for b in BAND_LABELS} for k in names}
    PT = {k: paired(err["Transfermarkt"], err[k], rng) for k in names[1:]}
    tm, mv, ex = S["Transfermarkt"], S["Website model"], {k: SB[k][EXAMPLE_BAND] for k in names}

    FIG.mkdir(parents=True, exist_ok=True)
    fig_ranges([("Transfermarkt", ex["Transfermarkt"], ORANGE), ("Website model", ex["Website model"], BLUE)],
               FIG / "stats_ranges_10m.png")
    fig_hist(err, [("Transfermarkt", ORANGE), ("Website model", BLUE)], FIG / "stats_error_distribution.png")

    flat = [{"source": k, "price_band": b, **{m: v for m, v in s.items() if np.isscalar(v)},
             **{"ratio_p%s" % p: v for p, v in s["q"].items()}}
            for k in names for b, s in [("all", S[k])] + list(SB[k].items())]
    pd.DataFrame(flat).to_csv(OUT / "statistics_summary.csv", index=False)

    def col(f):
        return [f(S[k]) for k in names]

    story = [
        P("Statistics: how far the prices can be trusted", h1),
        P("Working document, updated at each step. Last update: %s." % UPDATES[-1][0], sub),

        P("1. The question and the short answer", h2),
        P("A price source says a player is worth EUR 10m. If he is actually sold, how far from 10m can the fee "
          "be? The answer below is measured on %s real transfers from 2024 to mid-2026 that the model never saw "
          "in training. The rows use the %s transfers that Transfermarkt priced between EUR 5m and 20m and the %s "
          "that the website model priced in that band, because the spread depends on the price level (section 6)."
          % (format(tm["n"], ","), format(ex["Transfermarkt"]["n"], ","), format(ex["Website model"]["n"], ","))),
        table([["Stated price: EUR 10m", "Transfermarkt", "Website model"],
               ["Half of real fees fall between", rng_eur(ex["Transfermarkt"], 25, 75), rng_eur(ex["Website model"], 25, 75)],
               ["80% of real fees fall between", rng_eur(ex["Transfermarkt"], 10, 90), rng_eur(ex["Website model"], 10, 90)],
               ["95% of real fees fall between", rng_eur(ex["Transfermarkt"], 2.5, 97.5), rng_eur(ex["Website model"], 2.5, 97.5)],
               ["Real fee inside 5m to 15m (within 50%)", pct(ex["Transfermarkt"]["in50"]), pct(ex["Website model"]["in50"])],
               ["Real fee below 5m", pct(ex["Transfermarkt"]["below50"]), pct(ex["Website model"]["below50"])],
               ["Real fee above 15m", pct(ex["Transfermarkt"]["above50"]), pct(ex["Website model"]["above50"])],
               ["Typical real fee (median)", eur(EXAMPLE * ex["Transfermarkt"]["q"][50]), eur(EXAMPLE * ex["Website model"]["q"][50])],
               ["Top of the 80% range divided by its bottom", *["x%.1f" % (ex[k]["q"][90] / ex[k]["q"][10]) for k in names[:2]]],
               ["Standard deviation (log scale)", *["%.2f" % ex[k]["sd"] for k in names[:2]]]],
              [0.46, 0.27, 0.27]),
        Spacer(1, 8),
        image(FIG / "stats_ranges_10m.png"),
        P("So a range of 5m to 15m around a 10m price holds for %s of players with Transfermarkt and %s with the "
          "website model. Across all price levels the figures are %s and %s: the spread is wider for cheap players."
          % (pct(ex["Transfermarkt"]["in50"]), pct(ex["Website model"]["in50"]), pct(tm["in50"]), pct(mv["in50"]))),
        P("In this band the model's range is narrower in proportion (standard deviation %.2f against %.2f) but it "
          "sits higher: the typical fee is %s for a 10m model price, so the model misses mostly on the high side "
          "(%s of fees above 15m, %s below 5m), where Transfermarkt misses on both sides."
          % (ex["Website model"]["sd"], ex["Transfermarkt"]["sd"], eur(EXAMPLE * ex["Website model"]["q"][50]),
             pct(ex["Website model"]["above50"]), pct(ex["Website model"]["below50"]))),

        P("2. How the error is measured", h2),
        *bullets([
            "<b>Error.</b> For each transfer, e = ln(real fee / stated price). e = 0 means the price was exact, "
            "e = 0.69 means the fee was twice the price, e = -0.69 means half. Logs are used because price errors "
            "are proportional: being 5m off matters at 10m and not at 100m.",
            "<b>Mean.</b> The average of e. It is the systematic bias: above 0, the source prices too low.",
            "<b>Variance and standard deviation.</b> s2 = sum of (e - mean)2 / (n - 1), and s is its square root. "
            "s is the typical distance between a fee and the price. exp(s) turns it into a factor: about two "
            "thirds of fees lie between price / exp(s) and price x exp(s) if errors are bell-shaped.",
            "<b>Standard error.</b> s / square root of n. It is the uncertainty on the mean, not on one player. "
            "With %s transfers it is small, which is why averages are reliable and single prices are not." % format(tm["n"], ","),
            "<b>Ranges.</b> The 50%, 80% and 95% ranges are read directly from the data (percentiles of fee / "
            "price), with no assumption on the shape of the errors.",
            "<b>Transfermarkt price.</b> The last value published at least 30 days before the transfer, so that "
            "it cannot already contain the fee.",
            "<b>Model price.</b> The website model (no buyer information), trained on the %s transfers before "
            "2024 and applied to 2024-2026. The full model, which knows the buyer, is shown for comparison."
            % format(n_fit, ","),
        ]),

        P("3. Transfermarkt and the model: full statistics", h2),
        P("All %s test transfers. Log scale: statistics of e = ln(fee / price)." % format(tm["n"], ",")),
        table([["Log scale", *names],
               ["Mean (bias)", *col(lambda s: "%+.3f" % s["mean"])],
               ["Standard error of the mean", *col(lambda s: "%.3f" % s["se"])],
               ["95% confidence interval of the mean", *col(lambda s: "%+.3f to %+.3f" % s["ci_mean"])],
               ["Variance", *col(lambda s: "%.3f" % s["var"])],
               ["Standard deviation", *col(lambda s: "%.3f" % s["sd"])],
               ["95% confidence interval of the standard deviation", *col(lambda s: "%.3f to %.3f" % tuple(s["ci_sd"]))],
               ["Standard deviation as a factor, exp(s)", *col(lambda s: "x%.2f" % np.exp(s["sd"]))],
               ["Root mean squared error", *col(lambda s: "%.3f" % s["rmse"])],
               ["Skewness", *col(lambda s: "%+.2f" % s["skew"])],
               ["Excess kurtosis", *col(lambda s: "%.2f" % s["kurt"])]],
              [0.40, 0.20, 0.20, 0.20]),
        Spacer(1, 10),
        P("Same transfers, in plain ratios: real fee divided by stated price."),
        table([["Ratio fee / price", *names],
               ["Median", *col(lambda s: "%.2f" % s["q"][50])],
               ["Mean", *col(lambda s: "%.2f" % s["r_mean"])],
               ["Variance", *col(lambda s: "%.2f" % s["r_var"])],
               ["Standard deviation", *col(lambda s: "%.2f" % s["r_sd"])],
               ["Standard error of the mean", *col(lambda s: "%.3f" % s["r_se"])],
               ["Middle 50% of transfers", *col(lambda s: "%.2f to %.2f" % (s["q"][25], s["q"][75]))],
               ["Middle 68% (one standard deviation)", *col(lambda s: "%.2f to %.2f" % (s["q"][16], s["q"][84]))],
               ["Middle 80%", *col(lambda s: "%.2f to %.2f" % (s["q"][10], s["q"][90]))],
               ["Middle 95%", *col(lambda s: "%.2f to %.2f" % (s["q"][2.5], s["q"][97.5]))],
               ["Fee within 25% of the price", *col(lambda s: pct(s["in25"]))],
               ["Fee within 50% of the price", *col(lambda s: pct(s["in50"]))],
               ["Median absolute error, in % of the price", *col(lambda s: pct(s["med_ape"]))]],
              [0.40, 0.20, 0.20, 0.20]),
        Spacer(1, 4),
        P("The method note reports median errors of 50% and 42%. Those are measured in % of the fee; here the "
          "error is in % of the stated price, which is what the 10m example needs.", small),

        P("4. Reading the numbers", h2),
        *bullets([
            "<b>Transfermarkt.</b> Standard deviation %.2f in logs, a factor of %.2f. The mean is %+.3f with a "
            "standard error of %.3f: on average fees are %s Transfermarkt values (%+.0f%% at the median), and "
            "this bias is %s." % (tm["sd"], np.exp(tm["sd"]), tm["mean"], tm["se"],
                                  "above" if tm["mean"] > 0 else "below", 100 * (tm["q"][50] - 1),
                                  "statistically clear" if abs(tm["mean"]) > 1.96 * tm["se"] else "not distinguishable from zero"),
            "<b>Website model.</b> Standard deviation %.2f, a factor of %.2f. The variance falls from %.2f to "
            "%.2f, so the model removes %s of the variance of Transfermarkt's error. Its mean is %+.3f with a "
            "standard error of %.3f: %s." % (mv["sd"], np.exp(mv["sd"]), tm["var"], mv["var"],
                                              pct(1 - mv["var"] / tm["var"]), mv["mean"], mv["se"],
                                              "a bias remains" if abs(mv["mean"]) > 1.96 * mv["se"] else "no measurable bias"),
            "<b>The plain standard deviation misleads.</b> In ratios the standard deviation is %.2f for "
            "Transfermarkt and %.2f for the model, which would put a 10m player at 10m plus or minus %s and %s. "
            "A few transfers at several times the price inflate these figures; the percentile ranges are the "
            "ones to use." % (tm["r_sd"], mv["r_sd"], eur(EXAMPLE * tm["r_sd"]), eur(EXAMPLE * mv["r_sd"])),
            "<b>Errors are not bell-shaped.</b> Excess kurtosis is %.1f for Transfermarkt and %.1f for the model "
            "(0 for a normal curve): extreme misses are more frequent than a normal curve predicts. This is why "
            "the ranges are taken from percentiles." % (tm["kurt"], mv["kurt"]),
        ]),
        image(FIG / "stats_error_distribution.png"),

        P("5. Is the model really better than Transfermarkt?", h2),
        P("Both are compared on the same %s transfers, so the test is paired: for each transfer, the absolute "
          "log error of Transfermarkt minus that of the model." % format(tm["n"], ",")),
        table([["Paired comparison with Transfermarkt", "Website model", "Full model"],
               ["Mean reduction in absolute log error", *["%.3f" % PT[k]["diff"] for k in names[1:]]],
               ["Standard error", *["%.3f" % PT[k]["se"] for k in names[1:]]],
               ["95% confidence interval", *["%.3f to %.3f" % tuple(PT[k]["ci"]) for k in names[1:]]],
               ["Ratio to standard error (t)", *["%.1f" % (PT[k]["diff"] / PT[k]["se"]) for k in names[1:]]],
               ["Transfers where the model is closer", *[pct(PT[k]["share"]) for k in names[1:]]]],
              [0.46, 0.27, 0.27]),
        Spacer(1, 6),
        P("The website model is closer to the real fee than Transfermarkt on %s of transfers. The gain is "
          "%s, but on a single player it is not guaranteed: Transfermarkt is closer on the other %s."
          % (pct(PT["Website model"]["share"]),
             "far beyond chance" if PT["Website model"]["ci"][0] > 0 else "not distinguishable from chance",
             pct(1 - PT["Website model"]["share"]))),

        P("6. The spread depends on the price level", h2),
        P("Transfers are grouped by the price each source gave. Standard deviation in logs, then the 80% range "
          "of fee / price."),
        table([["Stated price (EUR)", "n", "Std dev", "80% range", "Within 50%", "n", "Std dev", "80% range", "Within 50%"],
               *[[b, *[x for k in names[:2] for x in (SB[k][b]["n"], "%.2f" % SB[k][b]["sd"],
                                                       "%.2f to %.2f" % (SB[k][b]["q"][10], SB[k][b]["q"][90]),
                                                       pct(SB[k][b]["in50"]))]] for b in BAND_LABELS]],
              [0.16, 0.08, 0.10, 0.17, 0.11, 0.08, 0.10, 0.17, 0.11]),
        Spacer(1, 4),
        P("Columns 2 to 5: Transfermarkt. Columns 6 to 9: website model.", small),

        *significance_sections(names, S, G, cover, tm["n"]),

        P("10. Limits of these statistics", h2),
        *bullets([
            "<b>A fee is not the true value.</b> Two clubs can agree on different fees for the same player "
            "depending on urgency, contract length or the buyer. Part of the spread is this deal noise, which "
            "no price can remove. The standard deviations above are an upper bound on the pricing error itself.",
            "<b>Only sold players are measured.</b> The statistics describe players who were transferred for a "
            "disclosed fee of at least EUR 50,000. Players who are not for sale may behave differently.",
            "<b>The floor at EUR 50,000</b> cuts the lowest fees, which narrows the measured spread for players "
            "priced under 1m.",
            "<b>One test period.</b> 2024 to mid-2026. The spread may differ in other market conditions.",
            "<b>Transfers are treated as independent.</b> Fees in the same window move together, so the true "
            "p-values are somewhat larger than shown. This cannot change results with t above 10; it matters for "
            "the borderline ones.",
            "<b>With %s transfers, small flaws become significant.</b> A failed test says a flaw exists, not that "
            "it is large. The size is given next to each test." % format(tm["n"], ","),
        ]),

        P("Update log", h2),
        *bullets(["<b>%s.</b> %s" % u for u in UPDATES]),
    ]
    return story


def footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawString(2 * cm, 1.2 * cm, "Football player fee model - statistics")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, "Page %d" % doc.page)


if __name__ == "__main__":
    story = build(*test_errors())
    SimpleDocTemplate("Statistics.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                      bottomMargin=2 * cm, title="Statistics: how far the prices can be trusted",
                      author="Football player fee model").build(story, onFirstPage=footer, onLaterPages=footer)
    print("written")
