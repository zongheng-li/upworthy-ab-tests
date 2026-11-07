"""Helpers shared by the analysis notebooks.

Everything statistical lives here so the notebooks stay short and readable.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "upworthy-archive-exploratory-packages-03.12.2020.csv"
SQL_PATH = ROOT / "sql" / "01_build_tables.sql"
FIG_DIR = ROOT / "report" / "figures"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def connect():
    """Open an in-memory DuckDB database and build the packages / tests / arms tables."""
    con = duckdb.connect()
    sql = SQL_PATH.read_text().replace("{data_path}", DATA_PATH.as_posix())
    for statement in sql.split(";"):
        if statement.strip():
            con.execute(statement)
    return con


def load_arms(con, include_problem_period=False):
    """Arms table as a DataFrame, dropping tests from the problem period and tests with no clicks."""
    where = "" if include_problem_period else "WHERE NOT in_problem_period"
    arms = con.execute(f"SELECT * FROM arms {where}").df()
    arms["is_aa"] = arms["is_aa"].astype(bool)
    arms["in_problem_period"] = arms["in_problem_period"].astype(bool)
    has_clicks = arms.groupby("test_id")["clicks"].transform("sum") > 0
    return arms[has_clicks].reset_index(drop=True)


# --------------------------------------------------------------------------
# Basic inference
# --------------------------------------------------------------------------
def wilson_ci(clicks, impressions, level=0.95):
    """Wilson confidence interval for a binomial proportion."""
    z = stats.norm.ppf(1 - (1 - level) / 2)
    n = np.asarray(impressions, dtype=float)
    p = np.asarray(clicks, dtype=float) / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half, center + half


def srm_pvalue(impressions):
    """Sample-ratio-mismatch test: were impressions split evenly across arms?"""
    return stats.chisquare(np.asarray(impressions, dtype=float)).pvalue


def per_test_chi2(arms):
    """Pearson chi-square test of 'all arms have the same CTR', one row per test.

    Returns test_id, chi2, df (= n_arms - 1), is_aa.
    """
    a = arms.copy()
    expected = a["impressions"] * a["ctr_pooled"]
    a["chi2_part"] = (a["clicks"] - expected) ** 2 / (
        a["impressions"] * a["ctr_pooled"] * (1 - a["ctr_pooled"])
    )
    out = a.groupby("test_id").agg(
        chi2=("chi2_part", "sum"),
        n_arms=("n_arms", "first"),
        is_aa=("is_aa", "first"),
    )
    out["df"] = out["n_arms"] - 1
    return out.reset_index()


def estimate_dispersion(chi2_table):
    """Overdispersion factor phi, estimated from A/A tests.

    If clicks were truly binomial, chi2 / df would average 1 in A/A tests.
    phi > 1 means CTRs vary more than the binomial model allows.
    """
    aa = chi2_table[chi2_table["is_aa"]]
    return aa["chi2"].sum() / aa["df"].sum()


def benjamini_hochberg(pvalues):
    """BH-adjusted p-values (q-values)."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1)
    return q


def storey_pi0(pvalues, lam=0.5):
    """Storey's estimate of the share of true nulls."""
    p = np.asarray(pvalues, dtype=float)
    return min(1.0, np.mean(p > lam) / (1 - lam))


def mde_two_arm(ctr, n_per_arm, phi=1.0, alpha=0.05, power=0.8):
    """Minimum detectable absolute difference between two arms."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * np.sqrt(2 * phi * ctr * (1 - ctr) / n_per_arm)


def n_per_arm_needed(ctr, rel_lift, phi=1.0, alpha=0.05, power=0.8):
    """Impressions per arm needed to detect a relative lift between two arms."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    delta = ctr * rel_lift
    return np.ceil(2 * phi * ctr * (1 - ctr) * (z / delta) ** 2)


# --------------------------------------------------------------------------
# Winner's curse and empirical Bayes
# --------------------------------------------------------------------------
def add_deviations(arms, phi):
    """Each arm's deviation from its test's pooled CTR, on a common scale.

    dev_scaled = (ctr - ctr_pooled) * sqrt(k / (k - 1))
    noise_var  = phi * p(1 - p) / impressions

    The sqrt(k / (k - 1)) factor undoes the shrinkage that comes from
    subtracting a mean that includes the arm itself.
    """
    a = arms.copy()
    k = a["n_arms"]
    a["dev"] = a["ctr"] - a["ctr_pooled"]
    a["dev_scaled"] = a["dev"] * np.sqrt(k / (k - 1))
    a["noise_var"] = phi * a["ctr_pooled"] * (1 - a["ctr_pooled"]) / a["impressions"]
    return a


def fit_spike_slab(dev_scaled, noise_var):
    """Fit the prior  theta ~ pi0 * (point mass at 0) + (1 - pi0) * Normal(0, tau^2)
    by maximum likelihood, where dev_scaled ~ Normal(theta, noise_var).

    Returns (pi0, tau2).
    """
    x = np.asarray(dev_scaled, dtype=float)
    v = np.asarray(noise_var, dtype=float)

    def neg_log_lik(params):
        pi0 = 1 / (1 + np.exp(-params[0]))
        tau2 = np.exp(params[1])
        lik = pi0 * stats.norm.pdf(x, 0, np.sqrt(v)) + (1 - pi0) * stats.norm.pdf(
            x, 0, np.sqrt(v + tau2)
        )
        return -np.log(lik).sum()

    start = [0.0, np.log(np.var(x))]
    res = minimize(neg_log_lik, start, method="Nelder-Mead")
    pi0 = 1 / (1 + np.exp(-res.x[0]))
    tau2 = np.exp(res.x[1])
    return pi0, tau2


def posterior_mean(dev_scaled, noise_var, pi0, tau2):
    """Posterior mean of theta and P(arm has a real effect | data)."""
    x = np.asarray(dev_scaled, dtype=float)
    v = np.asarray(noise_var, dtype=float)
    null_part = pi0 * stats.norm.pdf(x, 0, np.sqrt(v))
    real_part = (1 - pi0) * stats.norm.pdf(x, 0, np.sqrt(v + tau2))
    p_real = real_part / (null_part + real_part)
    shrink = tau2 / (tau2 + v)
    return p_real * shrink * x, p_real


def winner_lifts(arms):
    """For each test, the winning arm's lift over the other arms.

    Lift is relative to the test's pooled CTR:
        naive_lift = (winner CTR - CTR of the other arms) / pooled CTR
        eb_lift    = the same quantity using empirical-Bayes estimates
    Requires columns from add_deviations() plus theta_hat.
    """
    a = arms.copy()
    # EB estimate of (this arm - mean of the other arms) for every arm
    sum_theta = a.groupby("test_id")["theta_hat"].transform("sum")
    k_all = a["n_arms"]
    a["eb_diff"] = a["theta_hat"] - (sum_theta - a["theta_hat"]) / (k_all - 1)
    idx = a.groupby("test_id")["ctr"].idxmax()
    w = a.loc[idx].copy()

    totals = a.groupby("test_id")[["clicks", "impressions"]].sum()
    w = w.join(totals, on="test_id", rsuffix="_test")
    ctr_others = (w["clicks_test"] - w["clicks"]) / (
        w["impressions_test"] - w["impressions"]
    )
    w["naive_lift"] = (w["ctr"] - ctr_others) / w["ctr_pooled"]
    w["eb_lift"] = w["eb_diff"] / w["ctr_pooled"]
    return w.reset_index(drop=True)


def simulate_winners(arms, pi0, tau2, phi, seed=0):
    """Simulate tests that look like the real ones, where the true effects are known.

    Each real test keeps its number of arms, impressions and pooled CTR;
    true arm effects are drawn from the fitted prior, and the extra
    (overdispersion) noise is added before drawing clicks.
    Returns one row per simulated test: true, naive and EB winner lift.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _, t in arms.groupby("test_id"):
        n = t["impressions"].to_numpy()
        k = len(n)
        base = t["ctr_pooled"].iloc[0]

        is_real = rng.random(k) > pi0
        theta = np.where(is_real, rng.normal(0, np.sqrt(tau2), k), 0.0)
        extra_noise = rng.normal(0, np.sqrt((phi - 1) * base * (1 - base) / n))
        p_true = np.clip(base + theta + extra_noise, 1e-6, 0.5)
        clicks = rng.binomial(n, p_true)

        pooled = clicks.sum() / n.sum()
        if pooled == 0:
            continue
        ctr = clicks / n
        dev_scaled = (ctr - pooled) * np.sqrt(k / (k - 1))
        noise_var = phi * pooled * (1 - pooled) / n
        theta_hat, _ = posterior_mean(dev_scaled, noise_var, pi0, tau2)

        i = np.argmax(ctr)
        others = np.delete(np.arange(k), i)
        true_lift = (theta[i] - theta[others].mean()) / base
        naive_lift = (ctr[i] - clicks[others].sum() / n[others].sum()) / pooled
        eb_lift = (theta_hat[i] - theta_hat[others].mean()) / pooled
        rows.append((true_lift, naive_lift, eb_lift))
    return pd.DataFrame(rows, columns=["true_lift", "naive_lift", "eb_lift"])
