"""
nis_core.py — Wiederverwendbare Kernfunktionen fuer die generische (VA-freie) NIS-Analyse.

Dieses Modul enthaelt ausschliesslich Plain-Vanilla-Bausteine (konstantes R, kein
Volumen-Bezug) -- die Volume-Awareness-Erweiterung (VA) ist bewusst NICHT Teil dieses
Moduls, siehe Projektnotiz vom 17.09.2026 (VA wird fuer eine eigene, spaetere
Publikation zurueckgehalten).

Verwendet von:
  - notebooks/76_Generic_CUSUM_Comparison.ipynb
  - notebooks/77_Generic_Causal_False_Alarms.ipynb

Fuer das oeffentliche Repo: dieses Modul ist datenquellen-agnostisch (nimmt einfache
numpy-Arrays entgegen), funktioniert also identisch mit EODHD-Daten (Paper-Zahlen)
und mit den 10-20 Yahoo-Demo-Tickern (Repo-Illustration).
"""
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Ground Truth: Directional Change
# ---------------------------------------------------------------------------

def directional_change(prices, delta):
    """Zakamulin Directional-Change-Ereignisse. delta z.B. 0.10 fuer 10%.
    Rueckgabe: Liste von (confirm_idx, extreme_idx, 'peak'|'trough')."""
    n = len(prices)
    mode = None
    ext_idx = 0
    ext_price = prices[0]
    events = []
    for t in range(1, n):
        p = prices[t]
        if mode is None:
            if p >= ext_price * (1 + delta):
                mode = 'up'; events.append((t, ext_idx, 'trough')); ext_idx, ext_price = t, p
            elif p <= ext_price * (1 - delta):
                mode = 'down'; events.append((t, ext_idx, 'peak')); ext_idx, ext_price = t, p
            elif p > ext_price:
                ext_idx, ext_price = t, p
        elif mode == 'up':
            if p > ext_price:
                ext_idx, ext_price = t, p
            elif p <= ext_price * (1 - delta):
                events.append((t, ext_idx, 'peak')); mode = 'down'; ext_idx, ext_price = t, p
        elif mode == 'down':
            if p < ext_price:
                ext_idx, ext_price = t, p
            elif p >= ext_price * (1 + delta):
                events.append((t, ext_idx, 'trough')); mode = 'up'; ext_idx, ext_price = t, p
    return events


# ---------------------------------------------------------------------------
# 2. Etabliertes Signal: SMA x EMA Crossover
# ---------------------------------------------------------------------------

def sma_ema_crossings(prices, sma_window=200, ema_span=50):
    """SMA200 (langsam) x EMA50 (schnell) Crossover -- etablierter Standard,
    unabhaengig von jeder eigenen Architektur."""
    s = pd.Series(prices)
    sma = s.rolling(sma_window, min_periods=sma_window).mean().values
    ema = s.ewm(span=ema_span, min_periods=ema_span, adjust=False).mean().values
    diff = sma - ema
    sign = np.sign(diff)
    cross = np.zeros(len(sign), dtype=bool)
    valid = ~np.isnan(diff)
    for t in range(1, len(sign)):
        if valid[t] and valid[t-1] and sign[t] != sign[t-1]:
            cross[t] = True
    return np.where(cross)[0]


# ---------------------------------------------------------------------------
# 3. Plain-Vanilla Kalman-Filter: NIS und standardisierte Innovation
# ---------------------------------------------------------------------------

def kalman_plain_vanilla(log_prices, Q=1e-4, R=1e-4):
    """Lehrbuch-Kalman-Filter (Local Level / Random Walk + Noise), konstantes R.
    KEINE Volumen-Anpassung -- bewusst Plain-Vanilla.
    Rueckgabe: (nis, standardized_innovation) -- beide Arrays gleicher Laenge wie log_prices,
    erster Eintrag ist 0/NaN (kein Vorgaenger)."""
    n = len(log_prices)
    x_filt = np.zeros(n)
    P_filt = np.zeros(n)
    nis = np.full(n, np.nan)
    std_innov = np.full(n, np.nan)
    x_filt[0] = log_prices[0]
    P_filt[0] = R
    for t in range(1, n):
        x_pred = x_filt[t-1]
        P_pred = P_filt[t-1] + Q
        S_t = P_pred + R
        innov = log_prices[t] - x_pred
        K = P_pred / S_t
        x_filt[t] = x_pred + K * innov
        P_filt[t] = (1 - K) * P_pred
        nis[t] = (innov ** 2) / S_t if S_t > 0 else np.nan
        std_innov[t] = innov / np.sqrt(S_t) if S_t > 0 else np.nan
    return nis, std_innov


def cusum_from_std_innovations(std_innov, window=20):
    """Rollierendes CUSUM (Brown/Durbin/Evans-Familie) auf standardisierten
    Innovationen -- Summe SIGNIERTER Werte im Fenster, |Summe| als Teststatistik.
    Kausal (nur rueckwaerts), rollierend ueber `window` Tage."""
    n = len(std_innov)
    cusum = np.full(n, np.nan)
    s = pd.Series(std_innov)
    rolling_sum = s.rolling(window, min_periods=window).sum()
    cusum[:] = np.abs(rolling_sum.values)
    return cusum


# ---------------------------------------------------------------------------
# 4. Konkurrenzgroessen (kein KF noetig)
# ---------------------------------------------------------------------------

def naive_heuristic(log_prices, volume, window=20):
    """|Tagesrendite| in Std-Einheiten (rollierend, window Tage) * relatives Volumen."""
    n = len(log_prices)
    returns = np.diff(log_prices, prepend=log_prices[0])
    ret_series = pd.Series(returns)
    roll_std = ret_series.rolling(window, min_periods=window).std().values
    z = np.abs(returns) / np.where(roll_std > 0, roll_std, np.nan)
    median_vol = np.nanmedian(volume)
    rel_vol = volume / median_vol if median_vol > 0 else np.ones(n)
    return z * rel_vol


def amihud_illiquidity(log_prices, volume, dollar_volume=None, window=20):
    """Amihud (2002) Illiquiditaetsmass, rollierender Durchschnitt: |Rendite| / Dollarvolumen."""
    n = len(log_prices)
    returns = np.diff(log_prices, prepend=log_prices[0])
    if dollar_volume is None:
        prices = np.exp(log_prices)
        dollar_volume = prices * volume
    daily = np.abs(returns) / np.where(dollar_volume > 0, dollar_volume, np.nan)
    daily_series = pd.Series(daily)
    return daily_series.rolling(window, min_periods=window).mean().values


# ---------------------------------------------------------------------------
# 4b. Segmentierung gegen Nahtstellen-Artefakte (Luecken durch Bereinigung oder
#     im Rohdatensatz selbst) -- WICHTIG bei jedem bereinigten/gefilterten Panel.
# ---------------------------------------------------------------------------

def split_into_continuous_segments(dates, max_gap_days=10):
    """Zerlegt eine sortierte Datums-Sequenz in maximale zusammenhaengende Bloecke,
    in denen der Kalendertage-Abstand zwischen aufeinanderfolgenden Zeilen nie
    max_gap_days ueberschreitet. Verhindert, dass ein Kalman-Filter (oder CUSUM,
    DC, SMA/EMA) zwei durch eine echte Luecke getrennte Tage als normalen
    Ein-Tages-Schritt behandelt -- genau das Problem, das beim Entfernen von
    Stale-Quote-Streaks an der Nahtstelle entsteht (siehe EQNR.OL-Fund, 18.09.).

    Rueckgabe: Liste von (start_idx, end_idx_exklusiv)-Tupeln, sortiert, luecken-
    los deckend. Jedes Tupel ist ein eigenstaendiger Abschnitt fuer DC/SMA-EMA/
    KF/CUSUM -- niemals ueber eine Segmentgrenze hinweg rechnen.
    """
    n = len(dates)
    if n == 0:
        return []
    dates = pd.to_datetime(pd.Series(dates)).values
    gaps = np.diff(dates).astype('timedelta64[D]').astype(int)
    breaks = np.where(gaps > max_gap_days)[0] + 1  # Index der ersten Zeile NACH der Luecke
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [n]])
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# 5. Kausales Fenster, Matching, Konfusionsmatrix
# ---------------------------------------------------------------------------

def causal_window_max(series, signal_idx, window):
    """Maximalwert von `series` im kausalen Fenster [signal_idx-window+1, signal_idx].
    None, falls Fenster nicht vollstaendig verfuegbar oder NaN enthalten."""
    lo = signal_idx - window + 1
    if lo < 0:
        return None
    w = series[lo:signal_idx + 1]
    if np.any(np.isnan(w)):
        return None
    return np.max(w)


def is_hit(dc_extreme_idx, signal_idx, match_window):
    """True, wenn ein DC-Extremum innerhalb +-match_window Tagen des Signals liegt."""
    if len(dc_extreme_idx) == 0:
        return False
    return bool(np.any(np.abs(dc_extreme_idx - signal_idx) <= match_window))


def logistic_regression_manual(X, y, max_iter=100, tol=1e-8, l2_reg=1.0):
    """Newton-Raphson logistische Regression mit leichter L2-Regularisierung
    (Schutz gegen quasi-perfekte Separation bei kleinen Stichproben)."""
    n, k = X.shape
    beta = np.zeros(k)
    reg_vec = np.full(k, l2_reg)
    reg_vec[0] = 0.0
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-10, None)
        grad = X.T @ (y - p) - reg_vec * beta
        H = -(X * W[:, None]).T @ X - np.diag(reg_vec)
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta_new = beta - delta
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    W = np.clip(p * (1 - p), 1e-10, None)
    H_final = -(X * W[:, None]).T @ X - np.diag(reg_vec)
    cov = np.linalg.inv(-H_final)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return beta, p_vals


def threshold_sweep(df, score_col, hit_col, percentiles=(10, 20, 30, 40, 50, 70, 90, 95)):
    """Fuer jede Schwellen-Perzentile: Fehlalarme erwischt / Treffer verloren /
    Fehlalarmquote danach / Signale behalten. df braucht Spalten score_col, hit_col (bool)."""
    rows = []
    n_fa = int((~df[hit_col]).sum())
    n_hit = int(df[hit_col].sum())
    fa_rate_before = n_fa / len(df) if len(df) > 0 else np.nan
    for pct in percentiles:
        threshold = np.percentile(df[score_col], pct)
        hot = df[score_col] > threshold
        fa_caught = int(((~df[hit_col]) & (~hot)).sum())
        hit_lost = int((df[hit_col] & (~hot)).sum())
        kept = df[hot]
        fa_rate_after = (~kept[hit_col]).sum() / len(kept) if len(kept) > 0 else np.nan
        rows.append(dict(
            percentile=pct, threshold=threshold,
            fa_caught_pct=100 * fa_caught / max(n_fa, 1),
            hit_lost_pct=100 * hit_lost / max(n_hit, 1),
            fa_rate_after_pct=100 * fa_rate_after if not np.isnan(fa_rate_after) else np.nan,
            signals_kept_pct=100 * hot.mean(),
        ))
    return pd.DataFrame(rows), 100 * fa_rate_before
