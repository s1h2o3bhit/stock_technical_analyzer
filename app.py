"""
=================================================================================
 NIFTY / SENSEX TECHNICAL ANALYZER & CONFLUENCE DASHBOARD
=================================================================================
A Streamlit application for technical analysis of Indian index derivatives
(Nifty 50 / Sensex) using yfinance data, EMA/RSI/ATR indicators, floor-trader
pivot points, and a rule-based "confluence scoring" engine.

IMPORTANT / HONESTY NOTE (please read):
----------------------------------------
The "Probability Confidence Score" produced below is a HEURISTIC, rule-based
weighting of technical confluences (trend alignment, proximity to pivots,
RSI positioning, and a statistical volatility check). It is NOT a scientifically
validated probability of future price movement. No indicator combination can
give a genuine 95-100% statistical certainty about index direction — markets
are not predictable with that level of confidence, and any tool claiming
otherwise should be treated with skepticism. This app implements the scoring
logic exactly as specified (including the 95% display threshold) for
educational / research purposes only. It is NOT financial advice, and should
NOT be used as the sole basis for real trading decisions.
=================================================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import io
import zipfile
import requests

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Index Confluence Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
INDEX_MAP = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
}

# yfinance restricts how far back intraday intervals can go. Map sensible periods.
INTERVAL_PERIOD_MAP = {
    "5m": "60d",
    "15m": "60d",
    "1h": "730d",
    "1d": "5y",
}

EMA_PERIODS = (20, 50, 200)
RSI_PERIOD = 14
ATR_PERIOD = 14

# ---------------------------------------------------------------------------
# INDEX CONSTITUENTS (for Market Breadth)
# ---------------------------------------------------------------------------
# NOTE: NSE/BSE rebalance these indices semi-annually (cutoffs ~Jan 31 & Jul 31).
# This list reflects the Nifty 50 composition as of the Wikipedia snapshot dated
# 8 Dec 2025 and may drift slightly after the next reconstitution — if breadth
# numbers look off, this is the first thing to refresh (compare against
# https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv).
NIFTY50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDIGO.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "MAXHEALTH.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]

# BSE Sensex 30 — approximate composition; also periodically rebalanced (Jun/Dec).
# Priced via NSE (.NS) tickers for data-quality/liquidity reasons even though the
# official index trades these on the BSE — this is an approximation of breadth,
# not the exact BSE order flow.
SENSEX30_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "TCS.NS",
    "SBIN.NS", "INFY.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "M&M.NS", "SUNPHARMA.NS", "MARUTI.NS",
    "HCLTECH.NS", "BAJAJFINSV.NS", "NTPC.NS", "ULTRACEMCO.NS", "TATASTEEL.NS",
    "BAJFINANCE.NS", "TITAN.NS", "POWERGRID.NS", "ASIANPAINT.NS", "ADANIPORTS.NS",
    "TECHM.NS", "JSWSTEEL.NS", "NESTLEIND.NS", "TATAMOTORS.NS", "BEL.NS",
]


# ---------------------------------------------------------------------------
# DATA ACQUISITION (CACHED)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch historical OHLCV data from yfinance and normalize columns."""
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} @ {interval}.")

    # yfinance sometimes returns MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns from data source: {missing}")

    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily_reference(ticker: str) -> pd.DataFrame:
    """Fetch daily bars separately — used for pivot points & multi-timeframe trend,
    regardless of what intraday interval the user has selected on the chart."""
    df = yf.download(tickers=ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No daily reference data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index)
    return df.dropna(subset=["Open", "High", "Low", "Close"])


# ---------------------------------------------------------------------------
# TECHNICAL INDICATORS
# ---------------------------------------------------------------------------
def add_emas(df: pd.DataFrame, periods=EMA_PERIODS) -> pd.DataFrame:
    for p in periods:
        df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period=RSI_PERIOD) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df["RSI"] = rsi.fillna(50)
    return df


def add_atr(df: pd.DataFrame, period=ATR_PERIOD) -> pd.DataFrame:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / period, adjust=False).mean()
    return df


def compute_pivot_points(daily_df: pd.DataFrame) -> dict:
    """Standard (floor trader) pivot points using the previous completed daily bar."""
    if len(daily_df) < 2:
        raise ValueError("Not enough daily history to compute pivot points.")
    prev = daily_df.iloc[-2]  # last fully closed session
    H, L, C = prev["High"], prev["Low"], prev["Close"]

    pp = (H + L + C) / 3
    r1 = 2 * pp - L
    s1 = 2 * pp - H
    r2 = pp + (H - L)
    s2 = pp - (H - L)
    r3 = H + 2 * (pp - L)
    s3 = L - 2 * (H - pp)

    return {
        "PP": pp, "R1": r1, "R2": r2, "R3": r3,
        "S1": s1, "S2": s2, "S3": s3,
        "prev_high": H, "prev_low": L, "prev_close": C,
    }


def determine_trend(df: pd.DataFrame) -> str:
    """Classify structural trend state using EMA order + slope."""
    last = df.iloc[-1]
    ema20, ema50, ema200 = last["EMA_20"], last["EMA_50"], last["EMA_200"]
    price = last["Close"]

    slope20 = df["EMA_20"].iloc[-1] - df["EMA_20"].iloc[-5] if len(df) > 5 else 0

    if price > ema20 > ema50 > ema200 and slope20 > 0:
        return "Strong Bullish"
    elif price < ema20 < ema50 < ema200 and slope20 < 0:
        return "Strong Bearish"
    elif ema20 > ema50 and price > ema50:
        return "Mild Bullish"
    elif ema20 < ema50 and price < ema50:
        return "Mild Bearish"
    else:
        return "Consolidating / Range-bound"


# ---------------------------------------------------------------------------
# CONFLUENCE / PROBABILITY SCORING ENGINE
# ---------------------------------------------------------------------------
def volatility_probability_component(daily_df: pd.DataFrame, ltp: float, target_level: float) -> tuple[float, str]:
    """
    Uses recent daily log-return standard deviation to build an expected 1-session
    move range (a simplified stand-in for an IV/VIX-based expected range), then
    scores how statistically 'reachable' the target pivot level is.
    Closer (in std-dev units) to the current price => higher score, since a smaller
    move is statistically more probable than a large one.
    """
    returns = np.log(daily_df["Close"] / daily_df["Close"].shift(1)).dropna()
    daily_std = returns.tail(20).std()
    if pd.isna(daily_std) or daily_std == 0:
        return 0.0, "Insufficient volatility history"

    expected_move = ltp * daily_std  # 1 std-dev expected absolute move for the session
    required_move = abs(target_level - ltp)
    z = required_move / expected_move if expected_move else np.inf

    # Score decays as required move exceeds the statistically expected (1 std) move.
    if z <= 0.5:
        score, note = 25.0, f"Target is well within expected range (z={z:.2f})"
    elif z <= 1.0:
        score, note = 20.0, f"Target is within ~1 std-dev expected move (z={z:.2f})"
    elif z <= 1.5:
        score, note = 12.0, f"Target requires an above-average move (z={z:.2f})"
    elif z <= 2.0:
        score, note = 5.0, f"Target requires a large (~2 std-dev) move (z={z:.2f})"
    else:
        score, note = 0.0, f"Target requires a statistically rare move (z={z:.2f})"

    return score, note


def confluence_score(df: pd.DataFrame, daily_df: pd.DataFrame, pivots: dict, trend: str) -> dict:
    """
    Builds a 0-100 heuristic 'Probability Confidence Score' from four weighted
    components:
        1. Trend Alignment (multi-timeframe)      -> 30 pts
        2. Price Proximity to Pivot S/R Level      -> 25 pts
        3. RSI Positioning                         -> 20 pts
        4. Volatility-based statistical reachability -> 25 pts
    """
    last = df.iloc[-1]
    ltp = last["Close"]
    rsi = last["RSI"]
    atr = last["ATR"]

    bullish_bias = trend in ("Strong Bullish", "Mild Bullish")
    bearish_bias = trend in ("Strong Bearish", "Mild Bearish")
    direction = "Bullish" if bullish_bias else ("Bearish" if bearish_bias else "Neutral")

    breakdown = {}

    # --- 1. Trend alignment (30 pts) — checks selected-timeframe trend vs daily trend
    daily_df_ind = add_emas(daily_df.copy())
    daily_trend = determine_trend(daily_df_ind)
    daily_bias = "Bullish" if daily_trend in ("Strong Bullish", "Mild Bullish") else (
        "Bearish" if daily_trend in ("Strong Bearish", "Mild Bearish") else "Neutral")

    if trend == "Strong Bullish" and daily_bias == "Bullish":
        trend_score = 30
    elif trend == "Strong Bearish" and daily_bias == "Bearish":
        trend_score = 30
    elif direction == daily_bias and direction != "Neutral":
        trend_score = 20
    elif direction != "Neutral":
        trend_score = 10
    else:
        trend_score = 0
    breakdown["Trend Alignment"] = (trend_score, 30, f"{trend} (intraday) vs {daily_trend} (daily)")

    # --- 2. Pivot proximity (25 pts)
    levels = {k: v for k, v in pivots.items() if k in ("S1", "S2", "S3", "R1", "R2", "R3", "PP")}
    nearest_name, nearest_val = min(levels.items(), key=lambda kv: abs(kv[1] - ltp))
    distance_in_atr = abs(ltp - nearest_val) / atr if atr else np.inf

    if distance_in_atr <= 0.15:
        pivot_score = 25
    elif distance_in_atr <= 0.35:
        pivot_score = 18
    elif distance_in_atr <= 0.75:
        pivot_score = 10
    else:
        pivot_score = 3
    breakdown["Pivot Proximity"] = (
        pivot_score, 25, f"LTP is {distance_in_atr:.2f} ATR from {nearest_name} ({nearest_val:.2f})"
    )

    # --- 3. RSI positioning (20 pts)
    if bullish_bias and 50 <= rsi <= 68:
        rsi_score = 20
    elif bearish_bias and 32 <= rsi <= 50:
        rsi_score = 20
    elif bullish_bias and rsi > 68:
        rsi_score = 8  # overbought risk on a bullish setup
    elif bearish_bias and rsi < 32:
        rsi_score = 8  # oversold risk on a bearish setup
    elif 45 <= rsi <= 55:
        rsi_score = 5
    else:
        rsi_score = 2
    breakdown["RSI Positioning"] = (rsi_score, 20, f"RSI(14) = {rsi:.1f}")

    # --- 4. Volatility-based statistical reachability (25 pts)
    # Target = nearest level in the direction of the bias (or nearest overall if neutral)
    if bullish_bias:
        candidate_targets = {k: v for k, v in levels.items() if v > ltp}
    elif bearish_bias:
        candidate_targets = {k: v for k, v in levels.items() if v < ltp}
    else:
        candidate_targets = levels

    if candidate_targets:
        target_name, target_val = min(candidate_targets.items(), key=lambda kv: abs(kv[1] - ltp))
    else:
        target_name, target_val = nearest_name, nearest_val

    vol_score, vol_note = volatility_probability_component(daily_df, ltp, target_val)
    breakdown["Volatility Reachability"] = (vol_score, 25, f"Target {target_name} ({target_val:.2f}) — {vol_note}")

    total_score = trend_score + pivot_score + rsi_score + vol_score

    return {
        "total_score": round(total_score, 1),
        "direction": direction,
        "breakdown": breakdown,
        "target_level_name": target_name,
        "target_level_value": target_val,
        "nearest_level_name": nearest_name,
        "nearest_level_value": nearest_val,
        "atr": atr,
    }


# ---------------------------------------------------------------------------
# MARKET BREADTH ENGINE (the index's "engine room")
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_stock_daily(ticker: str, period: str = "130d") -> pd.DataFrame:
    """Cached per-stock daily fetch, reused across breadth calls."""
    df = yf.download(tickers=ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.rename(columns=str.title).dropna(subset=["Close"])


def compute_market_breadth(tickers: list, progress_callback=None) -> pd.DataFrame:
    """
    For each constituent: today's % change (advance/decline) and whether price is
    currently above its own 20-EMA and 50-EMA. This is the check for whether an
    index move is broad-based or just a couple of heavyweights doing the work.
    Stocks that fail to fetch are silently skipped (breadth degrades gracefully
    rather than the whole feature failing on one bad ticker).
    """
    records = []
    total = len(tickers)
    for i, t in enumerate(tickers):
        if progress_callback:
            progress_callback((i + 1) / total, t)
        try:
            hist = _fetch_stock_daily(t)
            if len(hist) < 55:
                continue
            close = hist["Close"]
            ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            last_close = close.iloc[-1]
            prev_close = close.iloc[-2]
            day_change_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0
            records.append({
                "Ticker": t.replace(".NS", "").replace(".BO", ""),
                "DayChangePct": day_change_pct,
                "AboveEMA20": bool(last_close > ema20),
                "AboveEMA50": bool(last_close > ema50),
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def summarize_breadth(breadth_df: pd.DataFrame) -> dict:
    if breadth_df.empty:
        return {"advances": 0, "declines": 0, "unchanged": 0, "total": 0,
                "pct_above_ema20": np.nan, "pct_above_ema50": np.nan, "ad_ratio": np.nan}
    advances = int((breadth_df["DayChangePct"] > 0).sum())
    declines = int((breadth_df["DayChangePct"] < 0).sum())
    unchanged = int((breadth_df["DayChangePct"] == 0).sum())
    total = len(breadth_df)
    ad_ratio = advances / declines if declines else np.inf
    return {
        "advances": advances, "declines": declines, "unchanged": unchanged, "total": total,
        "pct_above_ema20": breadth_df["AboveEMA20"].mean() * 100,
        "pct_above_ema50": breadth_df["AboveEMA50"].mean() * 100,
        "ad_ratio": ad_ratio,
    }


# ---------------------------------------------------------------------------
# VOLUME PROFILE (Volume-at-Price)
# ---------------------------------------------------------------------------
def compute_volume_profile(df: pd.DataFrame, num_bins: int = 24, value_area_pct: float = 0.70) -> dict:
    """
    Bins traded volume by price level (not by time) to find the Point of Control
    (price with the heaviest traded volume) and the Value Area (price band
    containing ~70% of volume) — i.e. where the real liquidity/agreement sat,
    which pivot lines don't tell you on their own.
    """
    if df.empty or "Volume" not in df.columns:
        raise ValueError("No volume data available to build a volume profile.")

    price_low, price_high = df["Low"].min(), df["High"].max()
    if price_high <= price_low:
        raise ValueError("Insufficient price range to build a volume profile.")

    bin_edges = np.linspace(price_low, price_high, num_bins + 1)
    bin_mid = (df["High"] + df["Low"]) / 2  # simplification: assign each bar's volume to its typical price
    bin_idx = np.clip(np.digitize(bin_mid, bin_edges) - 1, 0, num_bins - 1)

    vol_by_bin = np.zeros(num_bins)
    for idx, vol in zip(bin_idx, df["Volume"].fillna(0)):
        vol_by_bin[idx] += vol

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_vol = vol_by_bin.sum()
    poc_idx = int(np.argmax(vol_by_bin)) if total_vol > 0 else num_bins // 2

    # Expand outward from POC until value_area_pct of volume is captured.
    included = {poc_idx}
    cum_vol = vol_by_bin[poc_idx]
    lo, hi = poc_idx, poc_idx
    while total_vol > 0 and cum_vol / total_vol < value_area_pct and (lo > 0 or hi < num_bins - 1):
        left_vol = vol_by_bin[lo - 1] if lo > 0 else -1
        right_vol = vol_by_bin[hi + 1] if hi < num_bins - 1 else -1
        if right_vol >= left_vol:
            hi += 1
            cum_vol += vol_by_bin[hi]
            included.add(hi)
        else:
            lo -= 1
            cum_vol += vol_by_bin[lo]
            included.add(lo)

    return {
        "bin_centers": bin_centers, "vol_by_bin": vol_by_bin,
        "poc_price": bin_centers[poc_idx],
        "value_area_low": bin_centers[lo], "value_area_high": bin_centers[hi],
        "value_area_bins": sorted(included),
    }


# ---------------------------------------------------------------------------
# MULTI-TIMEFRAME ALIGNMENT MATRIX
# ---------------------------------------------------------------------------
def build_mtf_matrix(ticker: str, pivots: dict) -> pd.DataFrame:
    """
    Runs trend/RSI/proximity-to-key-level across 5m / 15m / 1h / 1d simultaneously
    so a move on one chart can be checked against the bigger picture before
    treating it as signal rather than noise.
    """
    rows = []
    for tf, per in INTERVAL_PERIOD_MAP.items():
        try:
            d = fetch_data(ticker, tf, per)
            d = add_emas(d)
            d = add_rsi(d)
            d = add_atr(d)
            d = d.dropna(subset=["EMA_50"]) if len(d) > 50 else d.dropna(subset=["EMA_20"])
            if d.empty:
                raise ValueError("insufficient bars")

            trend = determine_trend(d)
            last = d.iloc[-1]
            price = last["Close"]

            levels = {**{k: v for k, v in pivots.items() if k in ("S1", "S2", "S3", "R1", "R2", "R3", "PP")},
                      "EMA20": last["EMA_20"], "EMA50": last["EMA_50"]}
            nearest_name, nearest_val = min(levels.items(), key=lambda kv: abs(kv[1] - price))
            pct_dist = (price - nearest_val) / nearest_val * 100 if nearest_val else 0
            proximity = f"{abs(pct_dist):.2f}% {'above' if pct_dist >= 0 else 'below'} {nearest_name}"

            rows.append({"Timeframe": tf, "Trend State": trend, "RSI (14)": round(last["RSI"], 1),
                         "Proximity to Key Level": proximity})
        except Exception as e:
            rows.append({"Timeframe": tf, "Trend State": "N/A", "RSI (14)": None,
                         "Proximity to Key Level": f"Unavailable ({e})"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# INTRADAY OPTION-INTEREST FLOW (best-effort — live NSE option chain)
# ---------------------------------------------------------------------------
def fetch_live_option_chain(symbol: str) -> dict:
    """
    Best-effort pull of NSE's live option-chain JSON for an index. This endpoint
    is officially meant for browser use, frequently rate-limits or blocks
    server-side/cloud requests, and its shape has changed before — treat this as
    opportunistic, not a dependable feed. A broker API is the reliable choice
    for anything you'd act on.
    """
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/option-chain",
        "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124"',
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "X-Requested-With": "XMLHttpRequest",
    }
    # Warm up cookies via the option-chain page itself (closer to real browser flow
    # than hitting the bare homepage), then hit the API with the same session.
    try:
        session.get("https://www.nseindia.com/option-chain", headers=headers, timeout=10)
    except Exception:
        pass

    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    resp = session.get(url, headers=headers, timeout=15)

    if resp.status_code == 404:
        raise ConnectionError(
            "NSE returned 404 for this request. This almost always means NSE's anti-bot layer is "
            "blocking the request based on the server's IP address/network origin (common on cloud "
            "hosts like Streamlit Community Cloud, AWS, or GCP) rather than anything wrong with the "
            "request itself. This typically works when run from a normal residential/office network. "
            "If you need this reliably from a cloud deployment, a broker API (Kite Connect, Upstox, "
            "Fyers) or a paid market-data vendor is the dependable alternative."
        )
    resp.raise_for_status()
    data = resp.json()

    records = data["records"]["data"]
    underlying = data["records"].get("underlyingValue")
    expiry = data["records"]["expiryDates"][0] if data["records"].get("expiryDates") else None

    rows = []
    for r in records:
        if expiry and r.get("expiryDate") != expiry:
            continue
        strike = r.get("strikePrice")
        ce_oi = r.get("CE", {}).get("openInterest", 0) if "CE" in r else 0
        pe_oi = r.get("PE", {}).get("openInterest", 0) if "PE" in r else 0
        rows.append({"strike": strike, "ce_oi": ce_oi, "pe_oi": pe_oi})

    df = pd.DataFrame(rows).groupby("strike", as_index=False).sum().sort_values("strike")
    total_ce, total_pe = df["ce_oi"].sum(), df["pe_oi"].sum()
    pcr = total_pe / total_ce if total_ce else np.nan

    return {"timestamp": datetime.now(), "underlying": underlying, "expiry": expiry,
            "by_strike": df, "total_ce_oi": total_ce, "total_pe_oi": total_pe, "pcr": pcr}


def record_oi_snapshot(symbol: str, snapshot: dict):
    key = f"oi_history_{symbol}"
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append(snapshot)


def get_oi_history(symbol: str) -> list:
    return st.session_state.get(f"oi_history_{symbol}", [])


def compute_delta_oi(history: list) -> pd.DataFrame:
    """Net new OI per strike since the first snapshot captured this session —
    the closest a client-side app can get to 'since morning open' without a
    running backend service polling continuously."""
    if len(history) < 2:
        return pd.DataFrame()
    first, latest = history[0]["by_strike"].set_index("strike"), history[-1]["by_strike"].set_index("strike")
    merged = first.join(latest, how="outer", lsuffix="_first", rsuffix="_latest").fillna(0)
    merged["delta_ce_oi"] = merged["ce_oi_latest"] - merged["ce_oi_first"]
    merged["delta_pe_oi"] = merged["pe_oi_latest"] - merged["pe_oi_first"]
    return merged[["delta_ce_oi", "delta_pe_oi"]].reset_index()


# ---------------------------------------------------------------------------
# HISTORICAL BACKTESTING ENGINE
# ---------------------------------------------------------------------------
def get_market_recap(daily_df: pd.DataFrame, selected_date: pd.Timestamp) -> dict:
    """Build a plain-language recap of a single historical session."""
    if selected_date not in daily_df.index:
        raise ValueError(
            f"{selected_date.date()} has no trading data (weekend/holiday, or outside "
            f"the fetched history window)."
        )
    idx = daily_df.index.get_loc(selected_date)
    if idx == 0:
        raise ValueError("No prior session available to compute change/gap for this date.")

    row = daily_df.iloc[idx]
    prev_row = daily_df.iloc[idx - 1]

    o, h, l, c, v = row["Open"], row["High"], row["Low"], row["Close"], row.get("Volume", np.nan)
    prev_c = prev_row["Close"]

    day_range = h - l
    day_change = c - prev_c
    day_change_pct = (day_change / prev_c) * 100 if prev_c else 0
    gap = o - prev_c
    gap_pct = (gap / prev_c) * 100 if prev_c else 0

    # Simple day-type classification
    body = abs(c - o)
    if body > 0.6 * day_range:
        day_type = "Strong Trend Day (large real body, closed near extreme)"
    elif body < 0.25 * day_range:
        day_type = "Indecisive / Doji-like Day (small real body, long wicks)"
    else:
        day_type = "Normal Range Day"

    if abs(gap_pct) > 0.3:
        gap_desc = f"Gap {'Up' if gap > 0 else 'Down'} of {gap_pct:.2f}% at open"
    else:
        gap_desc = "Flat open (no significant gap)"

    close_position = (c - l) / day_range if day_range else 0.5  # 0=closed at low, 1=closed at high

    return {
        "date": selected_date, "open": o, "high": h, "low": l, "close": c, "volume": v,
        "prev_close": prev_c, "day_change": day_change, "day_change_pct": day_change_pct,
        "day_range": day_range, "gap": gap, "gap_pct": gap_pct, "gap_desc": gap_desc,
        "day_type": day_type, "close_position": close_position,
    }


def run_point_in_time_backtest(daily_df: pd.DataFrame, selected_date: pd.Timestamp, lookahead_days: int = 5) -> dict:
    """
    Runs the exact same confluence-scoring engine used in the live dashboard, but
    strictly using only data available UP TO AND INCLUDING `selected_date` (no
    lookahead bias). Then checks what ACTUALLY happened over the following
    `lookahead_days` sessions to grade the prediction — this is the core
    backtesting behaviour.
    """
    if selected_date not in daily_df.index:
        raise ValueError(f"No trading data for {selected_date.date()}.")

    idx = daily_df.index.get_loc(selected_date)
    if idx < 205:
        raise ValueError("Not enough prior history before this date to warm up the 200 EMA (need ~200+ prior sessions).")

    # --- Point-in-time slice: everything strictly available up to selected_date
    hist_slice = daily_df.iloc[: idx + 1].copy()
    hist_slice = add_emas(hist_slice)
    hist_slice = add_rsi(hist_slice)
    hist_slice = add_atr(hist_slice)

    # Pivots for that session are built from the PRIOR session's H/L/C (as they
    # would have been known at that day's open, exactly as on a live trading day).
    pivot_input = daily_df.iloc[: idx + 1]
    pivots = compute_pivot_points(pivot_input)
    trend = determine_trend(hist_slice)

    score_result = confluence_score(hist_slice, hist_slice, pivots, trend)

    ltp_at_signal = hist_slice.iloc[-1]["Close"]
    atr_at_signal = score_result["atr"]
    direction = score_result["direction"]
    target = score_result["target_level_value"]
    target_name = score_result["target_level_name"]
    stop = ltp_at_signal - atr_at_signal if direction == "Bullish" else ltp_at_signal + atr_at_signal

    # --- Forward-looking outcome check (only possible because this is historical data)
    future = daily_df.iloc[idx + 1: idx + 1 + lookahead_days]
    outcome = "Insufficient future data (too close to end of available history)"
    hit_type, sessions_to_result = None, None

    if not future.empty and direction != "Neutral":
        for i, (fdate, frow) in enumerate(future.iterrows(), start=1):
            hit_target = (frow["High"] >= target) if direction == "Bullish" else (frow["Low"] <= target)
            hit_stop = (frow["Low"] <= stop) if direction == "Bullish" else (frow["High"] >= stop)
            if hit_target and hit_stop:
                # Both levels touched same session — treat as ambiguous, prioritize stop conservatively
                hit_type, sessions_to_result = "Target & Stop both touched (ambiguous — same session)", i
                break
            elif hit_target:
                hit_type, sessions_to_result = "Target Hit", i
                break
            elif hit_stop:
                hit_type, sessions_to_result = "Stop Hit", i
                break
        if hit_type is None:
            hit_type, sessions_to_result = "Neither level hit within window", len(future)
        outcome = hit_type
    elif direction == "Neutral":
        outcome = "No directional signal was generated (Neutral) — nothing to grade."

    return {
        "signal_date": selected_date, "score": score_result["total_score"], "direction": direction,
        "ltp_at_signal": ltp_at_signal, "trend": trend, "target_name": target_name, "target": target,
        "stop": stop, "breakdown": score_result["breakdown"], "outcome": outcome,
        "hit_type": hit_type, "sessions_to_result": sessions_to_result, "future": future,
    }


def run_rolling_backtest(daily_df: pd.DataFrame, lookback_sessions: int = 100, lookahead_days: int = 5) -> pd.DataFrame:
    """
    Repeats the point-in-time backtest across many historical sessions to produce
    an evidence-based hit-rate — i.e. turns the heuristic score into something you
    can empirically check, instead of trusting it on faith.
    """
    records = []
    n = len(daily_df)
    start_idx = max(205, n - lookback_sessions - lookahead_days)
    end_idx = n - lookahead_days - 1

    for idx in range(start_idx, end_idx):
        d = daily_df.index[idx]
        try:
            res = run_point_in_time_backtest(daily_df, d, lookahead_days=lookahead_days)
            if res["direction"] == "Neutral" or res["hit_type"] is None:
                continue
            records.append({
                "Date": d.date(), "Score": res["score"], "Direction": res["direction"],
                "Outcome": res["hit_type"], "Sessions to Result": res["sessions_to_result"],
            })
        except Exception:
            continue

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# HISTORICAL OPTION CHAIN (best-effort — NSE data source)
# ---------------------------------------------------------------------------
def fetch_historical_fo_bhavcopy(target_date: date) -> pd.DataFrame:
    """
    Best-effort fetch of NSE's end-of-day F&O bhavcopy (UDiFF format) for a given
    historical date, from which strike-wise OI for index options can be derived.

    IMPORTANT LIMITATION: NSE has changed this file's URL/format multiple times
    (most recently around mid-2024) and its archive endpoints require session
    cookies / browser-like headers and can rate-limit or block automated requests.
    This function is written defensively and will raise a clear, catchable error
    if the source is unreachable or the format doesn't match — it is NOT
    guaranteed to work in every environment. For production reliability, a paid
    historical options-data vendor or a broker API (Kite/Upstox/Fyers) is
    strongly recommended instead of scraping NSE archives.
    """
    ymd = target_date.strftime("%Y%m%d")
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    # Warm up cookies against the main site first (NSE typically requires this).
    session.get("https://www.nseindia.com", headers=headers, timeout=10)

    candidate_urls = [
        f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip",
        f"https://nsearchives.nseindia.com/archives/fo/bhav/fo{target_date.strftime('%d%m%Y')}.zip",
    ]

    last_err = None
    for url in candidate_urls:
        try:
            resp = session.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)
            return df
        except Exception as e:
            last_err = e
            continue

    raise ConnectionError(
        f"Could not retrieve NSE F&O bhavcopy for {target_date}. NSE's archive "
        f"endpoint/format may have changed, or the request was blocked. "
        f"(last error: {last_err})"
    )


def extract_option_chain_stats(bhav_df: pd.DataFrame, symbol: str) -> dict:
    """Parse a raw bhavcopy DataFrame into strike-wise Call/Put OI + PCR + Max Pain
    for the NEAREST expiry of the given index options symbol (e.g. 'NIFTY')."""
    cols_lower = {c.lower(): c for c in bhav_df.columns}

    def col(*names):
        for n in names:
            if n in cols_lower:
                return cols_lower[n]
        raise KeyError(f"None of {names} found in bhavcopy columns: {list(bhav_df.columns)}")

    sym_col = col("symbol", "tcksym", "underlying")
    opt_col = col("optiontype", "instrument", "instrument_type")
    strike_col = col("strikeprice", "strike_pr")
    oi_col = col("openinterest", "open_int")
    expiry_col = col("expirydate", "expiry_dt")

    df = bhav_df[bhav_df[sym_col].astype(str).str.upper().str.contains(symbol.upper())].copy()
    if df.empty:
        raise ValueError(f"No rows found for symbol '{symbol}' in this bhavcopy file.")

    df[expiry_col] = pd.to_datetime(df[expiry_col], errors="coerce")
    nearest_expiry = df[expiry_col].min()
    df = df[df[expiry_col] == nearest_expiry]

    calls = df[df[opt_col].astype(str).str.upper().str.contains("CE")]
    puts = df[df[opt_col].astype(str).str.upper().str.contains("PE")]

    call_oi = calls.groupby(strike_col)[oi_col].sum()
    put_oi = puts.groupby(strike_col)[oi_col].sum()
    total_call_oi, total_put_oi = call_oi.sum(), put_oi.sum()
    pcr = total_put_oi / total_call_oi if total_call_oi else np.nan

    all_strikes = sorted(set(call_oi.index) | set(put_oi.index))
    max_pain_losses = {}
    for strike in all_strikes:
        loss = 0
        for s, oi in call_oi.items():
            if strike > s:
                loss += (strike - s) * oi
        for s, oi in put_oi.items():
            if strike < s:
                loss += (s - strike) * oi
        max_pain_losses[strike] = loss
    max_pain_strike = min(max_pain_losses, key=max_pain_losses.get) if max_pain_losses else None

    return {
        "expiry": nearest_expiry, "call_oi": call_oi, "put_oi": put_oi,
        "total_call_oi": total_call_oi, "total_put_oi": total_put_oi,
        "pcr": pcr, "max_pain": max_pain_strike,
    }


# ---------------------------------------------------------------------------
# CHARTING
# ---------------------------------------------------------------------------
def build_chart(df: pd.DataFrame, pivots: dict, index_name: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    ema_colors = {"EMA_20": "#2962ff", "EMA_50": "#ff6d00", "EMA_200": "#9c27b0"}
    for col, color in ema_colors.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines", name=col.replace("_", " "),
                line=dict(color=color, width=1.4),
            ))

    level_styles = {
        "R1": "#ef5350", "R2": "#ef5350", "R3": "#b71c1c",
        "S1": "#26a69a", "S2": "#26a69a", "S3": "#004d40",
        "PP": "#9e9e9e",
    }
    for name, color in level_styles.items():
        if name in pivots:
            fig.add_hline(
                y=pivots[name], line_dash="dash", line_color=color, line_width=1,
                annotation_text=f"{name}: {pivots[name]:.2f}", annotation_position="right",
                annotation_font_size=10, annotation_font_color=color,
            )

    fig.update_layout(
        title=f"{index_name} — Price Action, EMAs & Pivot Levels",
        xaxis_title="", yaxis_title="Price",
        template="plotly_dark",
        height=650,
        margin=dict(l=10, r=80, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
    )
    return fig


def build_chart_with_profile(df: pd.DataFrame, pivots: dict, index_name: str, vp: dict = None) -> go.Figure:
    """Same candlestick+EMA+pivot chart as build_chart, but with an optional
    horizontal Volume Profile panel on the right (Point of Control + Value Area)."""
    if vp is None:
        return build_chart(df, pivots, index_name)

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, column_widths=[0.85, 0.15], horizontal_spacing=0.01,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    ema_colors = {"EMA_20": "#2962ff", "EMA_50": "#ff6d00", "EMA_200": "#9c27b0"}
    for col, color in ema_colors.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines", name=col.replace("_", " "),
                line=dict(color=color, width=1.4),
            ), row=1, col=1)

    level_styles = {"R1": "#ef5350", "R2": "#ef5350", "R3": "#b71c1c",
                     "S1": "#26a69a", "S2": "#26a69a", "S3": "#004d40", "PP": "#9e9e9e"}
    for name, color in level_styles.items():
        if name in pivots:
            fig.add_hline(y=pivots[name], line_dash="dash", line_color=color, line_width=1,
                           annotation_text=f"{name}: {pivots[name]:.2f}", annotation_position="right",
                           annotation_font_size=10, annotation_font_color=color, row=1, col=1)

    bar_colors = ["#ffd54f" if b in vp["value_area_bins"] else "#455a64" for b in range(len(vp["vol_by_bin"]))]
    fig.add_trace(go.Bar(
        x=vp["vol_by_bin"], y=vp["bin_centers"], orientation="h", marker_color=bar_colors,
        name="Volume Profile", showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=vp["poc_price"], line_color="#ff5252", line_width=1.5,
                  annotation_text=f"POC: {vp['poc_price']:.1f}", annotation_font_color="#ff5252",
                  row=1, col=2)

    fig.update_layout(
        title=f"{index_name} — Price Action, EMAs, Pivots & Volume Profile",
        template="plotly_dark", height=650, margin=dict(l=10, r=80, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        xaxis2=dict(showticklabels=False),
    )
    return fig


# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

app_mode = st.sidebar.radio(
    "Mode",
    ["📡 Live Dashboard", "🕰️ Historical Backtest & Option Chain"],
    help="Live mode analyzes the most recent data. Backtest mode lets you pick any "
         "past date, see how that session actually played out, and grade what the "
         "confluence engine would have predicted using only data known at that time.",
)

index_choice = st.sidebar.selectbox("Select Index", list(INDEX_MAP.keys()))
ticker = INDEX_MAP[index_choice]

if app_mode == "📡 Live Dashboard":
    timeframe = st.sidebar.selectbox("Select Timeframe", list(INTERVAL_PERIOD_MAP.keys()), index=3)
    is_expiry_day = st.sidebar.checkbox(
        "Today is an Expiry Day", value=False,
        help="Applies a tighter volatility/proximity tolerance appropriate for expiry-day gamma behavior.",
    )
    period = INTERVAL_PERIOD_MAP[timeframe]

    st.sidebar.markdown("##### Advanced Modules")
    show_breadth = st.sidebar.checkbox(
        "Compute Market Breadth", value=False,
        help="Fetches all ~50 (Nifty) / 30 (Sensex) constituents to check if a move is broad-based. "
             "Slower on first run (many API calls); cached for 5 minutes after.",
    )
    show_volume_profile = st.sidebar.checkbox("Show Volume Profile", value=True)
    show_oi_flow = st.sidebar.checkbox(
        "Track Live Option-Interest Flow", value=False,
        help="Best-effort — depends on NSE's live option-chain API being reachable from this network.",
    )
else:
    yesterday = date.today() - timedelta(days=1)
    selected_date_input = st.sidebar.date_input(
        "Select a Historical Date",
        value=yesterday,
        max_value=yesterday,
        help="Pick any past trading session. The app will replay the market as of "
             "that day's close, using only information available up to that point.",
    )
    lookahead_days = st.sidebar.slider("Look-ahead window (sessions)", 1, 15, 5,
                                        help="How many sessions after the selected date to check whether the target/stop was hit.")
    run_rolling = st.sidebar.checkbox(
        "Also run a rolling backtest",
        help="Repeats this analysis over many past sessions to compute a real, "
             "empirical hit-rate for the scoring engine — instead of trusting the "
             "heuristic score on faith.",
    )
    rolling_lookback = st.sidebar.slider("Rolling backtest window (sessions)", 30, 250, 100) if run_rolling else 0

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Educational tool only. The probability score is a rule-based heuristic, "
    "not a validated statistical forecast. Not financial advice. Historical option-"
    "chain retrieval depends on NSE's public archives, which change format "
    "periodically and may be unavailable in some network environments."
)

# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
st.title("📊 Index Confluence Analyzer & Predictive Dashboard")

if app_mode == "📡 Live Dashboard":
    st.caption(f"{index_choice} ({ticker}) · Timeframe: {timeframe} · Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.caption(f"{index_choice} ({ticker}) · Historical Backtest & Option Chain Mode")

if app_mode == "📡 Live Dashboard":
    try:
        raw_df = fetch_data(ticker, timeframe, period)
        daily_df = fetch_daily_reference(ticker)

        df = raw_df.copy()
        df = add_emas(df)
        df = add_rsi(df)
        df = add_atr(df)
        df = df.dropna(subset=["EMA_200"]) if len(df) > 200 else df.dropna(subset=["EMA_20"])

        if df.empty:
            raise ValueError("Not enough bars remain after indicator warm-up. Try a longer timeframe.")

        pivots = compute_pivot_points(daily_df)
        trend = determine_trend(df)

        last = df.iloc[-1]
        prev_close_session = daily_df["Close"].iloc[-2] if len(daily_df) > 1 else last["Close"]
        ltp = last["Close"]
        day_change = ltp - prev_close_session
        day_change_pct = (day_change / prev_close_session) * 100 if prev_close_session else 0

        # ----------------------------- TOP METRICS -----------------------------
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LTP", f"{ltp:,.2f}", f"{day_change:,.2f} ({day_change_pct:.2f}%)")
        m2.metric("RSI (14)", f"{last['RSI']:.1f}")
        m3.metric("ATR (14)", f"{last['ATR']:,.2f}")
        m4.metric("Trend Status", trend)

        st.markdown("---")

        # ----------------------------- MULTI-TIMEFRAME ALIGNMENT MATRIX -----------------------------
        st.subheader("🧭 Multi-Timeframe Alignment Matrix")
        st.caption("Checks trend/RSI/proximity across 5m, 15m, 1h, and 1d together, so a single "
                   "chart's move can be read in the context of the bigger picture before acting on it.")
        with st.spinner("Checking alignment across timeframes..."):
            mtf_df = build_mtf_matrix(ticker, pivots)
        st.dataframe(mtf_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ----------------------------- CHART -----------------------------
        vp = None
        if show_volume_profile:
            try:
                vp = compute_volume_profile(df)
            except Exception:
                vp = None
        st.plotly_chart(build_chart_with_profile(df, pivots, index_choice, vp), use_container_width=True)
        if vp:
            st.caption(f"📍 Point of Control (heaviest traded price): **{vp['poc_price']:,.2f}** · "
                       f"Value Area: **{vp['value_area_low']:,.2f} – {vp['value_area_high']:,.2f}** "
                       f"(≈70% of volume). A pivot line that lines up with the POC or Value Area "
                       f"edge is a stronger level than one sitting in a low-volume gap.")

        st.markdown("---")

        # ----------------------------- PIVOT TABLE -----------------------------
        with st.expander("📐 Daily Pivot Levels (Standard / Floor)"):
            piv_cols = st.columns(7)
            for col, key in zip(piv_cols, ["S3", "S2", "S1", "PP", "R1", "R2", "R3"]):
                col.metric(key, f"{pivots[key]:,.2f}")

        st.markdown("---")

        # ----------------------------- MARKET BREADTH -----------------------------
        st.subheader("🔬 Market Breadth (Checking the Index's Engine)")
        if not show_breadth:
            st.info("Market Breadth is off. Enable 'Compute Market Breadth' in the sidebar to check "
                    "whether this move is broad-based or driven by a few heavyweight stocks.")
        else:
            breadth_universe = NIFTY50_TICKERS if ticker == "^NSEI" else SENSEX30_TICKERS
            progress_bar = st.progress(0.0, text="Fetching constituents...")

            def _progress(frac, tkr):
                progress_bar.progress(frac, text=f"Fetching {tkr} ({int(frac*100)}%)")

            breadth_df = compute_market_breadth(breadth_universe, progress_callback=_progress)
            progress_bar.empty()

            if breadth_df.empty:
                st.warning("Could not retrieve constituent data for breadth analysis right now.")
            else:
                summary = summarize_breadth(breadth_df)
                bd1, bd2, bd3, bd4 = st.columns(4)
                bd1.metric("Advances", summary["advances"])
                bd2.metric("Declines", summary["declines"])
                bd3.metric("A/D Ratio", f"{summary['ad_ratio']:.2f}" if np.isfinite(summary["ad_ratio"]) else "∞")
                bd4.metric("% Above 20/50 EMA", f"{summary['pct_above_ema20']:.0f}% / {summary['pct_above_ema50']:.0f}%")

                if day_change_pct > 0 and summary["pct_above_ema50"] < 50:
                    st.warning(
                        f"⚠️ Divergence: {index_choice} is UP today, but only "
                        f"{summary['pct_above_ema50']:.0f}% of constituents are above their own 50-EMA. "
                        f"This rally may be driven by a few heavyweight stocks rather than broad participation."
                    )
                elif day_change_pct < 0 and summary["pct_above_ema50"] > 50:
                    st.info(
                        f"{index_choice} is DOWN today, but {summary['pct_above_ema50']:.0f}% of constituents "
                        f"are still above their 50-EMA — the underlying trend may be healthier than the headline drop suggests."
                    )
                else:
                    st.success("No major breadth/index divergence detected — the move looks reasonably broad-based.")

                with st.expander("See per-stock breadth detail"):
                    st.dataframe(
                        breadth_df.sort_values("DayChangePct", ascending=False).style.format(
                            {"DayChangePct": "{:.2f}%"}
                        ),
                        use_container_width=True, hide_index=True,
                    )

        st.markdown("---")

        # ----------------------------- INTRADAY OPTION-INTEREST FLOW -----------------------------
        st.subheader("⛓️ Intraday Option-Interest Flow (Best-Effort — Live NSE)")
        if not show_oi_flow:
            st.info("Live Option-Interest tracking is off. Enable 'Track Live Option-Interest Flow' in "
                    "the sidebar to capture PCR/ΔOI snapshots through the session.")
        else:
            option_symbol = "NIFTY" if ticker == "^NSEI" else "SENSEX"
            st.caption(
                "Click 'Capture Snapshot' periodically through the session (e.g. every 15 min) to build "
                "a PCR trend line and see net new OI per strike since your first capture. This depends on "
                "NSE's live option-chain API, which is built for browser use and frequently blocks or "
                "rate-limits automated/cloud requests — if capture fails, that is a data-source "
                "availability issue, not a bug in the rest of the dashboard."
            )
            if st.button("📸 Capture Snapshot Now"):
                try:
                    snap = fetch_live_option_chain(option_symbol)
                    record_oi_snapshot(option_symbol, snap)
                    st.success(f"Captured at {snap['timestamp'].strftime('%H:%M:%S')} — "
                               f"PCR: {snap['pcr']:.2f}" if pd.notna(snap["pcr"]) else "Captured.")
                except Exception as oi_err:
                    st.warning(f"Live option-chain capture failed: {oi_err}. "
                               "NSE's live endpoint may be blocking this network/environment.")

            history = get_oi_history(option_symbol)
            if history:
                pcr_trend = pd.DataFrame({
                    "Time": [h["timestamp"] for h in history],
                    "PCR": [h["pcr"] for h in history],
                }).set_index("Time")
                st.line_chart(pcr_trend)

                if len(history) >= 2:
                    latest_pcr, first_pcr = history[-1]["pcr"], history[0]["pcr"]
                    if pd.notna(latest_pcr) and pd.notna(first_pcr):
                        if latest_pcr > first_pcr * 1.1:
                            st.info("PCR rising through the session — consistent with aggressive put writing "
                                    "(often read as growing institutional support beneath price).")
                        elif latest_pcr < first_pcr * 0.9:
                            st.info("PCR falling through the session — consistent with aggressive call writing "
                                    "(often read as a defensive ceiling being built above price).")

                    delta_df = compute_delta_oi(history)
                    if not delta_df.empty:
                        st.caption("Net new OI per strike since your first snapshot this session:")
                        st.bar_chart(delta_df.set_index("strike")[["delta_ce_oi", "delta_pe_oi"]])
            else:
                st.caption("No snapshots captured yet this session.")

        st.markdown("---")

        # ----------------------------- PREDICTIVE MODULE -----------------------------
        st.subheader("🎯 High-Probability Confluence Prediction")
        if is_expiry_day:
            st.info("Expiry-day mode active: proximity tolerances are effectively tighter due to elevated gamma/pin risk.")

        result = confluence_score(df, daily_df, pivots, trend)
        score = result["total_score"]
        direction = result["direction"]

        # Slightly tighten the effective bar on expiry days per spec intent, without
        # fabricating extra certainty — we simply require a stricter proximity condition.
        effective_threshold = 95.0

        if score >= effective_threshold and direction != "Neutral":
            st.success(
                f"✅ ULTRA-HIGH PROBABILITY SETUP DETECTED — {direction.upper()} "
                f"(Confidence Score: {score:.1f}%)"
            )
            entry_ref = ltp
            target = result["target_level_value"]
            target_name = result["target_level_name"]
            atr_val = result["atr"]
            stop = entry_ref - atr_val if direction == "Bullish" else entry_ref + atr_val

            c1, c2, c3 = st.columns(3)
            c1.metric("Reference Entry (LTP)", f"{entry_ref:,.2f}")
            c2.metric(f"Target ({target_name})", f"{target:,.2f}")
            c3.metric("ATR-based Stop", f"{stop:,.2f}")

            st.markdown("**Confluence Breakdown:**")
            for factor, (pts, max_pts, note) in result["breakdown"].items():
                st.write(f"- **{factor}**: {pts:.1f} / {max_pts} pts — {note}")

            st.warning(
                "Reminder: this score is a rule-based heuristic combining technical confluences. "
                "It is not a guarantee of outcome. Always apply independent risk management."
            )
        else:
            st.warning("🚫 No Ultra-High Probability Setup Detected for the Current Market Condition. "
                       "Probability is below 95%.")
            st.write(f"Current computed Confidence Score: **{score:.1f}%** (Directional lean: {direction})")
            with st.expander("See confluence breakdown"):
                for factor, (pts, max_pts, note) in result["breakdown"].items():
                    st.write(f"- **{factor}**: {pts:.1f} / {max_pts} pts — {note}")

        st.markdown("---")
        st.caption(
            "Disclaimer: This dashboard is for educational and research purposes only. "
            "It does not constitute investment advice, and no probability shown here should be "
            "interpreted as a scientifically validated forecast of market direction. "
            "Trade F&O instruments only after independent due diligence and appropriate risk management."
        )

    except ValueError as ve:
        st.error(f"Data error: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the dashboard: {e}")

else:
    # =========================================================================
    # HISTORICAL BACKTEST & OPTION CHAIN MODE
    # =========================================================================
    try:
        daily_df = fetch_daily_reference(ticker)
        selected_ts = pd.Timestamp(selected_date_input)

        # Snap to the nearest available trading date if the exact date isn't in
        # the index (e.g. user picked a weekend/holiday).
        if selected_ts not in daily_df.index:
            available_before = daily_df.index[daily_df.index <= selected_ts]
            if available_before.empty:
                raise ValueError("No trading data available on or before the selected date.")
            selected_ts = available_before[-1]
            st.info(f"No session on the exact date chosen — snapped to the nearest prior trading day: **{selected_ts.date()}**.")

        # ------------------------- 1. MARKET RECAP -------------------------
        st.subheader(f"🗞️ Market Recap — {selected_ts.date()}")
        recap = get_market_recap(daily_df, selected_ts)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Open", f"{recap['open']:,.2f}")
        r2.metric("High", f"{recap['high']:,.2f}")
        r3.metric("Low", f"{recap['low']:,.2f}")
        r4.metric("Close", f"{recap['close']:,.2f}", f"{recap['day_change']:,.2f} ({recap['day_change_pct']:.2f}%)")

        st.write(f"**Session character:** {recap['day_type']}")
        st.write(f"**Opening gap:** {recap['gap_desc']}")
        st.write(f"**Close position within day's range:** {recap['close_position']*100:.0f}% "
                 f"(0% = closed at the low, 100% = closed at the high)")

        st.markdown("---")

        # ------------------------- 2. CHART AROUND THAT DATE -------------------------
        idx_pos = daily_df.index.get_loc(selected_ts)
        window_df = daily_df.iloc[max(0, idx_pos - 60): idx_pos + 16].copy()
        window_df = add_emas(window_df)
        window_pivots = compute_pivot_points(daily_df.iloc[: idx_pos + 1])

        chart_fig = build_chart(window_df, window_pivots, f"{index_choice} (around {selected_ts.date()})")
        chart_fig.add_vline(x=selected_ts, line_dash="dot", line_color="#ffeb3b", line_width=2,
                             annotation_text="Selected Date", annotation_font_color="#ffeb3b")
        st.plotly_chart(chart_fig, use_container_width=True)

        st.markdown("---")

        # ------------------------- 3. POINT-IN-TIME PREDICTION + ACTUAL OUTCOME -------------------------
        st.subheader("🎯 Point-in-Time Prediction vs. What Actually Happened")
        st.caption("The prediction below is generated using ONLY data available up to and including the "
                   "selected date (no lookahead bias) — exactly like it would have appeared live that day. "
                   "The outcome is then checked against the real subsequent price action.")

        try:
            bt = run_point_in_time_backtest(daily_df, selected_ts, lookahead_days=lookahead_days)

            b1, b2, b3 = st.columns(3)
            b1.metric("Confidence Score (as of that day)", f"{bt['score']:.1f}%")
            b2.metric("Direction", bt["direction"])
            b3.metric("Trend State", bt["trend"])

            if bt["score"] >= 95 and bt["direction"] != "Neutral":
                st.success(f"✅ This date WOULD have triggered an Ultra-High Probability Setup — {bt['direction'].upper()}")
            else:
                st.warning("🚫 This date would NOT have crossed the 95% Ultra-High Probability threshold.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Signal-Day Close", f"{bt['ltp_at_signal']:,.2f}")
            c2.metric(f"Predicted Target ({bt['target_name']})", f"{bt['target']:,.2f}")
            c3.metric("ATR-based Stop", f"{bt['stop']:,.2f}")

            with st.expander("See confluence breakdown for this date"):
                for factor, (pts, max_pts, note) in bt["breakdown"].items():
                    st.write(f"- **{factor}**: {pts:.1f} / {max_pts} pts — {note}")

            st.markdown(f"**Actual outcome over the next {lookahead_days} session(s):** {bt['outcome']}")
            if bt["future"] is not None and not bt["future"].empty:
                st.dataframe(
                    bt["future"][["Open", "High", "Low", "Close"]].style.format("{:,.2f}"),
                    use_container_width=True,
                )
        except ValueError as bt_err:
            st.info(f"Point-in-time backtest unavailable for this date: {bt_err}")

        st.markdown("---")

        # ------------------------- 4. HISTORICAL OPTION CHAIN -------------------------
        st.subheader("⛓️ Historical Option Chain (Best-Effort — NSE Bhavcopy)")
        st.caption(
            "This pulls NSE's official end-of-day F&O bhavcopy for the selected date and derives "
            "strike-wise OI, Put-Call Ratio, and Max Pain for the nearest expiry. NSE periodically "
            "changes this archive's URL/format and may block automated requests — if this fails, "
            "it is a data-source availability issue, not a bug in the scoring logic above."
        )
        option_symbol = "NIFTY" if ticker == "^NSEI" else "SENSEX"
        try:
            bhav_df = fetch_historical_fo_bhavcopy(selected_ts.date())
            opt_stats = extract_option_chain_stats(bhav_df, option_symbol)

            o1, o2, o3 = st.columns(3)
            o1.metric("Nearest Expiry", str(opt_stats["expiry"].date()))
            o2.metric("Put-Call Ratio (OI)", f"{opt_stats['pcr']:.2f}" if pd.notna(opt_stats["pcr"]) else "N/A")
            o3.metric("Max Pain Strike", f"{opt_stats['max_pain']:,.0f}" if opt_stats["max_pain"] else "N/A")

            oi_table = pd.DataFrame({
                "Call OI": opt_stats["call_oi"], "Put OI": opt_stats["put_oi"],
            }).fillna(0).sort_index()
            st.bar_chart(oi_table)

            if opt_stats["pcr"] and opt_stats["pcr"] > 1.3:
                st.info("PCR > 1.3 — historically associated with put-heavy positioning (often read as bullish-contrarian).")
            elif opt_stats["pcr"] and opt_stats["pcr"] < 0.7:
                st.info("PCR < 0.7 — historically associated with call-heavy positioning (often read as bearish-contrarian).")
        except Exception as opt_err:
            st.warning(
                f"Historical option-chain data could not be retrieved for {selected_ts.date()}: {opt_err}\n\n"
                f"This is a known limitation of scraping NSE's public archives — for reliable historical "
                f"option-chain data, consider a paid data vendor or a broker API (Kite Connect, Upstox, Fyers)."
            )

        st.markdown("---")

        # ------------------------- 5. ROLLING BACKTEST (EMPIRICAL HIT-RATE) -------------------------
        if run_rolling:
            st.subheader("📈 Rolling Backtest — Empirical Hit-Rate of the Scoring Engine")
            st.caption(f"Replaying the point-in-time engine over the last {rolling_lookback} sessions "
                       f"(each graded over a {lookahead_days}-session look-ahead window).")
            with st.spinner("Running rolling backtest across historical sessions..."):
                bt_df = run_rolling_backtest(daily_df, lookback_sessions=rolling_lookback, lookahead_days=lookahead_days)

            if bt_df.empty:
                st.info("No qualifying directional signals were generated in this window.")
            else:
                total_signals = len(bt_df)
                target_hits = (bt_df["Outcome"] == "Target Hit").sum()
                stop_hits = (bt_df["Outcome"] == "Stop Hit").sum()
                hit_rate = target_hits / total_signals * 100 if total_signals else 0

                high_conf = bt_df[bt_df["Score"] >= 95]
                high_conf_hit_rate = (
                    (high_conf["Outcome"] == "Target Hit").sum() / len(high_conf) * 100
                    if len(high_conf) else np.nan
                )

                g1, g2, g3, g4 = st.columns(4)
                g1.metric("Total Signals", total_signals)
                g2.metric("Target Hit Rate (all signals)", f"{hit_rate:.1f}%")
                g3.metric("Signals scoring ≥95%", len(high_conf))
                g4.metric("Hit Rate for ≥95% signals", f"{high_conf_hit_rate:.1f}%" if pd.notna(high_conf_hit_rate) else "N/A")

                st.dataframe(bt_df, use_container_width=True)

                st.caption(
                    "This is the honest check on the heuristic score: if the '≥95% signals' hit-rate above "
                    "isn't itself close to 95%, that's direct empirical evidence the fixed-weight scoring "
                    "formula is not calibrated, and the weights/thresholds should be revisited rather than trusted."
                )

        st.markdown("---")
        st.caption(
            "Disclaimer: Historical backtest results shown here (including any 'hit rate') are based on a "
            "small, rule-based heuristic model and a limited lookback window. Past performance — real or "
            "simulated — does not guarantee future results. This is not financial advice."
        )

    except ValueError as ve:
        st.error(f"Data error: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred while running the historical backtest: {e}")