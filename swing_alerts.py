import datetime
import time
import yfinance as yf
import pytz
import pandas as pd
import os
from helpers import send_email

FROM_EMAIL = "amitadiraju3@gmail.com"
TO_EMAIL = "amith.adiraju@gmail.com"

'''
CSV File Format
Ticker,Support,Resistance,Vol_Length(Simple Look back Window)
AAPL,165.50,175.20,14
MDGL,90.00,102.50,14
TSLA,200.00,215.00,14

Monitoring two 1h candles
# Change "2h" to "1h"
df_2h = raw_data.resample("1h", origin="09:30:00").agg(agg_rules).dropna()
BREAKOUT REQUIRES ADDITIONAL CONFIDENCE AND MOMENTUM THAN HISTORICAL KEY LEVEL BOUNCE.
'''

last_alerted_breakout_candles = {}
last_alerted_keylevel_candles = {}
CHECK_INTERVAL_SECONDS = 120



# Global cache to prevent exhausting the API with SPY requests
SPY_CACHE = {"last_fetched_date": None, "is_bullish": True, "is_bearish": True}

def get_spy_regime():
    """Fetches SPY status exactly once per calendar day (EST strictly)."""
    global SPY_CACHE
    
    # FIX: Force 'today' to be evaluated in EST. 
    # If you are in IST, your local clock might be on Tuesday while NY is still on Monday.
    eastern = pytz.timezone("US/Eastern")
    now_est = datetime.datetime.now(eastern)
    today_str = now_est.strftime("%Y-%m-%d")

    if SPY_CACHE.get("last_fetched_date") == today_str:
        return SPY_CACHE["is_bullish"], SPY_CACHE["is_bearish"]

    print(f"[{now_est.strftime('%H:%M:%S')} EST] Refreshing global SPY Market Regime...")
    try:
        stock = yf.Ticker("SPY")
        spy_data = stock.history(period="3mo", interval="1d", auto_adjust=False)
        
        if not spy_data.empty and len(spy_data) > 20:
            spy_data["SMA_20"] = spy_data["Close"].rolling(window=20).mean()
            
            latest_spy_close = float(spy_data["Close"].iloc[-1])
            latest_spy_sma = float(spy_data["SMA_20"].iloc[-1])

            SPY_CACHE["is_bullish"] = latest_spy_close > latest_spy_sma
            SPY_CACHE["is_bearish"] = latest_spy_close < latest_spy_sma
            SPY_CACHE["last_fetched_date"] = today_str
    except Exception as e:
        print(f"Warning: SPY fetch failed ({e}). Defaulting to un-gated mode.")
        SPY_CACHE["is_bullish"] = True
        SPY_CACHE["is_bearish"] = True

    return SPY_CACHE["is_bullish"], SPY_CACHE["is_bearish"]

# TO CONFIRM LONG AND SHORT ENTRIES BASED ON SCORE OR CRUDE ANDs For Key Level based trades
def evaluate_bounce_confidence(
    swept,
    reclaimed,
    ms_flip,
    not_overextended, 
    setup_present,
    volume_confirmed,
    spy_aligned, 
    wick_rejection,
    strong_body, 
    confirmation_type="weight",
    confidence_threshold=85
):
    """
    Unified evaluation engine for both Long and Short setups.
    Eliminates code duplication by abstracting structural direction.
    """
    # 1. Non-Negotiable Structural Vetoes
    structural_vetoes_pass = swept and reclaimed and ms_flip and not_overextended
    if not structural_vetoes_pass:
        return False

    # 2. Strict Boolean Confirmation Pathway
    if confirmation_type != "weight":
        return (spy_aligned and setup_present and strong_body and wick_rejection and volume_confirmed)

    # 3. Weighted Scoring Pathway
    score = 0
    if setup_present:    score += 30  # Exhaustion / Divergence
    if volume_confirmed: score += 25  # Institutional buying/selling volume
    if spy_aligned:      score += 15  # Macro market regime alignment
    if wick_rejection:   score += 15  # Price action rejection signature
    if strong_body:      score += 10  # Momentum candle close profile

    return score >= confidence_threshold, score


# TO CONFIRM LONG AND SHORT ENTRIES BASED ON SCORE OR CRUDE ANDs for Breakout Trades
def evaluate_breakout_confidence(
    started_correct_side,
    closed_past_level,
    not_overextended,
    volume_surge,
    strong_close,
    originated_nearby,
    spy_aligned,
    confirmation_type="weight",
    confidence_threshold=85
):
    """
    Dedicated evaluation engine for Breakout setups.
    """
    # 1. Non-Negotiable Structural Vetoes
    structural_vetoes_pass = started_correct_side and closed_past_level and not_overextended
    if not structural_vetoes_pass:
        return False, 0

    # 2. Strict Boolean Confirmation Pathway
    if confirmation_type != "weight":
        is_confirmed = (spy_aligned and volume_surge and strong_close and originated_nearby)
        return is_confirmed, 100 if is_confirmed else 0

    # 3. Weighted Scoring Pathway
    score = 0
    if volume_surge:      score += 35  # Institutional backing is critical for breakouts
    if strong_close:      score += 25  # No massive wicks (avoids bull/bear traps)
    if originated_nearby: score += 25  # Price coiled before breaking (avoids exhaustion)
    if spy_aligned:       score += 15  # Macro market regime tailwind

    return score >= confidence_threshold, score

# Key level based entry confirmation
def check_bounce_entry_confirmation(TICKER, SUPPORT_LEVEL, RESISTANCE_LEVEL, VOL_LENGTH, FROM_EMAIL, TO_EMAIL):
    global last_alerted_keylevel_candles
    CONFIDENCE_THRESHOLD = 55

    # --- STEP 1: MARKET CONTEXT & DATA ACQUISITION ---
    spy_is_bullish, spy_is_bearish = get_spy_regime()

    raw_data = yf.download(tickers=TICKER, period="1mo", interval="30m", progress=False)
    if raw_data.empty or len(raw_data) < 100:
        return

    # FIX: Dynamic timezone anchoring to support running on IST/PST clocks
    raw_data.index = raw_data.index.tz_convert("US/Eastern")
    first_data_date = raw_data.index[0].date()
    data_timezone = raw_data.index.tz
    dynamic_origin = pd.Timestamp(f"{first_data_date} 09:30:00", tz=data_timezone)

    agg_rules = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    df_2h = raw_data.resample("2h", origin=dynamic_origin).agg(agg_rules).dropna().copy()

    if len(df_2h) < (VOL_LENGTH + 10):
        return

    # Technical Overlays
    df_2h["Vol_SMA"] = df_2h["Volume"].rolling(window=VOL_LENGTH).mean()
    tr1 = df_2h["High"] - df_2h["Low"]
    tr2 = (df_2h["High"] - df_2h["Close"].shift(1)).abs()
    tr3 = (df_2h["Low"] - df_2h["Close"].shift(1)).abs()
    df_2h["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    # Calculate 2H RSI for Divergence Math
    delta = df_2h["Close"].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    df_2h["RSI"] = 100 - (100 / (1 + (gain / loss)))

    # --- STEP 2: STRUCTURAL PIVOT TRACKING ---
    df_2h["Is_Swing_High"] = (df_2h["High"] == df_2h["High"].rolling(window=5, center=True).max())
    df_2h["Is_Swing_Low"] = (df_2h["Low"] == df_2h["Low"].rolling(window=5, center=True).min())

    df_2h["Last_Confirmed_High"] = df_2h["High"].where(df_2h["Is_Swing_High"]).shift(2).ffill()
    df_2h["Last_Confirmed_Low"] = df_2h["Low"].where(df_2h["Is_Swing_Low"]).shift(2).ffill()
    df_2h["RSI_at_High"] = df_2h["RSI"].where(df_2h["Is_Swing_High"]).shift(2).ffill()
    df_2h["RSI_at_Low"] = df_2h["RSI"].where(df_2h["Is_Swing_Low"]).shift(2).ffill()

    # Extract Closed Target Bars
    latest_closed_bar = df_2h.iloc[-2]
    prev_closed_bar = df_2h.iloc[-3]
    current_candle_time = str(df_2h.index[-2])

    if TICKER not in last_alerted_keylevel_candles:
        last_alerted_keylevel_candles[TICKER] = None

    # Bar Coordinate Unpacking
    o2h, h2h, l2h, c2h = (
        float(latest_closed_bar["Open"]), float(latest_closed_bar["High"]),
        float(latest_closed_bar["Low"]), float(latest_closed_bar["Close"])
    )
    v2h, v2h_avg, v2h_prev = float(latest_closed_bar["Volume"]), float(latest_closed_bar["Vol_SMA"]), float(prev_closed_bar["Volume"])
    atr, current_rsi = float(latest_closed_bar["ATR"]), float(latest_closed_bar["RSI"])
    candle_range = h2h - l2h

    if candle_range <= 0:
        return

    last_structural_high = float(latest_closed_bar["Last_Confirmed_High"])
    last_structural_low = float(latest_closed_bar["Last_Confirmed_Low"])

    # --- STEP 3: REACTIONARY VOLUME vs APPROACH VOLUME ---
    approach_vol_avg = float(df_2h["Volume"].iloc[-6:-3].mean())
    vol_drying_up = approach_vol_avg < v2h_avg
    volume_confirmed = (v2h > v2h_avg) and (v2h_prev > v2h_avg)

    # --- STEP 4: LONG SIDE CONFIRMATION PIPELINE ---
    bull_divergence = (l2h < last_structural_low) and (current_rsi > float(latest_closed_bar["RSI_at_Low"]))
    bullish_setup_present = bull_divergence or vol_drying_up

    price_long_confirmed, long_score = evaluate_bounce_confidence(
        swept=(l2h < SUPPORT_LEVEL),
        reclaimed=(c2h > SUPPORT_LEVEL),
        ms_flip=(c2h > last_structural_high),
        not_overextended=(c2h <= (SUPPORT_LEVEL + (0.5 * atr))),
        setup_present=bullish_setup_present,
        volume_confirmed=volume_confirmed,
        spy_aligned=spy_is_bullish,
        wick_rejection=((min(o2h, c2h) - l2h) > (candle_range * 0.3)),
        strong_body=((c2h - o2h) > (candle_range * 0.5)),
        confirmation_type="weight",
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # --- STEP 5: SHORT SIDE CONFIRMATION PIPELINE ---
    bear_divergence = (h2h > last_structural_high) and (current_rsi < float(latest_closed_bar["RSI_at_High"]))
    bearish_setup_present = bear_divergence or vol_drying_up

    price_short_confirmed, short_score = evaluate_bounce_confidence(
        swept=(h2h > RESISTANCE_LEVEL),
        reclaimed=(c2h < RESISTANCE_LEVEL),
        ms_flip=(c2h < last_structural_low),
        not_overextended=(c2h >= (RESISTANCE_LEVEL - (0.5 * atr))),
        setup_present=bearish_setup_present,
        volume_confirmed=volume_confirmed,
        spy_aligned=spy_is_bearish,
        wick_rejection=((h2h - max(o2h, c2h)) > (candle_range * 0.3)),
        strong_body=((o2h - c2h) > (candle_range * 0.5)),
        confirmation_type="weight",
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # --- STEP 6: EXECUTE ALERTS ---
    if price_long_confirmed and (current_candle_time != last_alerted_keylevel_candles[TICKER]):
        last_alerted_keylevel_candles[TICKER] = current_candle_time
        send_email(
            subject=f"*** SWING ALERT: A+ BULLISH KEY LEVEL CONFIRMED FOR : {TICKER} ***", # FIX: Added 'f' prefix
            body=f"""Use these ballpark entries , stop losses and take profits.
            Entry: {c2h}, Stop Loss (Place Below This): {l2h} and Take Profit: {RESISTANCE_LEVEL}.
            Long Confirmation Score: {long_score} with threshold at: {CONFIDENCE_THRESHOLD}
            """,
            from_email=FROM_EMAIL,
            to_email=TO_EMAIL,
            attachment=None
        )

    elif price_short_confirmed and (current_candle_time != last_alerted_keylevel_candles[TICKER]):
        last_alerted_keylevel_candles[TICKER] = current_candle_time
        send_email(
            subject=f"*** SWING ALERT: A+ BEARISH KEY LEVEL CONFIRMED FOR : {TICKER} ***", # FIX: Added 'f' prefix
            body=f"""Use these ballpark entries , stop losses and take profits.
            Entry: {c2h}, Stop Loss (Place Above This): {h2h} and Take Profit: {SUPPORT_LEVEL}.
            Short Confirmation Score: {short_score} with threshold at :{CONFIDENCE_THRESHOLD}
            """,
            from_email=FROM_EMAIL,
            to_email=TO_EMAIL,
            attachment=None
        )


# Breakout based entry confirmation
def check_breakout_entry_confirmation(TICKER, SUPPORT_LEVEL, RESISTANCE_LEVEL, VOL_LENGTH, FROM_EMAIL, TO_EMAIL):
    global last_alerted_breakout_candles
    CONFIDENCE_THRESHOLD = 70

    spy_is_bullish, spy_is_bearish = get_spy_regime()

    raw_data = yf.download(tickers=TICKER, period="1mo", interval="30m", progress=False)
    if raw_data.empty or len(raw_data) < 100:
        return

    # FIX: Dynamic timezone anchoring to support running on IST/PST clocks
    raw_data.index = raw_data.index.tz_convert("US/Eastern")
    first_data_date = raw_data.index[0].date()
    data_timezone = raw_data.index.tz
    dynamic_origin = pd.Timestamp(f"{first_data_date} 09:30:00", tz=data_timezone)

    agg_rules = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    df_2h = raw_data.resample("2h", origin=dynamic_origin).agg(agg_rules).dropna().copy()

    if len(df_2h) < (VOL_LENGTH + 10):
        return

    # Technical Overlays
    df_2h["Vol_SMA"] = df_2h["Volume"].rolling(window=VOL_LENGTH).mean()
    tr1 = df_2h["High"] - df_2h["Low"]
    tr2 = (df_2h["High"] - df_2h["Close"].shift(1)).abs()
    tr3 = (df_2h["Low"] - df_2h["Close"].shift(1)).abs()
    df_2h["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    # Extract Target Bars
    latest_closed_bar = df_2h.iloc[-2]
    current_candle_time = str(df_2h.index[-2])

    if TICKER not in last_alerted_breakout_candles:
        last_alerted_breakout_candles[TICKER] = None

    o2h, h2h, l2h, c2h = (
        float(latest_closed_bar["Open"]), float(latest_closed_bar["High"]),
        float(latest_closed_bar["Low"]), float(latest_closed_bar["Close"])
    )
    v2h = float(latest_closed_bar["Volume"])
    v2h_avg = float(latest_closed_bar["Vol_SMA"])
    atr = float(latest_closed_bar["ATR"])
    candle_range = h2h - l2h

    if candle_range <= 0 or pd.isna(atr):
        return

    volume_surge = v2h > (v2h_avg * 1.2)

    # --- LONG SIDE (BULLISH BREAKOUT) ---
    price_long_breakout, long_score = evaluate_breakout_confidence(
        started_correct_side=(o2h < (RESISTANCE_LEVEL + (0.2 * atr))),
        closed_past_level=(c2h > RESISTANCE_LEVEL),
        not_overextended=((c2h - RESISTANCE_LEVEL) <= (0.8 * atr)),
        volume_surge=volume_surge,
        strong_close=((h2h - c2h) <= (candle_range * 0.25)),
        originated_nearby=((RESISTANCE_LEVEL - o2h) <= (1.0 * atr)),
        spy_aligned=spy_is_bullish,
        confirmation_type="weight",
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # --- SHORT SIDE (BEARISH BREAKDOWN) ---
    price_short_breakout, short_score = evaluate_breakout_confidence(
        started_correct_side=(o2h > (SUPPORT_LEVEL - (0.2 * atr))),
        closed_past_level=(c2h < SUPPORT_LEVEL),
        not_overextended=((SUPPORT_LEVEL - c2h) <= (0.8 * atr)),
        volume_surge=volume_surge,
        strong_close=((c2h - l2h) <= (candle_range * 0.25)),
        originated_nearby=((o2h - SUPPORT_LEVEL) <= (1.0 * atr)),
        spy_aligned=spy_is_bearish,
        confirmation_type="weight",
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # --- EXECUTING ALERTS ---
    if price_long_breakout and (current_candle_time != last_alerted_breakout_candles[TICKER]):
        last_alerted_breakout_candles[TICKER] = current_candle_time
        send_email(
            subject=f"*** SWING ALERT: A+ BULLISH BREAKOUT CONFIRMED FOR : {TICKER} ***", # FIX: Added 'f' prefix
            body=f"""Use these ballpark entries , stop losses and take profits. 
            Entry: {c2h}, Stop Loss (Place Below This): {RESISTANCE_LEVEL} and Take Profit: {c2h + (RESISTANCE_LEVEL - SUPPORT_LEVEL)}.
            Long Confirmation Score: {long_score} with threshold at: {CONFIDENCE_THRESHOLD}
            """,
            from_email=FROM_EMAIL,
            to_email=TO_EMAIL,
            attachment=None
        )
        
    elif price_short_breakout and (current_candle_time != last_alerted_breakout_candles[TICKER]):
        last_alerted_breakout_candles[TICKER] = current_candle_time
        send_email(
            subject=f"*** SWING ALERT: A+ BEARISH BREAKOUT CONFIRMED FOR : {TICKER} ***", # FIX: Added 'f' prefix
            body=f"""Use these ballpark entries , stop losses and take profits.
            Entry: {c2h}, Stop Loss (Place Above This): {SUPPORT_LEVEL} and Take Profit: {c2h - (RESISTANCE_LEVEL - SUPPORT_LEVEL)}.
            Short Confirmation Score: {short_score} with threshold at: {CONFIDENCE_THRESHOLD}
            """,
            from_email=FROM_EMAIL,
            to_email=TO_EMAIL,
            attachment=None
        )


if __name__ == "__main__":
    eastern = pytz.timezone("US/Eastern")

    print(
        "Initializing Multi-Ticker Automated Reversal Script Engine...",
        flush=True,
    )

    while True:
        try:
            now = datetime.datetime.now(eastern)
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

            market_open_today = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close_today = now.replace(hour=16, minute=0, second=0, microsecond=0)

            # Check if market is currently open (Weekday AND between 9:30 AM - 4:00 PM EST)
            if now.weekday() < 5 and market_open_today <= now <= market_close_today:
                print(f"[{timestamp} EST] Market open. Scanning watchlist...", flush=True)

                if os.path.exists("watchlist.csv"):
                    watchlist = pd.read_csv(
                        "watchlist.csv",
                        usecols=["Ticker", "Support", "Resistance", "Vol_Length", "Trade_Type"],
                        dtype={
                            "Ticker": "string",
                            "Support": "float64",
                            "Resistance": "float64",
                            "Vol_Length": "int64",
                            "Trade_Type": "string",
                        },
                    )
                    watchlist["Trade_Type"] = watchlist["Trade_Type"].astype("string").str.strip().str.lower()

                    for ticker, support, resistance, vol_len, trade_type in watchlist.itertuples(index=False, name=None):
                        ticker = str(ticker).strip()

                        if trade_type == "breakout":
                            check_breakout_entry_confirmation(ticker, support, resistance, vol_len, FROM_EMAIL, TO_EMAIL)
                        elif trade_type == "bounce":
                            check_bounce_entry_confirmation(ticker, support, resistance, vol_len, FROM_EMAIL, TO_EMAIL)

                        time.sleep(2)
                else:
                    print(f"[{timestamp} EST] Alert: watchlist.csv not found.", flush=True)

                print(f"Cycle complete. Sleeping for {CHECK_INTERVAL_SECONDS}s...", flush=True)
                time.sleep(CHECK_INTERVAL_SECONDS)

            # MARKET IS CLOSED: Calculate exact time until next open
            else:
                
                next_open = market_open_today
                
                # If we are already past today's close, look to tomorrow
                if now >= market_close_today:
                    next_open += datetime.timedelta(days=1)
                
                # Push forward if next_open falls on a Saturday (5) or Sunday (6)
                while next_open.weekday() >= 5:
                    next_open += datetime.timedelta(days=1)
                
                # Calculate the exact seconds to sleep
                sleep_seconds = (next_open - now).total_seconds()
                hours, remainder = divmod(sleep_seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                
                print(
                    f"[{timestamp} EST] Market closed. Hibernating for {int(hours)}h {int(minutes)}m "
                    f"until {next_open.strftime('%A, %Y-%m-%d %H:%M:%S')} EST...", 
                    flush=True
                )
                
                # Add 1 second buffer to ensure we wake up slightly past the threshold
                time.sleep(sleep_seconds + 1)

        except Exception as global_error:
            error_time = datetime.datetime.now(pytz.timezone("US/Eastern")).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{error_time} EST] Runtime Exception Intercepted: {global_error}", flush=True)
            time.sleep(60)