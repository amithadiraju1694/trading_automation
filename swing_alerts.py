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
    """Fetches SPY status exactly once per calendar day."""
    global SPY_CACHE
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    if SPY_CACHE["last_fetched_date"] == today_str:
        return SPY_CACHE["is_bullish"], SPY_CACHE["is_bearish"]

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Refreshing global SPY Market Regime...")
    try:
        spy_data = yf.download(tickers="SPY", period="3mo", interval="1d", progress=False)
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
def evaluate_trade_confidence(
    swept, reclaimed, ms_flip, not_overextended, 
    setup_present, volume_confirmed, spy_aligned, 
    wick_rejection, strong_body, 
    confirmation_type="weight", confidence_threshold=85
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

    return score >= confidence_threshold


# TO CONFIRM LONG AND SHORT ENTRIES BASED ON SCORE OR CRUDE ANDs for Breakout Trades
def evaluate_breakout_confidence(
    started_correct_side, closed_past_level, not_overextended,
    volume_surge, strong_close, originated_nearby, spy_aligned,
    confirmation_type="weight", confidence_threshold=85
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
def check_keylevel_entry_confirmation(TICKER, SUPPORT_LEVEL, RESISTANCE_LEVEL, VOL_LENGTH, FROM_EMAIL, TO_EMAIL):
    global last_alerted_keylevel_candles
    CONFIDENCE_THRESHOLD = 55

    # --- STEP 1: MARKET CONTEXT & DATA ACQUISITION ---
    spy_is_bullish, spy_is_bearish = get_spy_regime()

    raw_data = yf.download(tickers=TICKER, period="1mo", interval="30m", progress=False)
    if raw_data.empty or len(raw_data) < 100:
        return

    raw_data.index = raw_data.index.tz_convert("US/Eastern")
    agg_rules = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    df_2h = raw_data.resample("2h", origin="09:30:00").agg(agg_rules).dropna().copy()

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
    # FIXED: Window shifted to -6:-3 to exclude index -3 from drying up validation
    approach_vol_avg = float(df_2h["Volume"].iloc[-6:-3].mean())
    vol_drying_up = approach_vol_avg < v2h_avg
    volume_confirmed = (v2h > v2h_avg) and (v2h_prev > v2h_avg)

    # --- STEP 4: LONG SIDE CONFIRMATION PIPELINE ---
    bull_divergence = (l2h < last_structural_low) and (current_rsi > float(latest_closed_bar["RSI_at_Low"]))
    bullish_setup_present = bull_divergence or vol_drying_up

    price_long_confirmed = evaluate_trade_confidence(
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

    price_short_confirmed = evaluate_trade_confidence(
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
        confidence_threshold=CONFIDENCE_THRESHOLD  # FIXED: Now properly passes threshold value instead of None
    )

    # --- STEP 6: EXECUTE ALERTS ---
    if price_long_confirmed and (current_candle_time != last_alerted_keylevel_candles[TICKER]):
        last_alerted_keylevel_candles[TICKER] = current_candle_time
        send_email(
            subject = "*** SWING ALERT: A+ BULLISH KEY LEVEL CONFIRMED FOR : {TICKER} ***",
            body=f"Use these ballpark entries , stop losses and take profits. Entry: {c2h}, Stop Loss (Place Below This): {l2h} and Take Profit: {RESISTANCE_LEVEL}",
            from_email = FROM_EMAIL,
            to_email= TO_EMAIL,
            attachment=None
        )

    elif price_short_confirmed and (current_candle_time != last_alerted_keylevel_candles[TICKER]):
        last_alerted_keylevel_candles[TICKER] = current_candle_time
        send_email(
            subject = "*** SWING ALERT: A+ BEARISH KEY LEVEL CONFIRMED FOR : {TICKER} ***",
            body=f"Use these ballpark entries , stop losses and take profits. Entry: {c2h}, Stop Loss (Place Above This): {h2h} and Take Profit: {SUPPORT_LEVEL}",
            from_email = FROM_EMAIL,
            to_email= TO_EMAIL,
            attachment=None
        )



# Breakout based entry confirmation
def check_breakout_entry_confirmation(TICKER, SUPPORT_LEVEL, RESISTANCE_LEVEL, VOL_LENGTH, FROM_EMAIL, TO_EMAIL):
    global last_alerted_breakout_candles
    CONFIDENCE_THRESHOLD = 70

    # 1. Fetch cached macro trend gate
    spy_is_bullish, spy_is_bearish = get_spy_regime()

    # 2. Pull Ticker Data
    raw_data = yf.download(tickers=TICKER, period="1mo", interval="30m", progress=False)
    if raw_data.empty or len(raw_data) < 100:
        return

    raw_data.index = raw_data.index.tz_convert("US/Eastern")
    agg_rules = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    df_2h = raw_data.resample("2h", origin="09:30:00").agg(agg_rules).dropna().copy()

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

    # Bar Coordinate Unpacking
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

    # Base breakout metrics
    volume_surge = v2h > (v2h_avg * 1.2)

    # --- 3. LONG SIDE (BULLISH BREAKOUT) PIPELINE ---
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

    # --- 4. SHORT SIDE (BEARISH BREAKDOWN) PIPELINE ---
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

    # --- 5. EXECUTING ALERTS ---
    if price_long_breakout and (current_candle_time != last_alerted_breakout_candles[TICKER]):
        last_alerted_breakout_candles[TICKER] = current_candle_time
        send_email(
            subject = "*** SWING ALERT: A+ BULLISH BREAKOUT CONFIRMED FOR : {TICKER} ***",
            #TP Is Textbook formula for "measure move", a conservative , probably first level TP
            body=f"Use these ballpark entries , stop losses and take profits. Entry: {c2h}, Stop Loss (Place Below This): {RESISTANCE_LEVEL} and Take Profit: {c2h + (RESISTANCE_LEVEL - SUPPORT_LEVEL)}",
            from_email = FROM_EMAIL,
            to_email= TO_EMAIL,
            attachment=None
        )
        

    elif price_short_breakout and (current_candle_time != last_alerted_breakout_candles[TICKER]):
        last_alerted_breakout_candles[TICKER] = current_candle_time
        send_email(
            subject = "*** SWING ALERT: A+ BEARISH KEY LEVEL CONFIRMED FOR : {TICKER} ***",
            # Similar texrtbook formula for downward move
            body=f"Use these ballpark entries , stop losses and take profits. Entry: {c2h}, Stop Loss (Place Above This): {SUPPORT_LEVEL} and Take Profit: {c2h - (RESISTANCE_LEVEL - SUPPORT_LEVEL)}",
            from_email = FROM_EMAIL,
            to_email= TO_EMAIL,
            attachment=None
        )
        





if __name__ == "__main__":
    eastern = pytz.timezone("US/Eastern")
    print("Initializing Multi-Ticker Automated Reversal Script Engine...")

    while True:
        try:
            now = datetime.datetime.now(eastern)

            # Check if current day is a weekday
            if now.weekday() < 5:
                market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

                if market_open <= now <= market_close:
                    
                    # Read Watchlist dynamically every loop
                    if os.path.exists("watchlist.csv"):

                        #TODO: Filter by column trade type in watchlist.csv
                        #TODO: if trade_type is "breakout" call "check_breakout_entry_confirmation" elif trade_type is "keylevel" call "check_keylevel_entry_confirmation"
                        watchlist = pd.read_csv("watchlist.csv")
                        
                        for index, row in watchlist.iterrows():
                            ticker = str(row['Ticker']).strip()
                            support = float(row['Support'])
                            resistance = float(row['Resistance'])
                            vol_len = int(row['Vol_Length'])
                            
                            #TODO: Change function call here based on trade typw
                            check_entry_confirmation(ticker, support, resistance, vol_len)
                            
                            # Pause briefly between tickers to prevent yfinance rate limiting
                            time.sleep(2) 
                    else:
                        print("watchlist.csv not found. Please create it.")
                    
                    print(f"Cycle complete. Sleeping for {CHECK_INTERVAL_SECONDS} seconds...")
                    time.sleep(CHECK_INTERVAL_SECONDS)
                else:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Market closed. Sleeping 60s...")
                    time.sleep(60)
            else:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Weekend Mode Active. Sleeping 300s...")
                time.sleep(300)

        except Exception as global_error:
            print(f"Runtime Exception Intercepted: {global_error}")
            time.sleep(60)

