import os
import yfinance as yf
import pandas as pd
import schedule
import time
from datetime import datetime
import pytz
from tqdm import tqdm
import sys
import numpy as np
from helpers import (
    send_email,
    compute_adx,
    compute_rsi,
    compute_squeeze,
    get_all_tickers,
    compute_ema_position,
    filter_by_mkt_cap,
    filter_by_perc_move
)

# ---------------- CONFIG ---------------- #
FROM_EMAIL = "amitadiraju3@gmail.com"
TO_EMAIL = "amith.adiraju@gmail.com"

CSV_NAME = "squeeze_scan_results.csv"
FILTERED_MKT_CAP_CSV = "filtered_by_market_cap.csv"
DEFAULT_PERCENT_MOVE = 1.8


def get_sr_and_atr_distances(df, min_touches=5, tolerance_pct=0.015, atr_length=14, min_bars_between_touches=15):
    """
    Identifies Key Support/Resistance zones based on minimum historical touches.
    Forces strict chronological separation between touches to ensure they are 
    true historical tests, not just multi-day consolidations.
    """
    if len(df) < (atr_length + 10):
        return {"ATR_DIST_FROM_SUPPORT": "N/A", "ATR_DIST_FROM_RESISTANCE": "N/A"}

    df = df.copy()
    
    # Assign an absolute integer index to track time distance between touches
    df['Bar_Num'] = range(len(df))

    # 1. Calculate ATR
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=atr_length).mean()

    # 2. Find True Swing Highs and Lows
    # Using window=11 center=True means 5 days before, the day itself, and 5 days after
    df['Max_11'] = df['High'].rolling(window=11, center=True).max()
    df['Min_11'] = df['Low'].rolling(window=11, center=True).min()

    df['Is_Peak'] = df['High'] == df['Max_11']
    df['Is_Trough'] = df['Low'] == df['Min_11']

    # Extract the prices and their specific bar numbers
    peaks = df[df['Is_Peak']][['High', 'Bar_Num']].dropna()
    troughs = df[df['Is_Trough']][['Low', 'Bar_Num']].dropna()

    # 3. Clustering & Chronological Validation
    def find_valid_zones(pivots_df, price_col, tol, min_req_touches, min_separation):
        if pivots_df.empty:
            return []
        
        # Sort entirely by price to group similar levels together
        sorted_pivots = pivots_df.sort_values(by=price_col)
        
        prices = sorted_pivots[price_col].values
        bar_nums = sorted_pivots['Bar_Num'].values
        
        clusters = []
        current_cluster_prices = [prices[0]]
        current_cluster_bars = [bar_nums[0]]

        # Group prices within the tolerance %
        for i in range(1, len(prices)):
            cluster_avg = np.mean(current_cluster_prices)
            
            if abs(prices[i] - cluster_avg) / cluster_avg <= tol:
                current_cluster_prices.append(prices[i])
                current_cluster_bars.append(bar_nums[i])
            else:
                clusters.append((current_cluster_prices, current_cluster_bars))
                current_cluster_prices = [prices[i]]
                current_cluster_bars = [bar_nums[i]]
        clusters.append((current_cluster_prices, current_cluster_bars))

        valid_zones = []

        # Validate each cluster chronologically
        for cluster_prices, cluster_bars in clusters:
            # Sort the bar numbers chronologically to check distance between touches
            sorted_time_bars = sorted(cluster_bars)
            
            valid_touches = 1
            last_touch_bar = sorted_time_bars[0]
            
            for current_bar in sorted_time_bars[1:]:
                # Only count the touch if it happened 'min_separation' days after the last one
                if (current_bar - last_touch_bar) >= min_separation:
                    valid_touches += 1
                    last_touch_bar = current_bar
                    
            if valid_touches >= min_req_touches:
                valid_zones.append(np.mean(cluster_prices))
                
        return valid_zones

    # Extract valid zones using 5 touches and 15 days of separation
    valid_resistances = find_valid_zones(peaks, 'High', tolerance_pct, min_touches, min_bars_between_touches)
    valid_supports = find_valid_zones(troughs, 'Low', tolerance_pct, min_touches, min_bars_between_touches)

    # 4. Find Nearest Zones & Calculate ATR Distance
    current_price = df['Close'].iloc[-1]
    current_atr = df['ATR'].iloc[-1]

    if pd.isna(current_atr) or current_atr == 0:
        return {"ATR_DIST_FROM_SUPPORT": "N/A", "ATR_DIST_FROM_RESISTANCE": "N/A"}

    # Nearest Support (Must be BELOW current price)
    supports_below = [s for s in valid_supports if s < current_price]
    nearest_support = max(supports_below) if supports_below else None

    # Nearest Resistance (Must be ABOVE current price)
    resistances_above = [r for r in valid_resistances if r > current_price]
    nearest_resistance = min(resistances_above) if resistances_above else None

    # 5. Math: (Distance to Level) / ATR
    dist_supp = round((current_price - nearest_support) / current_atr, 2) if nearest_support else "NO KEY SUPP"
    dist_res = round((nearest_resistance - current_price) / current_atr, 2) if nearest_resistance else "NO KEY RES"

    return {
        "ATR_DIST_FROM_SUPPORT": dist_supp,
        "ATR_DIST_FROM_RESISTANCE": dist_res,
        "NEAREST_SUPPORT": nearest_support,
        "NEAREST_RESISTANCE": nearest_resistance
    }


def generate_adirindic_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the 1-year OHLCV DataFrame from run_scan and applies the ADIRINDIC logic.
    Computes its own internal single-day squeeze status purely for score weighting.
    Returns the DataFrame with added BUY_SIGNAL and SELL_SIGNAL columns.
    """
    # Copy to avoid modifying the original dataframe used by other functions
    df = df.copy()
    
    # --- Helper: True Range & ATR ---
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    df['ATR_14'] = df['TR'].ewm(alpha=1/14, adjust=False).mean()
    df['ATR_20'] = df['TR'].ewm(alpha=1/20, adjust=False).mean()

    # --- Indicator Setup ---
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['SMA_50']  = df['Close'].rolling(window=50).mean()
    df['SMA_20']  = df['Close'].rolling(window=20).mean()
    df['EMA_21']  = df['Close'].ewm(span=21, adjust=False).mean()
    df['VWMA_14'] = (df['Close'] * df['Volume']).rolling(14).sum() / df['Volume'].rolling(14).sum()
    df['STD_20']  = df['Close'].rolling(window=20).std(ddof=0)

    # --- Internal Single-Day Squeeze Logic for ADIRINDIC ---
    df['BB_upper'] = df['SMA_20'] + (2.0 * df['STD_20'])
    df['BB_lower'] = df['SMA_20'] - (2.0 * df['STD_20'])
    df['KC_mid']   = df['SMA_20']
    df['KC_upper'] = df['KC_mid'] + (1.5 * df['ATR_20'])
    df['KC_lower'] = df['KC_mid'] - (1.5 * df['ATR_20'])

    # Single day squeeze check
    df['Squeeze_on'] = (df['BB_upper'] < df['KC_upper']) & (df['BB_lower'] > df['KC_lower'])

    squeeze_active = df['Squeeze_on'] | df['Squeeze_on'].shift(1) | df['Squeeze_on'].shift(2)
    df['Bull_Squeeze'] = squeeze_active & (df['Close'] > df['KC_mid'])
    df['Bear_Squeeze'] = squeeze_active & (df['Close'] < df['KC_mid'])

    # --- Candlestick Scanners ---
    df['Body_Size']  = (df['Close'] - df['Open']).abs()
    df['Range_Size'] = df['High'] - df['Low']
    df['Is_Doji']    = df['Body_Size'] <= (df['Range_Size'] * 0.1)

    df['Bull_Doji'] = df['Is_Doji'] & (df['Close'] >= df['Close'].shift(1))
    df['Bear_Doji'] = df['Is_Doji'] & (df['Close'] <= df['Close'].shift(1))

    df['Bull_Eng'] = (df['Close'].shift(1) < df['Open'].shift(1)) & \
                     (df['Close'] > df['Open']) & \
                     (df['Close'] >= df['Open'].shift(1)) & \
                     (df['Open'] <= df['Close'].shift(1))
                     
    df['Bear_Eng'] = (df['Close'].shift(1) > df['Open'].shift(1)) & \
                     (df['Close'] < df['Open']) & \
                     (df['Close'] <= df['Open'].shift(1)) & \
                     (df['Open'] >= df['Close'].shift(1))

    df['Bull_Candle'] = df['Bull_Eng'] | df['Bull_Doji']
    df['Bear_Candle'] = df['Bear_Eng'] | df['Bear_Doji']

    # --- Order Flow (Supply & Demand) ---
    # Ensure your pivot lookback window helper function has enough padding
    def get_pivot(x, is_high=True):
        if len(x) < 11: return np.nan
        # x[5] is the center bar of an 11-bar array (5 left, center, 5 right)
        center = x[5] 
        if is_high and center == max(x): return center
        if not is_high and center == min(x): return center
        return np.nan

    df['Pivot_High'] = df['High'].rolling(window=11).apply(lambda x: get_pivot(x, True), raw=True)
    df['Pivot_Low']  = df['Low'].rolling(window=11).apply(lambda x: get_pivot(x, False), raw=True)
    df['Supply_Zone'] = df['Pivot_High'].ffill()
    df['Demand_Zone'] = df['Pivot_Low'].ffill()

    df['Bull_SD'] = (df['Close'] <= df['Demand_Zone'] + (df['ATR_14'] * 1.5)) | (df['Close'] > df['Supply_Zone'])
    df['Bear_SD'] = (df['Close'] >= df['Supply_Zone'] - (df['ATR_14'] * 1.5)) | (df['Close'] < df['Demand_Zone'])

    # --- Score Generation ---
    w_trend, w_mom, w_vol, w_kc, w_sd, w_candle, threshold = 20, 15, 15, 15, 25, 10, 51

    df['Bull_Score'] = (
        np.where(df['Close'] > df['SMA_50'], w_trend, 0) +
        np.where(df['Close'] > df['EMA_21'], w_mom, 0) +
        np.where(df['Close'] > df['VWMA_14'], w_vol, 0) +
        np.where(df['Bull_Squeeze'], w_kc, 0) +
        np.where(df['Bull_SD'], w_sd, 0) +
        np.where(df['Bull_Candle'], w_candle, 0)
    )

    df['Bear_Score'] = (
        np.where(df['Close'] < df['SMA_50'], w_trend, 0) +
        np.where(df['Close'] < df['EMA_21'], w_mom, 0) +
        np.where(df['Close'] < df['VWMA_14'], w_vol, 0) +
        np.where(df['Bear_Squeeze'], w_kc, 0) +
        np.where(df['Bear_SD'], w_sd, 0) +
        np.where(df['Bear_Candle'], w_candle, 0)
    )

    df['Long_Condition']  = df['Bull_Score'] >= threshold
    df['Short_Condition'] = df['Bear_Score'] >= threshold

    df['BUY_SIGNAL']  = df['Long_Condition'] & ~df['Long_Condition'].shift(1).fillna(False)
    df['SELL_SIGNAL'] = df['Short_Condition'] & ~df['Short_Condition'].shift(1).fillna(False)

    
    return df
    

def generate_kelfry_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a historical OHLCV DataFrame and applies Kelfry98's 'Buy Sell Signal' logic.

    Maintains active position states bar-by-bar to calculate entries, hard
    trailing stops, and risk-to-reward target exits exactly like Pine Script.

    Returns the original DataFrame with added boolean columns 'BUY_SIGNAL' and
    'SELL_SIGNAL'.
    """
    # Copy to avoid modifying the original dataframe passed from run_scan
    df = df.copy()

    # --- 1. CONFIGURABLE PARAMETERS ---
    ema_fast_len = 5
    ema_slow_len = 13
    atr_len = 14
    atr_mult_sl = 0.5
    risk_reward = 3.0
    confirm_candle = True

    # --- 2. INDICATOR ENGINE CALCULATIONS ---
    # Fast & Slow EMAs
    df["ema_fast"] = df["Close"].ewm(span=ema_fast_len, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=ema_slow_len, adjust=False).mean()

    # True Range & ATR
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].ewm(alpha=1 / atr_len, adjust=False).mean()

    # --- 3. TREND CHANGE DETECTION ---
    df["bull_trend"] = df["ema_fast"] > df["ema_slow"]
    df["bear_trend"] = df["ema_fast"] < df["ema_slow"]
    df["trend_change"] = df["bull_trend"] != df["bull_trend"].shift(1)

    # --- 4. CONDITIONAL ENTRY STRATEGIES ---
    if confirm_candle:
        df["raw_buy"] = (
            df["bull_trend"] & df["trend_change"] & (df["Close"] > df["Open"])
        )
        df["raw_sell"] = (
            df["bear_trend"] & df["trend_change"] & (df["Close"] < df["Open"])
        )
    else:
        df["raw_buy"] = df["bull_trend"] & df["trend_change"]
        df["raw_sell"] = df["bear_trend"] & df["trend_change"]

    # Fill NaNs generated by shifts to avoid boolean loop failures
    df["raw_buy"] = df["raw_buy"].fillna(False)
    df["raw_sell"] = df["raw_sell"].fillna(False)

    # --- 5. SEQUENTIAL STATE MACHINE LOOP ---
    # Extract structural numpy arrays for performance optimization
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    atrs = df["ATR"].values
    raw_buys = df["raw_buy"].values
    raw_sells = df["raw_sell"].values

    buy_signals = []
    sell_signals = []

    # State variables tracking the active position
    in_position = False
    position_type = ""  # 'LONG' or 'SHORT'
    stop_loss = 0.0
    take_profit_final = 0.0

    for i in range(len(df)):
        show_buy = False
        show_sell = False

        # Phase A: Check exit thresholds on the active position
        if in_position:
            if position_type == "LONG":
                if highs[i] >= take_profit_final or lows[i] <= stop_loss:
                    in_position = False
                    position_type = ""
            elif position_type == "SHORT":
                if lows[i] <= take_profit_final or highs[i] >= stop_loss:
                    in_position = False
                    position_type = ""

        # Phase B: Evaluate new setups and invalidation entries
        if raw_buys[i]:
            # Trigger setup only if no active LONG position exists
            if not in_position or position_type != "LONG":
                # If currently short, this triggers an internal invalidation flip
                in_position = True
                position_type = "LONG"

                entry_price = closes[i]
                stop_loss = lows[i] - (atrs[i] * atr_mult_sl)
                risk = entry_price - stop_loss
                take_profit_final = entry_price + (risk * risk_reward)

                show_buy = True

        elif raw_sells[i]:
            # Trigger setup only if no active SHORT position exists
            if not in_position or position_type != "SHORT":
                # If currently long, this triggers an internal invalidation flip
                in_position = True
                position_type = "SHORT"

                entry_price = closes[i]
                stop_loss = highs[i] + (atrs[i] * atr_mult_sl)
                risk = stop_loss - entry_price
                take_profit_final = entry_price - (risk * risk_reward)

                show_sell = True

        buy_signals.append(show_buy)
        sell_signals.append(show_sell)

    # --- 6. ATTACH SIGNAL ARRAYS TO DATAFRAME ---
    df["BUY_SIGNAL"] = buy_signals
    df["SELL_SIGNAL"] = sell_signals

    return df


def run_scan(filtered_tickers=None, percent_move=DEFAULT_PERCENT_MOVE):
    if filtered_tickers is None:
        if os.path.exists(FILTERED_MKT_CAP_CSV):
            filtered_tickers = filter_by_mkt_cap(FILTERED_MKT_CAP_CSV)
        else:
            filtered_tickers = get_all_tickers()
            pd.DataFrame({"TICKER": filtered_tickers}).to_csv(FILTERED_MKT_CAP_CSV, index=False)

    filtered_tickers = filter_by_perc_move(filtered_tickers, percent_move)

    if len(filtered_tickers) < 1:
        print("No tickers available after filtering. Nothing to scan.")
        return

    print(f"\nScanning {len(filtered_tickers)} filtered tickers for Squeezes, ADIRINDIC, & KELFRY98 Signals...")
    results = []

    for ticker in tqdm(filtered_tickers, desc="Evaluating Trading Logic"):
        try:
            # 1. Fetch data
            stock = yf.Ticker(ticker)
            df = stock.history(period="max", auto_adjust=False, actions=False)

            if df is None or df.empty or len(df) < 250:
                continue
            
            
            # DO NOT CONSIDER STOCKS which don't have enough volume
            df["Vol_SMA20"] = df["Volume"].rolling(window = 20).mean()
            if df['Vol_SMA20'].iloc[-1] < 750000:
                continue

            # ---> 1. NEW: CALCULATE RSI & ADX <---
            df['RSI_14'] = compute_rsi(df, period=14)
            df['ADX_14'] = compute_adx(df, period=14)
            
            # Extract the current day's value and round it cleanly
            current_rsi = round(df['RSI_14'].iloc[-1], 2)
            current_adx = round(df['ADX_14'].iloc[-1], 2)


            # 2. Compute 6-day Squeeze metrics
            squeeze_series = compute_squeeze(df)
            
            if len(squeeze_series) < 7:
                continue

            # LOGIC 1: Is it currently in a 6-day squeeze?
            # .iloc[-6:] precisely evaluates the last 6 trading days (including today)
            in_6_day_squeeze = "YES" if bool(squeeze_series.iloc[-6:].all()) else "NO"

            # LOGIC 2: Did it squeeze for 6 days prior to today, and FIRE today?
            # .iloc[-7:-1] precisely evaluates the 6 days prior to today
            # .iloc[-1] specifically targets today's candle to confirm it fired
            six_days_prior_squeeze = bool(squeeze_series.iloc[-7:-1].all())
            fired_today = not bool(squeeze_series.iloc[-1])
            
            squeeze_fired = "YES" if (six_days_prior_squeeze and fired_today) else "NO"

            # 3. Compute Strategy Signals
            adir_df = generate_adirindic_signal(df)
            kelfry_df = generate_kelfry_signal(df)    
            ema_position, ema_comparison = compute_ema_position(df)

            # 4. Computing Distance from Key levels for this ticker
            # Using 1.5% tolerance (0.015) for zone clustering. Only sending 3 years to counter stock split s&r levels
            sr_metrics = get_sr_and_atr_distances(
                df.tail(750), 
                min_touches=3, 
                tolerance_pct=0.015, 
                atr_length=14
            )
            
            # Extract ADIRINDIC signals from Day 't' down to 't-7'
            adir_lookback = adir_df.tail(8)
            adir_signals = []
            for timestamp, row in adir_lookback.iterrows():
                date_str = timestamp.strftime('%Y-%m-%d')
                if row.get('BUY_SIGNAL'):
                    adir_signals.append(f"BUY @ ${row['Low']:.2f} ({date_str})")
                if row.get('SELL_SIGNAL'):
                    adir_signals.append(f"SELL @ ${row['High']:.2f} ({date_str})")

            # Extract KELFRY98 signals from Day 't' down to 't-7'
            kelfry_lookback = kelfry_df.tail(8)
            kelfry_signals = []
            for timestamp, row in kelfry_lookback.iterrows():
                date_str = timestamp.strftime('%Y-%m-%d')
                if row.get('BUY_SIGNAL'):
                    kelfry_signals.append(f"BUY @ ${row['Low']:.2f} ({date_str})")
                if row.get('SELL_SIGNAL'):
                    kelfry_signals.append(f"SELL @ ${row['High']:.2f} ({date_str})")




            # Define your indicator states
            has_any_signal = len(adir_signals) > 0 or len(kelfry_signals) > 0
            is_squeezing = in_6_day_squeeze == "YES" or squeeze_fired == "YES"
            
            # Extract calculated metrics
            dist_to_supp = sr_metrics["ATR_DIST_FROM_SUPPORT"]
            dist_to_res = sr_metrics["ATR_DIST_FROM_RESISTANCE"]
            
            # Setup our strict conditional gates
            is_valid_breakout = False; is_valid_bounce = False

            
            # PRONG A: THE BREAKOUT PROFILE. May need to remove rsi in future
            if is_squeezing or has_any_signal:
                # Long Breakout Criteria
                if (ema_position == "ABOVE_EMAS" or ema_position == "BETWEEN") and current_adx > 22:
                    is_valid_breakout = True
                # Short Breakout Criteria
                elif ema_position == "BELOW_EMAS" and current_adx > 22:
                    is_valid_breakout = True

            # PRONG B: THE KEY-LEVEL BOUNCE PROFILE
            # Long Bounce near Support. May need to remove rsi in future
            if has_any_signal and isinstance(dist_to_supp, (int, float)) and dist_to_supp <= 0.75:
                if current_adx < 20 or current_rsi < 40: # Ranging market or deeply oversold
                    is_valid_bounce = True
                    
            # Short Bounce near Resistance
            elif has_any_signal and isinstance(dist_to_res, (int, float)) and dist_to_res <= 0.75:
                if current_adx < 20 or current_rsi > 60: # Ranging market or overbought
                    is_valid_bounce = True


            
            # 4. Actionable Filter: Include if ANY condition is met
            if is_valid_breakout or is_valid_bounce:
                setup_type = "BREAKOUT" if is_valid_breakout else "BOUNCE"
                adir_value = " | ".join(adir_signals) if adir_signals else "NONE"
                kelfry_value = " | ".join(kelfry_signals) if kelfry_signals else "NONE"
                stock_price = df['Close'].iloc[-1]
                
                results.append({
                    "STOCK TICKER": ticker,
                    "SETUP TYPE": setup_type,
                    "STOCK PRICE": round(stock_price, 2),
                    "IN 6 DAY SQUEEZE": in_6_day_squeeze,
                    "SQUEEZE_FIRED": squeeze_fired,
                    "PRICE RELATIVE TO EMA": ema_position,
                    "EMA_COMPARISON": ema_comparison,
                    "ADIRINDIC": adir_value,
                    "KELFRY98": kelfry_value,
                    "ATR DIST TO SUPPORT": sr_metrics["ATR_DIST_FROM_SUPPORT"],
                    "ATR DIST TO RESISTANCE": sr_metrics["ATR_DIST_FROM_RESISTANCE"],
                    "NEAREST_SUPPORT": sr_metrics['NEAREST_SUPPORT'],
                    "NEAREST_RESISTANCE": sr_metrics['NEAREST_RESISTANCE'],
                    "RSI (14)": current_rsi,
                    "ADX (14)": current_adx
                })


        except Exception:
            continue

    if len(results) == 0:
        print("\nNo stocks met the criteria today.")
        return

    print(f"\nFound {len(results)} actionable setups! Sending email now...")
    result_df = pd.DataFrame(results)
    result_df.to_csv(CSV_NAME, index=True)
    
    send_email(
        subject="Daily Squeeze & Indicator Alerts",
        body="Please find attached today's tracking results.",
        from_email=FROM_EMAIL,
        to_email=TO_EMAIL,
        attachment=CSV_NAME
    )
    print("\n--- RESULTS PREVIEW ---")
    print(result_df.head())

# ---------------- SCHEDULER ---------------- #
def job(run_type="SCHEDULED"):
    """
    Executes the trading logic. 
    Accepts run_type to differentiate between test runs and scheduled runs in the logs.
    """
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)
    
    print(f"\n--- [{run_type} RUN] Executing scan at: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    
    # Actually calling your target scanning and trading logic
    run_scan() 
    
    print(f"--- [{run_type} RUN] Scan complete. ---")


def main():
    # 1. Handle Immediate Test Run if flag is passed
    if "--test" in sys.argv:
        print("Test flag detected. Executing an immediate test run before scheduling...")
        job(run_type="TEST")
        print("Test run finished")
    else:
        # 2. Setup the Daily Schedule
        runat = "17:00"
        
        # Pass the run_type argument directly into the .do() method
        schedule.every().day.at(runat, "US/Eastern").do(job, run_type="SCHEDULED")
        
        print(f"Scanner initialized. Waiting quietly in background to run daily at {runat} US/Eastern.")

        # 3. Keep the script alive and checking the clock
        while True:
            schedule.run_pending()
            time.sleep(30)


if __name__ == "__main__":
    main()
