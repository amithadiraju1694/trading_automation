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
from helpers import send_email

# ---------------- CONFIG ---------------- #
FROM_EMAIL = "amitadiraju3@gmail.com"
TO_EMAIL = "amith.adiraju@gmail.com"

CSV_NAME = "squeeze_scan_results.csv"
FILTERED_MKT_CAP_CSV = "filtered_by_market_cap.csv"
DEFAULT_PERCENT_MOVE = 3.0

# ---------------- GET ALL STOCKS ---------------- #
def get_all_tickers():
    nasdaq = pd.read_csv(
        "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"
    )
    tickers = nasdaq["Symbol"].dropna().tolist()

    # Remove weird tickers and warrants
    tickers = [t for t in tickers if "-" not in t and "^" not in t and "." not in t]
    return tickers


# ---------------- INDICATORS ---------------- #
def compute_squeeze(df):
    length = 20

    # Bollinger Bands
    sma = df['Close'].rolling(length).mean()
    std = df['Close'].rolling(length).std()
    bb_upper = sma + (2 * std)
    bb_lower = sma - (2 * std)

    # Keltner Channels
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(length).mean()
    kc_upper = sma + (1.5 * atr)
    kc_lower = sma - (1.5 * atr)

    # Squeeze is ON when Bollinger Bands are completely inside Keltner Channels
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    return squeeze_on


def compute_ema_position(df):
    emas = [8, 21, 34, 55, 89]
    ema_values = []

    for e in emas:
        ema = df['Close'].ewm(span=e, adjust=False).mean()
        ema_values.append(ema.iloc[-1])

    close = df['Close'].iloc[-1]

    if close > max(ema_values):
        return "ABOVE"
    elif close < min(ema_values):
        return "BELOW"
    else:
        return "BETWEEN"


def filter_by_mkt_cap(tickers_or_csv):
    if isinstance(tickers_or_csv, str) and tickers_or_csv.lower().endswith(".csv"):
        df = pd.read_csv(tickers_or_csv)
        if df.empty:
            return []
        first_col = df.columns[0]
        return df[first_col].dropna().astype(str).tolist()

    filtered = []
    for ticker in tqdm(tickers_or_csv, desc="Filtering Market Cap"):
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info # faster than stock.info
            market_cap = info.get("market_cap", 0)

            if market_cap >= 100_000_000:
                filtered.append(ticker)
        except Exception:
            # Silently catch delisted tickers
            continue

    pd.DataFrame({"TICKER": filtered}).to_csv(FILTERED_MKT_CAP_CSV, index=False)
    return filtered


def filter_by_perc_move(tickers, percent_move):
    filtered = []
    for ticker in tqdm(tickers, desc="Filtering stocks for % move"):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1mo") # Changed to 1mo for speed

            if df is None or df.empty or len(df) < 2:
                continue

            last_close = df['Close'].iloc[-1]
            if last_close <= 7:
                continue

            daily_move = (abs(last_close - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100

            if daily_move >= percent_move:
                filtered.append(ticker)
        except Exception:
            continue

    return filtered


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
    



def run_scan(filtered_tickers=None, percent_move=DEFAULT_PERCENT_MOVE):
    if filtered_tickers is None:
        if os.path.exists(FILTERED_MKT_CAP_CSV):
            filtered_tickers = filter_by_mkt_cap(FILTERED_MKT_CAP_CSV)
        else:
            filtered_tickers = get_all_tickers()

    filtered_tickers = filter_by_perc_move(filtered_tickers, percent_move)

    if len(filtered_tickers) < 1:
        print("No tickers available after filtering. Nothing to scan.")
        return

    print(f"\nScanning {len(filtered_tickers)} filtered tickers for Squeezes & ADIRINDIC Signals...")
    results = []

    for ticker in tqdm(filtered_tickers, desc="Evaluating Trading Logic"):
        try:
            # 1. Fetch 1y data to ensure the 200 SMA padding works in the indicator
            stock = yf.Ticker(ticker)
            df = stock.history(period="max", auto_adjust = False, actions = False)

            if df is None or df.empty or len(df) < 250:
                continue

            # 2. Compute 6-day Squeeze using original external function
            squeeze_series = compute_squeeze(df)
            
            if len(squeeze_series) < 7:
                continue

            # LOGIC 1: Is it currently in a 6-day squeeze? (The last 6 days up to today are True)
            in_6_day_squeeze = "YES" if squeeze_series.iloc[-6:].all() else "NO"

            # LOGIC 2: Did it squeeze for 6 days prior to today, and FIRE today?
            # Days -7 to -2 (the 6 days before today) were True, and Day -1 (Today) is False
            six_days_prior_squeeze = squeeze_series.iloc[-7:-1].all()
            fired_on_7th = not squeeze_series.iloc[-1]
            squeeze_fired = "YES" if (six_days_prior_squeeze and fired_on_7th) else "NO"

            # 3. Compute ADIRINDIC signals passing the fetched df
            adir_df = generate_adirindic_signal(df)
            
            # Extract any signals from Day 't' down to 't-7'
            lookback_window = adir_df.tail(8)
            signals_found = []
            
            for timestamp, row in lookback_window.iterrows():
                date_str = timestamp.strftime('%Y-%m-%d')
                if row.get('BUY_SIGNAL'):
                    signals_found.append(f"BUY @ ${row['Low']:.2f} ({date_str})")
                if row.get('SELL_SIGNAL'):
                    signals_found.append(f"SELL @ ${row['High']:.2f} ({date_str})")

            # 4. Actionable Filter: ONLY append if it meets the main criteria
            if in_6_day_squeeze == "YES" or squeeze_fired == "YES" or len(signals_found) > 0:
                
                # Rule: Only compute EMA if it's currently in a 6-day squeeze
                ema_position = "N/A"
                if in_6_day_squeeze == "YES":
                    ema_position = compute_ema_position(df)
                
                adirindic_value = " | ".join(signals_found) if signals_found else "NONE"
                stock_price = df['Close'].iloc[-1]
                
                results.append({
                    "STOCK TICKER": ticker,
                    "STOCK PRICE": round(stock_price, 2),
                    "IN 6 DAY SQUEEZE": in_6_day_squeeze,
                    "SQUEEZE_FIRED": squeeze_fired,
                    "PRICE RELATIVE TO EMA": ema_position,
                    "ADIRINDIC": adirindic_value
                })

        except Exception:
            continue

    if len(results) == 0:
        print("\nNo stocks met the criteria today.")
        return

    print(f"\nFound {len(results)} actionable setups! Sending email now...")
    result_df = pd.DataFrame(results)
    result_df.to_csv(CSV_NAME, index=False)
    
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