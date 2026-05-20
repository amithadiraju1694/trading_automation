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
'''

# Track timestamps per ticker: {'AAPL': '2023-10-25 14:00:00', 'MDGL': '...'}
last_alerted_candles = {} 
CHECK_INTERVAL_SECONDS = 120

def check_entry_confirmation(TICKER, SUPPORT_LEVEL, RESISTANCE_LEVEL, VOL_LENGTH):
    global last_alerted_candles

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Evaluating {TICKER}...")

    # Fetch 30-minute intervals
    raw_data = yf.download(
        tickers=TICKER, period="1mo", interval="30m", progress=False
    )

    if raw_data.empty or len(raw_data) < 100:
        print(f"[{TICKER}] Insufficient historical data.")
        return

    raw_data.index = raw_data.index.tz_convert("US/Eastern")

    # Resample to 2H candles. This can change to two 1H candles if needed
    agg_rules = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    df_2h = raw_data.resample("2h", origin="09:30:00").agg(agg_rules).dropna()
    df_2h["Vol_SMA"] = df_2h["Volume"].rolling(window=VOL_LENGTH).mean()

    if len(df_2h) < (VOL_LENGTH + 4):
        print(f"[{TICKER}] Dataframe processing window incomplete.")
        return

    # --- FIBONACCI REVERSAL MATHEMATICS ---
    price_range = RESISTANCE_LEVEL - SUPPORT_LEVEL
    bull_091 = SUPPORT_LEVEL + (price_range * 0.09)
    bear_091 = RESISTANCE_LEVEL - (price_range * 0.09)
    bull_max_chase = SUPPORT_LEVEL + (price_range * 0.214)
    bear_max_chase = RESISTANCE_LEVEL - (price_range * 0.214)

    # --- EXTRACT FULLY CLOSED DATA ARRAYS ---
    latest_closed_bar = df_2h.iloc[-2]
    prev_closed_bar = df_2h.iloc[-3]
    two_bars_ago = df_2h.iloc[-4]

    c2h = float(latest_closed_bar["Close"])
    h2h = float(latest_closed_bar["High"])
    l2h = float(latest_closed_bar["Low"])
    v2h = float(latest_closed_bar["Volume"])
    v2h_avg = float(latest_closed_bar["Vol_SMA"])

    c2h_prev = float(prev_closed_bar["Close"])
    v2h_prev = float(prev_closed_bar["Volume"])
    c2h_prev2 = float(two_bars_ago["Close"])

    current_candle_time = str(df_2h.index[-2])
    
    # Initialize tracking for this ticker if it doesn't exist
    if TICKER not in last_alerted_candles:
        last_alerted_candles[TICKER] = None

    # --- VALIDATION GATES ---
    long_wick_ok = (h2h - c2h) <= ((h2h - l2h) * 0.35)
    short_wick_ok = (c2h - l2h) <= ((h2h - l2h) * 0.35)
    volume_confirmed = (v2h > v2h_avg) and (v2h_prev > v2h_avg)

    price_long_confirmed = (
        (c2h_prev2 <= bull_091)
        and (c2h_prev > bull_091)
        and (c2h > bull_091)
        and (c2h <= bull_max_chase)
    )
    
    price_short_confirmed = (
        (c2h_prev2 >= bear_091)
        and (c2h_prev < bear_091)
        and (c2h < bear_091)
        and (c2h >= bear_max_chase)
    )

    # --- EVALUATE SIGNALS ---
    if (
        price_long_confirmed
        and volume_confirmed
        and long_wick_ok
        and (current_candle_time != last_alerted_candles[TICKER])
    ):
        subject = f"SWING ALERT: PRECISE LONG CONFIRMED FOR {TICKER}"
        body = (
            f"Asset {TICKER} has cleared structural validation boundaries.\n\n"
            f"Metrics Log:\n"
            f"- Support Target: {SUPPORT_LEVEL}\n"
            f"- Trigger Execution Close: {c2h}\n"
            f"- Allowed Chase Limit: {bull_max_chase}\n"
            f"- Validated Candle Timestamp: {current_candle_time}\n\n"
            f"Action: Evaluate execution parameters for standard Long entries."
        )
        send_email(subject=subject, body=body, from_email=FROM_EMAIL, to_email=TO_EMAIL, attachment=None)
        last_alerted_candles[TICKER] = current_candle_time
        print(f"*** LONG ALERT SENT FOR {TICKER} ***")

    elif (
        price_short_confirmed
        and volume_confirmed
        and short_wick_ok
        and (current_candle_time != last_alerted_candles[TICKER])
    ):
        subject = f"SWING ALERT: PRECISE SHORT CONFIRMED FOR {TICKER}"
        body = (
            f"Asset {TICKER} has cleared structural validation boundaries.\n\n"
            f"Metrics Log:\n"
            f"- Resistance Target: {RESISTANCE_LEVEL}\n"
            f"- Trigger Execution Close: {c2h}\n"
            f"- Allowed Chase Limit: {bear_max_chase}\n"
            f"- Validated Candle Timestamp: {current_candle_time}\n\n"
            f"Action: Evaluate execution parameters for standard Short entries."
        )
        send_email(subject=subject, body=body, from_email=FROM_EMAIL, to_email=TO_EMAIL, attachment=None)
        last_alerted_candles[TICKER] = current_candle_time
        print(f"*** SHORT ALERT SENT FOR {TICKER} ***")



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
                        watchlist = pd.read_csv("watchlist.csv")
                        
                        for index, row in watchlist.iterrows():
                            ticker = str(row['Ticker']).strip()
                            support = float(row['Support'])
                            resistance = float(row['Resistance'])
                            vol_len = int(row['Vol_Length'])
                            
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

