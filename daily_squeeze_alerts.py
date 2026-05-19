# pip install yfinance pandas numpy schedule pytz tqdm

import os
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import schedule
import time
from email.message import EmailMessage
from datetime import datetime
import pytz
from tqdm import tqdm

# ---------------- CONFIG ---------------- #
# CRITICAL EMAIL FIX: If using Gmail, you MUST use an "App Password". 
# Normal passwords will be blocked by Google security.
EMAIL_ADDRESS = "amitadiraju3@gmail.com"
EMAIL_PASSWORD = "cedh jcng vpuc onex" 
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


# ---------------- MAIN SCAN ---------------- #
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

    print(f"\nScanning {len(filtered_tickers)} filtered tickers for Squeezes...")
    results = []

    for ticker in tqdm(filtered_tickers, desc="Checking Squeeze Logic"):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo")

            # Handle delisted/empty data cleanly
            if df is None or df.empty or len(df) < 100:
                continue

            # Compute indicators
            squeeze = compute_squeeze(df)
            
            # Need at least 7 days to evaluate 6 days of squeeze + 1 day of firing
            last_7 = squeeze.tail(7).tolist()
            if len(last_7) < 7:
                continue

            # LOGIC 1: Is it currently in a 6-day squeeze? (Looking at the most recent 6 days)
            in_6_day_squeeze = "YES" if all(last_7[-6:]) else "NO"

            # LOGIC 2: Did it squeeze for 6 days, and FIRE on the 7th?
            six_days_prior_squeeze = all(last_7[:6]) # Days 1 through 6 were True
            fired_on_7th = not last_7[6]             # Day 7 (Today) is False
            squeeze_fired = "YES" if (six_days_prior_squeeze and fired_on_7th) else "NO"

            # ONLY append to our list if it is actively doing one of these two things!
            if in_6_day_squeeze == "YES" or squeeze_fired == "YES":
                ema_position = compute_ema_position(df)
                stock_price = df['Close'].iloc[-1]
                year_of_ipo = None
                try:
                    year_of_ipo = stock.info.get("ipoYear")
                except Exception:
                    year_of_ipo = None

                results.append({
                    "STOCK TICKER": ticker,
                    "STOCK PRICE": stock_price,
                    "YEAR OF IPO": year_of_ipo,
                    "IN 6 DAY SQUEEZE": in_6_day_squeeze,
                    "SQUEEZE_FIRED": squeeze_fired,
                    "PRICE RELATIVE TO EMA": ema_position
                })

        except Exception:
            continue

    if len(results) == 0:
        print("\nNo stocks met the Squeeze criteria today.")
        return

    # Print accurate numbers
    print(f"\nFound {len(results)} actionable Squeeze setups! Sending email now...")
    result_df = pd.DataFrame(results)
    # result_df.to_csv(CSV_NAME, index=False)
    
    # Send email and print preview
    send_email(CSV_NAME)
    print("\n--- RESULTS PREVIEW ---")
    print(result_df)


# ---------------- EMAIL ---------------- #
def send_email(csv_file):
    msg = EmailMessage()
    msg["Subject"] = "Daily Squeeze Scanner Results"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg.set_content("Attached are today's actionable squeeze scan results.")

    try:
        with open(csv_file, "rb") as f:
            file_data = f.read()

        msg.add_attachment(file_data, maintype="application", subtype="csv", filename=csv_file)

        # Port 587 + starttls() is the most robust method for bypassing standard SMTP blocks
        print("Connecting to SMTP server...")
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("Email sent successfully!")

    except smtplib.SMTPAuthenticationError:
        print("EMAIL FAILED: Authentication Error. Ensure you are using a Gmail App Password, not your normal password.")
    except Exception as e:
        print(f"EMAIL FAILED: {e}")


# ---------------- SCHEDULER ---------------- #
def job():
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)
    print(f"\n--- Running scan at: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    
    # Your target scanning and trading logic goes here:
    # run_scan() 


def main():
    runat = "17:00"
    
    # Explicitly passing the timezone string inside .at() requires schedule version 1.2.0+
    schedule.every().day.at(runat, "US/Eastern").do(job)
    
    print(f"Scanner initialized. Waiting quietly in background to run daily at {runat} US/Eastern.")
    # REMOVED: job() <- This line was causing the instant execution bug on startup.

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()