import datetime
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
import pytz
import yfinance as yf

# =====================================================================
# --- 1. USER CONFIGURATION ---
# =====================================================================
TICKER = "AAPL"  # Change to any stock ticker
SUPPORT_LEVEL = 80.0  # Your custom confluence support
RESISTANCE_LEVEL = 110.0  # Your custom confluence resistance
VOL_LENGTH = 24  # Rolling 2H bars to calculate baseline (~7 trading days)

# Email Setup (For Gmail, generate an 'App Password' instead of your main password)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"
RECEIVER_EMAIL = "recipient_email@gmail.com"

# Checking Interval (e.g., poll every 15 minutes to capture new bar completions)
CHECK_INTERVAL_SECONDS = 900

# Tracker variable to prevent duplicate emails for the same 2-Hour candle
last_alerted_candle_timestamp = None


# =====================================================================
# --- 2. EMAIL DELIVERY SYSTEM ---
# =====================================================================
def send_email_alert(subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print(f"[{datetime.datetime.now()}] Email alert sent successfully.")
    except Exception as e:
        print(f"[{datetime.datetime.now()}] Failed to send email alert: {e}")


# =====================================================================
# --- 3. ANALYTICS & EXECUTION ENGINE ---
# =====================================================================
def run_trading_logic():
    global last_alerted_candle_timestamp

    print(f"[{datetime.datetime.now()}] Fetching data and verifying structures...")

    # Fetch 30-minute intervals for the last month to construct clean 2H bars
    raw_data = yf.download(
        tickers=TICKER, period="1mo", interval="30m", progress=False
    )

    if raw_data.empty or len(raw_data) < 100:
        print("Insufficient historical data fetched from Yahoo Finance.")
        return

    # Convert Index to Eastern Time to align perfectly with US Market sessions
    raw_data.index = raw_data.index.tz_convert("US/Eastern")

    # Resample 30m candles into 2H candles, anchored precisely to market open (09:30 AM)
    agg_rules = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    df_2h = raw_data.resample("2h", origin="09:30:00").agg(agg_rules).dropna()

    # Calculate rolling Volume SMA strictly across 2-Hour blocks
    df_2h["Vol_SMA"] = df_2h["Volume"].rolling(window=VOL_LENGTH).mean()

    # Ensure dataset is large enough to check history safely
    if len(df_2h) < (VOL_LENGTH + 4):
        print("Dataframe processing window initialization incomplete.")
        return

    # --- FIBONACCI REVERSAL MATHEMATICS ---
    price_range = RESISTANCE_LEVEL - SUPPORT_LEVEL
    bull_091 = SUPPORT_LEVEL + (price_range * 0.09)
    bear_091 = RESISTANCE_LEVEL - (price_range * 0.09)

    # Risk Control Management Bounds (The Chasing Cap)
    bull_max_chase = SUPPORT_LEVEL + (price_range * 0.214)
    bear_max_chase = RESISTANCE_LEVEL - (price_range * 0.214)

    # --- EXTRACT FULLY CLOSED DATA ARRAYS ---
    # iloc[-1] is the current open/uncompleted 2H candle.
    # To match Pine Script's bar-close rules, we extract completely processed indices:
    latest_closed_bar = df_2h.iloc[-2]
    prev_closed_bar = df_2h.iloc[-3]
    two_bars_ago = df_2h.iloc[-4]

    # Assign target analytical metrics
    c2h = float(latest_closed_bar["Close"])
    h2h = float(latest_closed_bar["High"])
    l2h = float(latest_closed_bar["Low"])
    v2h = float(latest_closed_bar["Volume"])
    v2h_avg = float(latest_closed_bar["Vol_SMA"])

    c2h_prev = float(prev_closed_bar["Close"])
    v2h_prev = float(prev_closed_bar["Volume"])

    c2h_prev2 = float(two_bars_ago["Close"])

    # Unique identification handle for the evaluated bar
    current_candle_time = str(df_2h.index[-2])

    # --- VALIDATION GATES ---
    # 1. Wick Proportions Check (Ensuring dominant institutional close)
    long_wick_ok = (h2h - c2h) <= ((h2h - l2h) * 0.35)
    short_wick_ok = (c2h - l2h) <= ((h2h - l2h) * 0.35)

    # 2. Volume Consistency Verification
    volume_confirmed = (v2h > v2h_avg) and (v2h_prev > v2h_avg)

    # 3. Structural Shifts Verification
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
        and (current_candle_time != last_alerted_candle_timestamp)
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
        send_email_alert(subject, body)
        last_alerted_candle_timestamp = current_candle_time

    elif (
        price_short_confirmed
        and volume_confirmed
        and short_wick_ok
        and (current_candle_time != last_alerted_candle_timestamp)
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
        send_email_alert(subject, body)
        last_alerted_candle_timestamp = current_candle_time
    
    else:
        print(
            f"Status Check: Criteria scanned. No valid configurations met for {TICKER} at closed timestamp {current_candle_time}."
        )


# =====================================================================
# --- 4. CONTINUOUS MONITORING LOOP ---
# =====================================================================
if __name__ == "__main__":
    eastern = pytz.timezone("US/Eastern")
    print("Initializing Automated Reversal Script Engine...")

    while True:
        try:
            now = datetime.datetime.now(eastern)

            # Check if current day is a weekday (Monday=0 to Friday=4)
            if now.weekday() < 5:
                # Establish market bounds
                market_open = now.replace(
                    hour=9, minute=30, second=0, microsecond=0
                )
                market_close = now.replace(
                    hour=16, minute=0, second=0, microsecond=0
                )

                if market_open <= now <= market_close:
                    run_trading_logic()
                    time.sleep(CHECK_INTERVAL_SECONDS)
                else:
                    print(
                        f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Market is currently closed. Sleeping for 60 seconds..."
                    )
                    time.sleep(60)
            else:
                print(
                    f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Weekend Mode Active. Sleeping for 300 seconds..."
                )
                time.sleep(300)

        except Exception as global_error:
            print(f"Runtime Exception Intercepted: {global_error}")
            time.sleep(60)  # Cool down before trying again if internet drops