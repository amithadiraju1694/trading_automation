import os
import mimetypes
import smtplib
from email.message import EmailMessage
import numpy as np
import pandas as pd
from tqdm import tqdm
import yfinance as yf

FILTERED_MKT_CAP_CSV = "filtered_by_market_cap.csv"
def send_email(subject, body, from_email, to_email, attachment=None):
    """
    Send an email with optional attachment.
    
    Args:
        subject (str): Email subject (mandatory)
        body (str): Email body content (mandatory)
        from_email (str): Sender email address (mandatory)
        to_email (str): Recipient email address (mandatory)
        attachment (str, optional): Path to file to attach. If None, no attachment is sent.
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    email_password = os.getenv("EMAIL_PASSWORD")
    
    if not email_password:
        print("EMAIL FAILED: EMAIL_PASSWORD environment variable not set.")
        return False
    
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        msg.set_content(body)
        
        # Add attachment if provided
        if attachment:
            if not os.path.exists(attachment):
                print(f"EMAIL FAILED: Attachment file not found: {attachment}")
                return False
            
            with open(attachment, "rb") as f:
                file_data = f.read()
            
            file_name = os.path.basename(attachment)
            # Detect MIME type based on file extension
            mime_type, _ = mimetypes.guess_type(attachment)
            
            if mime_type:
                maintype, subtype = mime_type.split("/", 1)
            else:
                # Default to octet-stream for unknown types
                maintype, subtype = "application", "octet-stream"
            
            msg.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=file_name)
        
        # Port 587 + starttls() is the most robust method for bypassing standard SMTP blocks
        print("Connecting to SMTP server...")
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(from_email, email_password)
            smtp.send_message(msg)
            print("Email sent successfully!")
        
        return True
    
    except smtplib.SMTPAuthenticationError:
        print("EMAIL FAILED: Authentication Error. Ensure you are using a Gmail App Password, not your normal password.")
        return False
    except Exception as e:
        print(f"EMAIL FAILED: {e}")
        return False


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




def compute_rsi(df, period=14):
    """Calculates the Relative Strength Index (RSI) using Wilder's Smoothing."""
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Wilder's Smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def compute_adx(df, period=14):
    """Calculates the Average Directional Index (ADX) using Wilder's Smoothing."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Plus/Minus Directional Movement
    up_move = high.diff()
    down_move = low.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder's Smoothing for TR and DM
    tr_smoothed = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smoothed)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smoothed)
    
    # Directional Index (DX) and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx


import pandas as pd
import yfinance as yf
from tqdm import tqdm

FILTERED_MKT_CAP_CSV = "filtered_market_cap.csv"

# ---------------- GET ALL STOCKS ---------------- #
def get_all_tickers():
    """Fetches real-time US tickers (NASDAQ, NYSE, AMEX) directly from NASDAQ Trader."""
    try:
        # Fetch both NASDAQ and non-NASDAQ (Other) listings
        nasdaq_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
        other_url = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
        
        nasdaq = pd.read_csv(nasdaq_url, sep="|")
        other = pd.read_csv(other_url, sep="|")
        
        # Remove the text footer row ("File Creation Time...") that NASDAQ appends
        nasdaq = nasdaq[nasdaq["Symbol"].str.contains("File Creation Time", na=False) == False]
        other = other[other["ACT Symbol"].str.contains("File Creation Time", na=False) == False]
        
        # Filter out Test listings
        nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]
        other = other[other["Test Issue"] == "N"]
        
        # Combine lists
        tickers = nasdaq["Symbol"].dropna().tolist() + other["ACT Symbol"].dropna().tolist()
        
        # Filter out warrants, preferred shares, and weird tickers
        tickers = [t for t in tickers if not any(char in t for char in ["-", "^", ".", "$"])]
        
        return sorted(list(set(tickers)))
        
    except Exception as e:
        print(f"Error fetching ticker directory: {e}")
        return []

#TODO: Filter functions are too slow, is it the WiFi or Code? Check it.
#TODO: Filtering for 1-2% movers out of specific market cap gives too many stocks ( ~2K ) what other basic filters to trim down to few high probability setups
# either breakout wise or trend wise ( near supp & res )
# ---------------- FILTER BY MARKET CAP ---------------- #
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
            
            # 1. Try fast_info property access (No .get())
            info = stock.fast_info
            market_cap = getattr(info, "market_cap", None)

            # 2. Robust fallback to regular .info dictionary if fast_info fails/returns None
            if market_cap is None or market_cap == 0:
                market_cap = stock.info.get("marketCap", 0)

            # Filter threshold (100 Million)
            if market_cap and market_cap >= 100_000_000:
                filtered.append(ticker)
                
        except Exception:
            # Silently catch truly delisted/broken tickers
            continue

    
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

