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
    sma = df['Close'].rolling(window=length).mean()
    
    # FIX 1: Force ddof=0 for Population Standard Deviation. 
    # This aligns the math exactly with TradingView & ThinkOrSwim.
    std = df['Close'].rolling(window=length).std(ddof=0)
    
    bb_upper = sma + (2 * std)
    bb_lower = sma - (2 * std)

    # Keltner Channels
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=length).mean()
    
    kc_upper = sma + (1.5 * atr)
    kc_lower = sma - (1.5 * atr)

    # Squeeze is ON when Bollinger Bands are completely inside Keltner Channels
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    
    # FIX 2: Fill NaNs from rolling windows to avoid boolean logic crashes
    return squeeze_on.fillna(False)




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


# Computes a stock's Long levels #
def calculate_long_levels(stock_price, atr, sl_mult_1, tp_mult_1, sl_mult_2=None, tp_mult_2=None):
    """
    Calculates Long entry Stop Loss (SL) and Take Profit (TP) price targets.
    For a long position: SL sits below entry, TP sits above entry.
    """
    levels = {
        'SL1': round(stock_price - (sl_mult_1 * atr), 2),
        'TP1': round(stock_price + (tp_mult_1 * atr), 2)
    }
    
    if sl_mult_2 is not None and tp_mult_2 is not None:
        levels['SL2'] = round(stock_price - (sl_mult_2 * atr), 2)
        levels['TP2'] = round(stock_price + (tp_mult_2 * atr), 2)
        
    return levels



# Computes a stocks' short levels #
def calculate_original_levels_short(stock_price, atr, sl_mult_1, tp_mult_1, sl_mult_2=None, tp_mult_2=None):
    """
    Calculates Short entry Stop Loss (SL) and Take Profit (TP) price targets.
    For a short position: SL sits above entry, TP sits below entry.
    """
    levels = {
        'SL1': round(stock_price + (sl_mult_1 * atr), 2),
        'TP1': round(stock_price - (tp_mult_1 * atr), 2)
    }
    
    # Check if a second tier of risk/reward levels is requested
    if sl_mult_2 is not None and tp_mult_2 is not None:
        levels['SL2'] = round(stock_price + (sl_mult_2 * atr), 2)
        levels['TP2'] = round(stock_price - (tp_mult_2 * atr), 2)
        
    return levels


# Evaluate if a option contract makes sense for small account #
# For calls Delta must be +ve. Works for both.
def evaluate_option_trade_viability(
    stock_price, 
    target_price1, 
    stop_loss1, 
    daily_atr, 
    option_price, 
    delta, 
    gamma, 
    theta, 
    dte, 
    target_price2=None, 
    stop_loss2=None, 
    account_size=5000
):
    
    # 1. SMALL ACCOUNT GATE: Max Allocation (10% of Account)
    max_contract_cost = account_size * 0.10
    if (option_price * 100) > max_contract_cost:
        return False, f"FAIL: Contract too expensive (${option_price * 100}). Max allowed is ${max_contract_cost}."

    abs_delta = abs(delta)
    abs_theta = abs(theta)

    # --- EVALUATE PRIMARY LEVEL (TP1 / SL1) ---
    dist_tp1 = abs(target_price1 - stock_price)
    dist_sl1 = abs(stock_price - stop_loss1)
    
    # 2. SMALL ACCOUNT GATE: The Velocity / DTE Rule (Max 30% of DTE for Primary Target)
    est_days_tp1 = dist_tp1 / daily_atr
    if est_days_tp1 > (dte * 0.30):
        return False, f"FAIL: TP1 takes {round(est_days_tp1, 1)} days based on ATR. Too slow for {dte} DTE."

    # Project Option Premium Gain at TP1
    gross_opt_gain1 = (abs_delta * dist_tp1) + (0.5 * gamma * (dist_tp1 ** 2))
    theta_tax1 = abs_theta * est_days_tp1
    net_opt_gain1 = gross_opt_gain1 - theta_tax1
    
    if net_opt_gain1 <= 0.10: 
        return False, "FAIL: Theta decay eats the TP1 profit before target is hit."

    # Project Option Premium Loss at SL1
    gross_opt_loss1 = (abs_delta * dist_sl1) - (0.5 * gamma * (dist_sl1 ** 2))
    net_opt_loss1 = gross_opt_loss1 + abs_theta # 1 day baseline decay
    net_opt_loss1 = max(0.01, min(net_opt_loss1, option_price)) # Cannot lose more than cost

    # 3. SMALL ACCOUNT GATE: Max Absolute Risk (2% of Account on Primary Stop)
    projected_dollar_loss1 = net_opt_loss1 * 100
    max_dollar_loss = account_size * 0.02
    
    if projected_dollar_loss1 > max_dollar_loss:
        return False, f"FAIL: SL1 risks ${round(projected_dollar_loss1, 2)}. Max allowed is ${max_dollar_loss}."

    # Risk / Reward Analysis for Level 1
    stock_rr1 = dist_tp1 / dist_sl1
    option_rr1 = net_opt_gain1 / net_opt_loss1

    # Final Check: Primary Option R:R must be 2.0 or better
    if option_rr1 < 2.0:
        return False, f"FAIL: Option TP1/SL1 R:R is {round(option_rr1, 2)}. Minimum 2.0 required."

    # --- EVALUATE SECONDARY LEVEL (TP2 / SL2) IF PROVIDED ---
    net_opt_gain2 = None
    projected_dollar_loss2 = None
    option_rr2 = None
    est_days_tp2 = None

    if target_price2 is not None and stop_loss2 is not None:
        dist_tp2 = abs(target_price2 - stock_price)
        dist_sl2 = abs(stock_price - stop_loss2)
        
        # Calculate TP2 Time & Profit
        est_days_tp2 = dist_tp2 / daily_atr
        gross_opt_gain2 = (abs_delta * dist_tp2) + (0.5 * gamma * (dist_tp2 ** 2))
        theta_tax2 = abs_theta * est_days_tp2
        net_opt_gain2 = gross_opt_gain2 - theta_tax2
        
        # Calculate SL2 Risk
        gross_opt_loss2 = (abs_delta * dist_sl2) - (0.5 * gamma * (dist_sl2 ** 2))
        net_opt_loss2 = gross_opt_loss2 + abs_theta
        net_opt_loss2 = max(0.01, min(net_opt_loss2, option_price))
        
        projected_dollar_loss2 = net_opt_loss2 * 100
        option_rr2 = net_opt_gain2 / net_opt_loss2 if net_opt_loss2 > 0 else 0

    # Package everything neatly for your scanner logs
    metrics = {
        "TP1_Est_Days": round(est_days_tp1, 1),
        "TP1_Total_Theta_Tax": round(theta_tax1, 2),
        "TP1_Projected_Gain_$": round(net_opt_gain1 * 100, 2), 
        "SL1_Projected_Loss_$": round(projected_dollar_loss1, 2), 
        "Level_1_Stock_RR": round(stock_rr1, 2),
        "Level_1_Option_RR": round(option_rr1, 2),
        
        "TP2_Est_Days": round(est_days_tp2, 1) if est_days_tp2 is not None else None,
        "TP2_Projected_Gain_$": round(net_opt_gain2 * 100, 2) if net_opt_gain2 is not None else None,
        "SL2_Projected_Loss_$": round(projected_dollar_loss2, 2) if projected_dollar_loss2 is not None else None,
        "Level_2_Option_RR": round(option_rr2, 2) if option_rr2 is not None else None
    }

    return True, metrics



# ----- Computes Option Exit prices based on stock TP and SL and other best practices ------ #
# Works for both CALL BUY AND PUT BUY OPTIONS. For calls Delta must be positive
def calculate_option_exit_prices(
    stock_entry: float,
    option_entry: float,
    stock_tp1: float,
    stock_sl1: float,
    delta: float,
    side: str,              # "CALL" or "PUT"
    stock_atr: float,       # Dynamically computes Theta decay based on distance
    stock_tp2: float = None, # OPTIONAL
    stock_sl2: float = None, # OPTIONAL
    theta: float = 0.0,     
    gamma: float = 0.0
) -> dict:
 
    side = side.upper()
    if side not in ["CALL", "PUT"]:
        raise ValueError("Side must be 'CALL' or 'PUT'")

    # Ensure Delta is handled correctly: Calls are positive, Puts are negative.
    abs_delta = abs(delta)
    effective_delta = abs_delta if side == "CALL" else -abs_delta
    abs_theta = abs(theta)

    def get_option_price(stock_target_price: float) -> float:
        # If no optional level was provided, return None
        if stock_target_price is None:
            return None
            
        stock_move = stock_target_price - stock_entry
        
        # 1. Dynamic Theta Deduction
        distance_to_target = abs(stock_move)
        est_days_to_target = distance_to_target / stock_atr if stock_atr > 0 else 0
        total_decay = abs_theta * est_days_to_target
        
        # 2. Linear approximation (Delta)
        # For a PUT: negative move * negative delta = positive premium increase
        delta_impact = stock_move * effective_delta
        
        # 3. Curvature approximation (Gamma)
        # Gamma is always a positive tailwind for long options
        gamma_impact = 0.5 * gamma * (stock_move ** 2)
        
        # Total theoretical change in the option's value
        option_price_change = delta_impact + gamma_impact
        
        # Baseline contract premium before time decay
        estimated_premium = option_entry + option_price_change
        
        # Deduct the time decay for this specific target distance
        final_premium = estimated_premium - total_decay
        
        # Options premium cannot drop below $0.01 in the real market
        return round(max(0.01, final_premium), 2)

    # Output dictionary mapping out the limit/stop trigger premiums
    return {
        "option_tp1": get_option_price(stock_tp1),
        "option_sl1": get_option_price(stock_sl1),
        "option_tp2": get_option_price(stock_tp2),
        "option_sl2": get_option_price(stock_sl2)
    }