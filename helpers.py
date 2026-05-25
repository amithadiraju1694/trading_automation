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
    close_series = df["Close"]
    ema8 = close_series.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close_series.ewm(span=21, adjust=False).mean().iloc[-1]
    ema34 = close_series.ewm(span=34, adjust=False).mean().iloc[-1]
    ema55 = close_series.ewm(span=55, adjust=False).mean().iloc[-1]
    ema89 = close_series.ewm(span=89, adjust=False).mean().iloc[-1]

    close = close_series.iloc[-1]
    max_ema = max(ema8, ema21, ema34, ema55, ema89)
    min_ema = min(ema8, ema21, ema34, ema55, ema89)

    if close > max_ema:
        ema_position = "ABOVE_EMAS"
    elif close < min_ema:
        ema_position = "BELOW_EMAS"
    else:
        ema_position = "BETWEEN"

    if (ema8 > ema21) and (ema21 > ema34) and (ema34 > ema55) and (ema55 > ema89):
        ema_comparison = "STRICT BULLISH"
    elif (ema8 < ema21) and (ema21 < ema34) and (ema34 < ema55) and (ema55 < ema89):
        ema_comparison = "STRICT BEARISH"
    elif (ema8 < ema21) and (ema21 > ema34) and (ema34 > ema55) and (ema55 > ema89):
        ema_comparison = "PULLBACK BULLISH LT BUY"
    elif (ema8 > ema21) and (ema21 > ema34) and ((ema34 < ema55) or (ema55 < ema89)):
        ema_comparison = "REVERSAL BULLISH STRBUY"
    else:
        ema_comparison = "NEUTRAL"

    return ema_position, ema_comparison


 
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

# Inverse Stock Long Levels ( when original is short)
def calculate_inverse_levels_short(inverse_entry, stock_entry, original_levels, leverage=1):
    """
    Translates underlying stock targets to an inverse Long ETP position.
    - Underlying increases (hits SL) -> Inverse ETP drops to ETP SL.
    - Underlying decreases (hits TP) -> Inverse ETP rises to ETP TP.
    """
    inverse_levels = {}
    
    for level_name, original_price in original_levels.items():
        if 'SL' in level_name:
            # Underlying stock moved UP: Inverse ETP moves DOWN
            pct_change = (original_price - stock_entry) / stock_entry
            inverse_levels[level_name] = round(inverse_entry * (1 - (leverage * pct_change)), 2)
            
        elif 'TP' in level_name:
            # Underlying stock moved DOWN: Inverse ETP moves UP
            pct_change = (stock_entry - original_price) / stock_entry
            inverse_levels[level_name] = round(inverse_entry * (1 + (leverage * pct_change)), 2)
            
    return inverse_levels


# If Original is long, how much will inverse be shorted
def calculate_short_inverse_levels(inverse_entry, stock_entry, original_long_levels, leverage=1):
    """
    Translates underlying Long targets to a Short Inverse ETP position.
    - Underlying increases (hits TP) -> Inverse ETP drops (ETP TP target achieved).
    - Underlying decreases (hits SL) -> Inverse ETP rises (ETP SL hit).
    """
    inverse_levels = {}
    
    for level_name, original_price in original_long_levels.items():
        if 'SL' in level_name:
            # Underlying stock moved DOWN: Inverse ETP moves UP (This is your ETP Stop Loss)
            pct_change = (stock_entry - original_price) / stock_entry
            inverse_levels[level_name] = round(inverse_entry * (1 + (leverage * pct_change)), 2)
            
        elif 'TP' in level_name:
            # Underlying stock moved UP: Inverse ETP moves DOWN (This is your ETP Take Profit target)
            pct_change = (original_price - stock_entry) / stock_entry
            inverse_levels[level_name] = round(inverse_entry * (1 - (leverage * pct_change)), 2)
            
    return inverse_levels

# If QQQ is Long, how much will TQQQ move in same direction
def calculate_leveraged_long_levels(leveraged_entry, stock_entry, original_long_levels, leverage=3):
    """
    Translates underlying Long targets to a Long Leveraged ETP position (e.g., TQQQ).
    - Underlying increases (hits TP) -> Leveraged ETP moves UP (ETP TP).
    - Underlying decreases (hits SL) -> Leveraged ETP moves DOWN (ETP SL).
    """
    leveraged_levels = {}
    
    for level_name, original_price in original_long_levels.items():
        if 'SL' in level_name:
            # Underlying stock moved DOWN: Leveraged ETP moves DOWN
            pct_change = (stock_entry - original_price) / stock_entry
            leveraged_levels[level_name] = round(leveraged_entry * (1 - (leverage * pct_change)), 2)
            
        elif 'TP' in level_name:
            # Underlying stock moved UP: Leveraged ETP moves UP
            pct_change = (original_price - stock_entry) / stock_entry
            leveraged_levels[level_name] = round(leveraged_entry * (1 + (leverage * pct_change)), 2)
            
    return leveraged_levels



# ------------------------------------------- #
# SINGLE LEG OPTION TP AND SL LEVELS ALONG WITH VIABILITY
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

"""
Scan: Run the script with only your Long Leg data. Leave the short parameters at 0.0.
Identify: The script will output "Recommended_Short_Strike": 160.
Fetch & Re-run: Look up the $160 Call on your broker, grab its premium, Delta, Gamma, and Theta, and plug them into the short_ parameters of the function.
Deploy: The script will return exactly what the Net Spread will be worth at your TP1 and SL1 levels.

"""
# ------------------------------------------- #
# DOUBLE LEG OPTION TP AND SL Levels computation WITHOUT R:R Filter
def calculate_debit_spread_exits(
    major_direction: str,
    stock_entry: float,
    stock_atr: float,
    stock_tp1: float,
    stock_sl1: float,
    long_strike: float,
    long_premium: float,
    long_delta: float,
    long_gamma: float,
    long_theta: float,
    stock_tp2: float = None,
    stock_sl2: float = None,
    short_premium: float = 0.0,
    short_delta: float = 0.0,
    short_gamma: float = 0.0,
    short_theta: float = 0.0
) -> dict:
    
    direction = major_direction.strip().upper()
    if direction not in ["CALL BUY", "PUT BUY"]:
        raise ValueError("major_direction must be 'CALL BUY' or 'PUT BUY'")

    # 1. Determine the Optimal Short Strike
    # For debit spreads, always sell the strike exactly at your primary target.
    ideal_short_strike = stock_tp1
    
    # 2. Safety Gate: Prevent inverted/credit spreads
    if direction == "CALL BUY" and ideal_short_strike <= long_strike:
        print("WARNING: Selling a call below your long strike creates a Bearish Credit Spread. Short strike should be > Long strike.")
    elif direction == "PUT BUY" and ideal_short_strike >= long_strike:
        print("WARNING: Selling a put above your long strike creates a Bullish Credit Spread. Short strike should be < Long strike.")

    # Greeks handling: Calls are positive delta, Puts are negative delta.
    # The Short Leg delta is naturally inverted because you are selling it, 
    # but we handle the subtraction logically below, so input the raw broker Greeks.
    eff_long_delta = abs(long_delta) if direction == "CALL BUY" else -abs(long_delta)
    eff_short_delta = abs(short_delta) if direction == "CALL BUY" else -abs(short_delta)
    
    abs_long_theta = abs(long_theta)
    abs_short_theta = abs(short_theta) # Short theta becomes a positive tailwind

    def project_leg_price(stock_target: float, entry_prem: float, delta: float, 
                          gamma: float, abs_theta: float) -> float:
        if stock_target is None: return 0.0
        
        stock_move = stock_target - stock_entry
        est_days = abs(stock_move) / stock_atr if stock_atr > 0 else 0
        
        # Taylor Series Expansion
        delta_impact = stock_move * delta
        gamma_impact = 0.5 * gamma * (stock_move ** 2)
        decay = abs_theta * est_days
        
        final_prem = (entry_prem + delta_impact + gamma_impact) - decay
        return max(0.01, final_prem) # Minimum market price

    def get_net_spread_price(stock_target: float) -> float:
        if stock_target is None: return None
        
        long_leg_val = project_leg_price(stock_target, long_premium, eff_long_delta, long_gamma, abs_long_theta)
        short_leg_val = project_leg_price(stock_target, short_premium, eff_short_delta, short_gamma, abs_short_theta)
        
        # Net Spread Value = Value of the leg you own MINUS value of the leg you owe
        net_value = long_leg_val - short_leg_val
        return round(max(0.01, net_value), 2)

    # 3. Compile the Output
    net_entry_cost = round(long_premium - short_premium, 2)
    
    results = {
        "Recommended_Short_Strike": ideal_short_strike,
        "Net_Spread_Entry_Cost": net_entry_cost,
        "Net_Spread_TP1": get_net_spread_price(stock_tp1),
        "Net_Spread_SL1": get_net_spread_price(stock_sl1)
    }
    
    if stock_tp2 is not None and stock_sl2 is not None:
        results["Net_Spread_TP2"] = get_net_spread_price(stock_tp2)
        results["Net_Spread_SL2"] = get_net_spread_price(stock_sl2)
        
    return results


# WITH R:R Filter and other gaurdrails for small accounts
def evaluate_debit_spread_trade(
    major_direction: str,
    stock_entry: float,
    stock_atr: float,
    stock_tp1: float,
    stock_sl1: float,
    long_strike: float,
    long_premium: float,
    long_delta: float,
    long_gamma: float,
    long_theta: float,
    short_premium: float = 0.0,
    short_delta: float = 0.0,
    short_gamma: float = 0.0,
    short_theta: float = 0.0,
    stock_tp2: float = None,
    stock_sl2: float = None
) -> tuple:
    
    direction = major_direction.strip().upper()
    if direction not in ["CALL BUY", "PUT BUY"]:
        raise ValueError("major_direction must be 'CALL BUY' or 'PUT BUY'")

    # 1. Determine the Optimal Short Strike & Spread Width
    ideal_short_strike = stock_tp1
    spread_width = abs(ideal_short_strike - long_strike)

    # Greeks handling: Calls are positive delta, Puts are negative delta.
    eff_long_delta = abs(long_delta) if direction == "CALL BUY" else -abs(long_delta)
    eff_short_delta = abs(short_delta) if direction == "CALL BUY" else -abs(short_delta)
    
    abs_long_theta = abs(long_theta)
    abs_short_theta = abs(short_theta) 

    def project_leg_price(stock_target: float, entry_prem: float, delta: float, 
                          gamma: float, abs_theta: float) -> float:
        if stock_target is None: return 0.0
        
        stock_move = stock_target - stock_entry
        est_days = abs(stock_move) / stock_atr if stock_atr > 0 else 0
        
        delta_impact = stock_move * delta
        gamma_impact = 0.5 * gamma * (stock_move ** 2)
        decay = abs_theta * est_days
        
        final_prem = (entry_prem + delta_impact + gamma_impact) - decay
        return max(0.01, final_prem) 

    def get_net_spread_price(stock_target: float) -> float:
        if stock_target is None: return None
        
        long_leg_val = project_leg_price(stock_target, long_premium, eff_long_delta, long_gamma, abs_long_theta)
        short_leg_val = project_leg_price(stock_target, short_premium, eff_short_delta, short_gamma, abs_short_theta)
        
        net_value = long_leg_val - short_leg_val
        return round(max(0.01, net_value), 2)

    # 2. Calculate Base Metrics
    net_entry_cost = round(long_premium - short_premium, 2)
    net_spread_tp1 = get_net_spread_price(stock_tp1)
    net_spread_sl1 = get_net_spread_price(stock_sl1)

    # 3. Calculate Risk vs. Reward
    # Reward is what you make on top of what you paid. Risk is what you lose from your initial cost.
    projected_reward = net_spread_tp1 - net_entry_cost
    projected_risk = net_entry_cost - net_spread_sl1
    
    # Prevent division by zero if Stop Loss projection perfectly equals entry cost
    if projected_risk <= 0:
        projected_risk = 0.01 
        
    spread_rr = round(projected_reward / projected_risk, 2)

    # 4. SMALL ACCOUNT GUARDRAILS
    is_viable = True
    fail_reasons = []

    # Guardrail A: The 50% Cap Rule
    max_allowed_cost = spread_width * 0.50
    if net_entry_cost > max_allowed_cost:
        is_viable = False
        fail_reasons.append(f"Cost (${net_entry_cost}) exceeds 50% of Spread Width (${max_allowed_cost}).")

    # Guardrail B: The 2.0 R:R Rule
    if spread_rr < 2.0:
        is_viable = False
        fail_reasons.append(f"Spread R:R is only {spread_rr}. Minimum required is 2.0.")

    # 5. Compile the Output
    metrics = {
        "Recommended_Short_Strike": ideal_short_strike,
        "Spread_Width": spread_width,
        "Net_Spread_Entry_Cost": net_entry_cost,
        "Net_Spread_TP1": net_spread_tp1,
        "Net_Spread_SL1": net_spread_sl1,
        "Projected_Reward": round(projected_reward, 2),
        "Projected_Risk": round(projected_risk, 2),
        "Spread_RR": spread_rr,
        "Fail_Reasons": fail_reasons if not is_viable else ["None - Trade is strictly viable."]
    }
    
    if stock_tp2 is not None and stock_sl2 is not None:
        metrics["Net_Spread_TP2"] = get_net_spread_price(stock_tp2)
        metrics["Net_Spread_SL2"] = get_net_spread_price(stock_sl2)
        
    return is_viable, metrics
