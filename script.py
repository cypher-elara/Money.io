from flask import Flask, request, jsonify
import requests
import pandas as pd
import numpy as np
import os
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

app = Flask(__name__)

def get_stock_data(symbol, api_key, interval='daily', output_size='compact'):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_{interval.upper()}&symbol={symbol}&apikey={api_key}&outputsize={output_size}'
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'Error Message' in data:
            raise ValueError(f"Error retrieving data: {data['Error Message']}")
        
        if interval == 'daily':
            time_series_key = 'Time Series (Daily)'
        elif interval == 'weekly':
            time_series_key = 'Weekly Time Series'
        elif interval == 'monthly':
            time_series_key = 'Monthly Time Series'
        else:
            raise ValueError("Invalid interval. Choose from 'daily', 'weekly', or 'monthly'.")
        
        df = pd.DataFrame(data[time_series_key]).T
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_moving_average(data, window=7):
    return data['close'].rolling(window=window).mean()

def calculate_ema(data, span=7):
    return data['close'].ewm(span=span, adjust=False).mean()

def predict_next_week_combined(data, window=7):
    moving_avg = calculate_moving_average(data, window)
    macd_line, signal_line = calculate_macd(data)
    rsi = calculate_rsi(data)
    
    if len(moving_avg) == 0 or len(macd_line) == 0 or len(rsi) == 0:
        return []

    combined_prediction = (moving_avg.iloc[-1] + macd_line.iloc[-1] + rsi.iloc[-1]) / 3
    predictions = [combined_prediction] * 7
    
    return predictions

def calculate_macd(data, short_span=12, long_span=26, signal_span=9):
    short_ema = data['close'].ewm(span=short_span, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_span, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return rolling_mean, upper_band, lower_band

def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_support_resistance(data, window=14):
    support_levels = []
    resistance_levels = []
    
    for i in range(window, len(data) - window):
        window_data = data['close'][i-window:i+window]
        local_min = window_data.min()
        local_max = window_data.max()
        
        if data['close'][i] == local_min:
            support_levels.append((data.index[i], local_min))
        
        if data['close'][i] == local_max:
            resistance_levels.append((data.index[i], local_max))
    
    return support_levels, resistance_levels

def calculate_volume_analysis(data, window=20, spike_factor=2):
    average_volume = data['volume'].rolling(window=window).mean()
    volume_spikes = data['volume'] > (average_volume * spike_factor)
    return average_volume, volume_spikes

def calculate_elliott_wave(data):
    if len(data) < 5:
        return "Not enough data for Elliott Wave analysis."

    last_close = data['close'].iloc[-1]
    prev_close = data['close'].iloc[-2]

    if last_close > prev_close:
        return "Upward impulse or wave 1"

    elif last_close < prev_close:
        return "Downward correction or wave A"

    else:
        return "No clear Elliott Wave pattern identified"

def calculate_trend_lines(data, min_points=3):
    trend_lines = []
    prices = data['close'].values
    dates = data.index.values

    local_min_indices = (np.diff(np.sign(np.diff(prices))) > 0).nonzero()[0] + 1
    local_max_indices = (np.diff(np.sign(np.diff(prices))) < 0).nonzero()[0] + 1

    for min_idx in local_min_indices:
        for max_idx in local_max_indices:
            if max_idx > min_idx and max_idx - min_idx >= min_points:
                start_date = dates[min_idx - 1]
                start_price = prices[min_idx - 1]
                end_date = dates[max_idx]
                end_price = prices[max_idx]
                trend_lines.append((start_date, start_price, end_date, end_price))

    return trend_lines

def calculate_fibonacci_retracement(data, high, low):
    prices = data.loc[(data.index >= high) & (data.index <= low), 'close']
    high_price = prices.max()
    low_price = prices.min()
    range_price = high_price - low_price

    fib_levels = {
        '23.6%': high_price - range_price * 0.236,
        '38.2%': high_price - range_price * 0.382,
        '50.0%': high_price - range_price * 0.5,
        '61.8%': high_price - range_price * 0.618,
        '76.4%': high_price - range_price * 0.764
    }

    return fib_levels

@app.route('/predict', methods=['POST'])
def handle_request():
    symbol = request.form['symbol']
    api_key = 'YOUR_API_KEY'
    stock_data = get_stock_data(symbol, api_key)
    if not stock_data.empty:
        moving_avg = calculate_moving_average(stock_data)
        ema = calculate_ema(stock_data)
        next_week_predictions = predict_next_week_combined(stock_data)
        macd_line, signal_line = calculate_macd(stock_data)
        bollinger_avg, upper_band, lower_band = calculate_bollinger_bands(stock_data)
        rsi = calculate_rsi(stock_data)
        support_levels, resistance_levels = calculate_support_resistance(stock_data)
        average_volume, volume_spikes = calculate_volume_analysis(stock_data)
        elliott_wave_count = calculate_elliott_wave(stock_data)
        trend_lines = calculate_trend_lines(stock_data)
        high_date = '2024-06-27'  # Replace with actual high point date
        low_date = '2024-06-01'   # Replace with actual low point date
        fib_levels = calculate_fibonacci_retracement(stock_data, high_date, low_date)

        output_dict = {
            "Simple Moving Average": moving_avg.to_dict(),
            "Exponential Moving Average": ema.to_dict(),
            "Next Week Predictions": next_week_predictions,
            "MACD Line": macd_line.to_dict(),
            "Signal Line": signal_line.to_dict(),
            "Bollinger Bands - Average": bollinger_avg.to_dict(),
            "Bollinger Bands - Upper Band": upper_band.to_dict(),
            "Bollinger Bands - Lower Band": lower_band.to_dict(),
            "RSI": rsi.to_dict(),
            "Support Levels": support_levels,
            "Resistance Levels": resistance_levels,
            "Average Volume": average_volume.to_dict(),
            "Volume Spikes": volume_spikes.to_list(),
            "Elliott Wave Count": elliott_wave_count,
            "Trend Lines": trend_lines,
            "Fibonacci Retracement Levels": fib_levels,
        }

        # Convert output to strings for the chatbot
        output_lines = [
            f"Simple Moving Average:\n {moving_avg}",
            f"Exponential Moving Average:\n {ema}",
            f"Next week's predicted closing prices: {next_week_predictions}",
            f"MACD Line:\n {macd_line}",
            f"Signal Line:\n {signal_line}",
            f"Bollinger Bands - Moving Average:\n {bollinger_avg}",
            f"Upper Band:\n {upper_band}",
            f"Lower Band:\n {lower_band}",
            f"RSI:\n {rsi}",
            f"Support Levels:\n {support_levels}",
            f"Resistance Levels:\n {resistance_levels}",
            f"Average Volume:\n {average_volume}",
            f"Volume Spikes:\n {volume_spikes}",
            f"Elliott Wave count: {elliott_wave_count}",
            "Trend Lines:"
        ] + [f"{line[0]} to {line[2]}: {line[1]} to {line[3]}" for line in trend_lines] + [
            "Fibonacci Retracement Levels:"
        ] + [
            f"{level}: {price:.2f}" for level, price in fib_levels.items()
        ]

        # Assuming correct usage of the genai library
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        # Replace with actual model initialization
        response = genai.generate_text(
            model="gemini-1.5-flash",
            prompt="What is the next price of the stock? " + "\n".join(output_lines),
            **generation_config
        )

        return jsonify(response)

    return jsonify({"error": "Error retrieving stock data"})

if __name__ == '__main__':
    app.run(debug=True)
