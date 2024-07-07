import asyncio
import websockets
import requests
import pandas as pd
import numpy as np
import os

import google.generativeai as genai

genai.configure(api_key="KEY")

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
    
    combined_prediction = (moving_avg[-1] + macd_line[-1] + rsi[-1]) / 3
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

async def handle_websocket(websocket, path):
    async for message in websocket:
        print(f"Received message from client: {message}")
        api_key = 'key'
        symbol = message
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
                "Simple Moving Average": moving_avg,
                "Exponential Moving Average": ema,
                "Next Week Predictions": next_week_predictions,
                "MACD Line": macd_line,
                "Signal Line": signal_line,
                "Bollinger Bands - Average": bollinger_avg,
                "Bollinger Bands - Upper Band": upper_band,
                "Bollinger Bands - Lower Band": lower_band,
                "RSI": rsi,
                "Support Levels": support_levels,
                "Resistance Levels": resistance_levels,
                "Average Volume": average_volume,
                "Volume Spikes": volume_spikes,
                "Elliott Wave Count": elliott_wave_count,
                "Trend Lines": trend_lines,
                "Fibonacci Retracement Levels": fib_levels,
            }

            # Prepare output for the chatbot
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
            ] + [line for line in trend_lines] + [
                "Fibonacci Retracement Levels:"
            ] + [
                f"{level}: {price:.2f}" for level, price in fib_levels.items()
            ]

            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }

            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=generation_config,
                system_instruction="You are the best stock market predictor, you are willing to take any necessary risk and your users know the risk as well",
            )
            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            "From now on you will get my data and try to predict the next price accurately with no fear\n",
                        ],
                    },
                    {
                        "role": "model",
                        "parts": [
                            "Yes, Sir. From now on I'll say the next price only. If you give me data X,Y,Z I'll tell you my prediction P trying to decide only using the given data, since this is just an experiment. Since I know the prediction price will be included I'll use an HTML format when replying"
                        ],
                    }
                ]
            )

            svar = str(output_lines).strip('[]')
            qvar = "What is the next price of the stock? if the data is " + svar
            response = chat_session.send_message(qvar)
            print(response.text)
            final = symbol + response.text
        else:
            final = "Failed to retrieve stock data. Please check your API key and symbol."
        await websocket.send(final)

async def main():
    async with websockets.serve(handle_websocket, "localhost", 8765):
        await asyncio.Future()  # Keep the server running indefinitely

asyncio.run(main())
