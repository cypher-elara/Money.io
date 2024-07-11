# Stock Prediction API with Flask

This Flask application provides a simple API for predicting stock prices based on historical data using various technical indicators.

## Features

- **Simple Moving Average (SMA)**: Calculates the moving average of closing prices.
- **Exponential Moving Average (EMA)**: Calculates the exponential moving average of closing prices.
- **MACD (Moving Average Convergence Divergence)**: Computes the MACD line and signal line.
- **Bollinger Bands**: Calculates the moving average, upper band, and lower band.
- **RSI (Relative Strength Index)**: Computes the RSI value.
- **Support and Resistance Levels**: Identifies support and resistance levels based on price data.
- **Average Volume and Volume Spikes**: Analyzes average volume and detects volume spikes.
- **Elliott Wave Analysis**: Determines the Elliott Wave pattern based on closing prices.
- **Trend Lines**: Detects trend lines in the price data.
- **Fibonacci Retracement Levels**: Calculates Fibonacci retracement levels based on high and low prices.

## Requirements

- Python 3.x
- Flask
- requests
- pandas
- numpy
- google.generativeai

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your/repository.git
    cd repository
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up API keys:
   
   - **Alpha Vantage API**: Get your API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).
   - **Google Generative AI**: Obtain your API key from [Google Generative AI](https://generativeai.dev/).

   Replace `YOUR_API_KEY` in `script.py` with your actual API keys.

## Usage

1. Start the Flask application:

    ```bash
    python script.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.
   - Enter a stock symbol in the form provided and click "Get Prediction".
   - The application will fetch historical data, perform analysis, and display the predictions and analysis results.

## Example

Here's an example of what the response JSON might look like:

```json
{
    "Simple Moving Average": {
        "2024-06-01": 123.45,
        "2024-06-02": 124.56,
        ...
    },
    "Exponential Moving Average": {
        "2024-06-01": 123.67,
        "2024-06-02": 124.78,
        ...
    },
    "Next Week Predictions": [125.34, 126.45, 127.56, 128.67, 129.78, 130.89, 132.00],
    "MACD Line": {
        "2024-06-01": 1.23,
        "2024-06-02": 1.34,
        ...
    },
    ...
}
```

## Notes

- This application uses a development server (`app.run(debug=True)`) which is not suitable for production. Use a production WSGI server like Gunicorn for deployment.
- Ensure that you comply with the terms of service of Alpha Vantage and Google Generative AI when using their APIs in production.
