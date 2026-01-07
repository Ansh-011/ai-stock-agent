import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np

from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from GoogleNews import GoogleNews
from transformers import pipeline

# ==============================
# LOAD FINBERT
# ==============================
sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    device=-1
)

# ==============================
# DATA FETCH
# ==============================
def fetch_live_data(symbol):
    df = yf.download(symbol, period="7d", interval="5m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df

# ==============================
# INDICATORS
# ==============================
def add_indicators(df):
    close = df["Close"]
    df["SMA_20"] = SMAIndicator(close, 20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close, 50).sma_indicator()
    df["RSI"] = RSIIndicator(close).rsi()
    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df.dropna(inplace=True)
    return df

# ==============================
# NEWS SENTIMENT
# ==============================
def news_sentiment(query):
    googlenews = GoogleNews(lang="en", period="1d")
    googlenews.search(query)
    score = 0
    for r in googlenews.result()[:5]:
        try:
            label = sentiment_model(r["title"][:512])[0]["label"].lower()
            score += 1 if label == "positive" else -1 if label == "negative" else 0
        except:
            pass
    return score

# ==============================
# AI PREDICTION
# ==============================
def ai_predict(df, sentiment):
    df = df.copy()
    df["Future"] = df["Close"].shift(-6)
    df["Target"] = (df["Future"] > df["Close"]).astype(int)
    df.dropna(inplace=True)
    df["sentiment"] = sentiment

    X = df[["SMA_20","SMA_50","RSI","MACD","MACD_Signal","sentiment"]]
    y = df["Target"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X, y)
    prob = model.predict_proba(X.iloc[-1:])[0][1]
    trend = "Bullish 📈" if df.iloc[-1]["SMA_20"] > df.iloc[-1]["SMA_50"] else "Bearish 📉"
    return prob, trend

