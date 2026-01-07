import streamlit as st
from agent_core import fetch_live_data, add_indicators, news_sentiment, ai_predict

st.set_page_config(page_title="AI Stock Agent", layout="wide")

st.title("📈 AI Stock Intelligence Agent")

symbol = st.text_input("Enter Stock Symbol", "TCS.NS")

if st.button("Run AI Analysis"):
    df = fetch_live_data(symbol)
    df = add_indicators(df)

    sentiment = news_sentiment(symbol)
    prob, trend = ai_predict(df, sentiment)

    st.metric("Trend", trend)
    st.metric("Up Probability (next 30–60 min)", f"{round(prob*100,2)}%")

    st.line_chart(df["Close"])

st.caption("⚠️ Educational purpose only. Not financial advice.")
