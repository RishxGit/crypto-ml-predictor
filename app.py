import streamlit as st
import joblib
import numpy as np

model = joblib.load('crypto_xgb_model.joblib')

st.title('Crypto Price Movement Predictor')
st.write('Enter coin metrics to predict if a coin will go UP or DOWN over 1 year')

market_cap_rank = st.number_input('Market Cap Rank', min_value=1, max_value=1000, value=100)
market_cap = st.number_input('Market Cap (USD)', value=1000000000)
total_volume = st.number_input('24h Trading Volume (USD)', value=50000000)
circulating_supply = st.number_input('Circulating Supply', value=1000000)
ath_change_percentage = st.number_input('% Change from All Time High', value=-50.0)
atl_change_percentage = st.number_input('% Change from All Time Low', value=500.0)
price_change_1h = st.number_input('1h Price Change %', value=0.5)
price_change_24h = st.number_input('24h Price Change %', value=1.0)
price_change_7d = st.number_input('7d Price Change %', value=2.0)
price_change_30d = st.number_input('30d Price Change %', value=5.0)
supply_utilization = st.number_input('Supply Utilization %', value=50.0)
volume_to_marketcap = st.number_input('Volume to Market Cap Ratio', value=0.05)

if st.button('Predict Movement'):
    features = np.array([[
        market_cap_rank,
        market_cap,
        total_volume,
        circulating_supply,
        ath_change_percentage,
        atl_change_percentage,
        price_change_24h,
        price_change_1h,
        price_change_7d,
        price_change_30d,
        supply_utilization,
        volume_to_marketcap
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f'📈 BULLISH — Coin likely to go UP | Confidence: {probability:.1%}')
    else:
        st.error(f'📉 BEARISH — Coin likely to go DOWN | Confidence: {1-probability:.1%}')