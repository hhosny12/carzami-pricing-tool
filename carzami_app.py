# Carzami Used Car Pricing Tool (Streamlit Ready)

import requests
from bs4 import BeautifulSoup
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

# ----------------------
# 1. Listing Scraper (Placeholder)
# ----------------------
class ListingScraper:
    def scrape_dubizzle(self, model, year):
        return []

    def scrape_hatla2ee(self, model, year):
        return []

    def scrape_contactcars(self, model, year):
        return []

    def aggregate_listings(self, model, year):
        listings = []
        listings += self.scrape_dubizzle(model, year)
        listings += self.scrape_hatla2ee(model, year)
        listings += self.scrape_contactcars(model, year)
        return listings

# ----------------------
# 2. Preprocessing
# ----------------------
def preprocess_listings(listings):
    df = pd.DataFrame(listings)
    if 'date_listed' in df.columns:
        df['price'] = df['price'].astype(float)
        df['days_listed'] = (datetime.datetime.now() - df['date_listed']).dt.days
    else:
        df['days_listed'] = 10
    return df

# ----------------------
# 3. Model
# ----------------------
class PricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)

    def train(self, df):
        if df.empty:
            return
        X = df[['year', 'mileage', 'days_listed']]
        y = df['price']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)

    def predict_price(self, car):
        X = pd.DataFrame([car])[['year', 'mileage', 'days_listed']]
        return self.model.predict(X)[0]

# ----------------------
# 4. Strategy & Holding
# ----------------------
def calculate_holding_cost(days_held, daily_cost_rate=100):
    return days_held * daily_cost_rate

def generate_listing_strategy(predicted_price, target_margin=0.10):
    ideal_buy_price = predicted_price * (1 - target_margin)
    initial = predicted_price * 1.1
    price_drops = [initial * 0.97, initial * 0.94, initial * 0.91]
    return {
        'ideal_buy_price': round(ideal_buy_price),
        'initial_listing_price': round(initial),
        'price_drops': [round(p) for p in price_drops],
        'expected_final_price': round(price_drops[-1])
    }

# ----------------------
# 5. Streamlit App
# ----------------------
def run_dashboard():
    st.set_page_config(page_title="Carzami Pricing Tool", layout="centered")
    st.title("🚗 Carzami Pricing Tool")

    model = st.text_input("Car Model (e.g., Skoda Kodiaq)")
    year = st.number_input("Year", value=2022, min_value=2005, step=1)
    mileage = st.number_input("Mileage (KM)", value=50000, step=1000)
    days_listed = st.number_input("Estimated Holding Days", value=10, step=1)

    if st.button("Analyze Price"):
        car = {'year': year, 'mileage': mileage, 'days_listed': days_listed}

        scraper = ListingScraper()
        listings = scraper.aggregate_listings(model, year)
        df = preprocess_listings(listings)

        predictor = PricePredictor()
        if not df.empty:
            predictor.train(df)
            predicted_price = predictor.predict_price(car)
        else:
            predicted_price = 1_000_000  # Fallback if scraping is empty

        strategy = generate_listing_strategy(predicted_price)
        holding_cost = calculate_holding_cost(days_listed)

        st.subheader("💰 Predicted Selling Price")
        st.metric(label="Expected Value", value=f"{round(predicted_price):,} EGP")

        st.subheader("🛒 Suggested Buy Price")
        st.write(f"{strategy['ideal_buy_price']:,} EGP")

        st.subheader("📉 Listing Strategy")
        st.write(f"Initial: {strategy['initial_listing_price']:,} EGP")
        st.write(f"Drops: {strategy['price_drops']}")
        st.write(f"Expected Final Price: {strategy['expected_final_price']:,} EGP")

        st.subheader("📦 Holding Cost")
        st.write(f"{holding_cost:,} EGP for {days_listed} days")

if __name__ == '__main__':
    run_dashboard()
