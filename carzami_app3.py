# Carzami Used Car Pricing Tool with Live Scraping (Streamlit Ready)

"""
Carzami Pricing Instructions:
-----------------------------

Please price this used car for Carzami. Our goal is to buy and resell used cars in Egypt with a 15% target margin (do not overprice the sell price to compensate—adjust for the sell price only). 

Interest Cost: Carzami pays ~0.0695% interest per day on the vehicle's paid amount. This interest accumulates during the holding period and reduces our profit margin.

Pricing Flow:
1. Start by checking the official brand new 2025 price in Egypt (if available) and compare it to the seller's asking price.
2. Run a pricing analysis using at least 6 comparable listings from:
   - Dubizzle
   - Hatla2ee
   - ContactCars
   (All listings must be from the past 30 days)
3. Remember: listing prices ≠ real transaction prices. Adjust accordingly.
4. Adjust comparables for trim, model year, mileage, and condition.

Carzami Listing Policy:
- Carzami does not list cars with inflated prices.
- Carzami does not negotiate with buyers. Our listings are final transaction prices.

Your Pricing Output Must Include:
- Recommended **buy price range**
- Recommended **listing price range**
- **Ideal buy price**
- **Ideal listing price**
- **Estimated sell-through time**
- **Negotiation points** to help Carzami employees convince the seller
- **Sell price strategy**: Starting price, price drop interval, expected final price

Car details:
(Provide car specs below or input them into the app interface)
"""

import requests
from bs4 import BeautifulSoup
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

# ----------------------
# 1. Listing Scraper (Live Web Scraping)
# ----------------------
class ListingScraper:
    headers = {'User-Agent': 'Mozilla/5.0'}

    def scrape_hatla2ee(self, model, year):
        # Simple example search URL - may need more exact structure
        url = f"https://eg.hatla2ee.com/en/car/{model.lower()}/used"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            cars = []
            listings = soup.select(".car-list .car")[:10]  # Limit to 10 results
            for car in listings:
                price_tag = car.select_one(".price")
                mileage_tag = car.select_one(".details span")
                year_tag = car.select_one(".title")
                if price_tag and year_tag:
                    cars.append({
                        'price': int(price_tag.text.strip().split()[0].replace(',', '')),
                        'year': year,
                        'mileage': int(mileage_tag.text.replace(',', '').split()[0]) if mileage_tag else 60000,
                        'trim': 'Unknown',
                        'condition': 'Used',
                        'date_listed': datetime.datetime.now(),
                        'source': 'Hatla2ee'
                    })
            return cars
        except:
            return []

    def scrape_dubizzle(self, model, year):
        # Placeholder: Requires advanced scraping (pagination + JS)
        return []

    def scrape_contactcars(self, model, year):
        # Placeholder: JS-heavy website
        return []

    def aggregate_listings(self, model, year):
        listings = []
        listings += self.scrape_hatla2ee(model, year)
        listings += self.scrape_dubizzle(model, year)
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
def calculate_holding_cost(price, days_held, daily_rate_percent=0.000695):
    return price * daily_rate_percent * days_held

def generate_listing_strategy(predicted_price, target_margin=0.15):
    ideal_buy_price = predicted_price * (1 - target_margin)
    initial = predicted_price * 1.0
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

    model = st.text_input("Car Model (e.g., Sportage)")
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
            predicted_price = 1_000_000

        strategy = generate_listing_strategy(predicted_price)
        holding_cost = calculate_holding_cost(predicted_price, days_listed)

        st.subheader("💰 Predicted Selling Price")
        st.metric(label="Expected Value", value=f"{round(predicted_price):,} EGP")

        st.subheader("🛒 Suggested Buy Price")
        st.write(f"{strategy['ideal_buy_price']:,} EGP")

        st.subheader("📉 Listing Strategy")
        st.write(f"Initial: {strategy['initial_listing_price']:,} EGP")
        st.write(f"Drops: {strategy['price_drops']}")
        st.write(f"Expected Final Price: {strategy['expected_final_price']:,} EGP")

        st.subheader("📦 Holding Cost")
        st.write(f"{round(holding_cost):,} EGP for {days_listed} days")

if __name__ == '__main__':
    run_dashboard()
