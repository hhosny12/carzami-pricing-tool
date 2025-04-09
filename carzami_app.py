# Carzami Used Car Pricing Tool – Streamlit App Version

import streamlit as st
import requests
from bs4 import BeautifulSoup
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import re

# ----------------------
# Fetch Developer Price (Hatla2ee prioritized, source reported)
# ----------------------
def fetch_developer_price(model_name, trim_name=None):
    try:
        query = model_name.lower().replace(" ", "-")
        url = f"https://eg.hatla2ee.com/en/car/{query}/new"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        price_element = soup.find('span', class_='price')
        if price_element:
            price = int(re.sub('[^0-9]', '', price_element.text))
            return price, 'Hatla2ee (new cars)'
    except Exception as e:
        pass
    return 0, 'Not found'

# ----------------------
# Listing Scraper
# ----------------------
class ListingScraper:
    headers = {'User-Agent': 'Mozilla/5.0'}

    def scrape_hatla2ee(self, model, year):
        url = f"https://eg.hatla2ee.com/en/car/{model.lower()}/used"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            cars = []
            listings = soup.select(".car-list .car")[:10]
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
        url = f"https://www.dubizzle.com.eg/en/properties/cars/{model.lower()}"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            listings = soup.select("article")[:10]
            results = []
            for item in listings:
                price = item.find('span', {'aria-label': 'Price'})
                mileage = item.find('span', string=re.compile(r'km'))
                if price:
                    results.append({
                        'price': int(re.sub('[^0-9]', '', price.text)),
                        'year': year,
                        'mileage': int(re.sub('[^0-9]', '', mileage.text)) if mileage else 60000,
                        'trim': 'Unknown',
                        'condition': 'Used',
                        'date_listed': datetime.datetime.now(),
                        'source': 'Dubizzle'
                    })
            return results
        except:
            return []

    def scrape_contactcars(self, model, year):
        url = f"https://www.contactcars.com/en/used-cars/search/{model.lower()}"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            listings = soup.select("div.usedCar_card__details__price")[:10]
            results = []
            for item in listings:
                price_text = item.text.strip()
                price = int(re.sub(r'[^0-9]', '', price_text))
                results.append({
                    'price': price,
                    'year': year,
                    'mileage': 60000,
                    'trim': 'Unknown',
                    'condition': 'Used',
                    'date_listed': datetime.datetime.now(),
                    'source': 'ContactCars'
                })
            return results
        except:
            return []

    def aggregate_listings(self, model, year):
        fallback_models = [
            model.lower(),
            model.lower().strip(),
            model.lower().replace(" ", "-"),
            model.lower().replace("-", ""),
        ]

        synonyms = {
            "sportage": ["kia sportage"],
            "elantra": ["hyundai elantra"],
            "cerato": ["kia cerato"],
            "tucson": ["hyundai tucson"],
            "accent": ["hyundai accent"]
        }
        if model.lower() in synonyms:
            fallback_models.extend(synonyms[model.lower()])

        for alt_model in fallback_models:
            hatla = self.scrape_hatla2ee(alt_model, year)
            dub = self.scrape_dubizzle(alt_model, year)
            con = self.scrape_contactcars(alt_model, year)
            combined = hatla + dub + con
            if len(combined) >= 6:
                for item in combined:
                    item['matched_model'] = alt_model
                return combined
        return []

# ----------------------
# Preprocessing
# ----------------------
def preprocess_listings(listings, target_year, target_mileage, target_trim='Unknown'):
    df = pd.DataFrame(listings)
    if df.empty:
        return df
    df['price'] = df['price'].astype(float)
    df['days_listed'] = (datetime.datetime.now() - df['date_listed']).dt.days

    def adjust_price(row):
        adjustment = 0
        year_diff = target_year - row['year']
        adjustment -= 0.02 * year_diff * row['price']
        mileage_diff = (row['mileage'] - target_mileage) / 10000
        adjustment -= 0.015 * mileage_diff * row['price']
        if target_trim.lower() in ['luxury', 'highline'] and row['trim'].lower() in ['comfort', 'base']:
            adjustment -= 0.03 * row['price']
        elif target_trim.lower() == 'comfort' and row['trim'].lower() == 'base':
            adjustment -= 0.02 * row['price']
        return row['price'] + adjustment

    df['adjusted_price'] = df.apply(adjust_price, axis=1)
    return df

# ----------------------
# Model
# ----------------------
class PricePredictor:
    def __init__(self, max_price_cap=None):
        self.model = RandomForestRegressor(n_estimators=100)
        self.max_price_cap = max_price_cap
        self.trained = False

    def train(self, df):
        if df.empty:
            return
        X = df[['year', 'mileage', 'days_listed']]
        y = df['adjusted_price']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        self.trained = True

    def predict_price(self, car):
        if not self.trained:
            raise NotFittedError("Model not trained. Run train(df) first with non-empty data.")
        X = pd.DataFrame([car])[['year', 'mileage', 'days_listed']]
        prediction = self.model.predict(X)[0]
        if self.max_price_cap:
            prediction = min(prediction, self.max_price_cap * 0.95)
        return prediction

# ----------------------
# Strategy
# ----------------------
def generate_listing_strategy(predicted_price, target_margin=0.15):
    ideal_buy_price = predicted_price * (1 - target_margin)
    initial = predicted_price * 1.0
    price_drops = [initial * 0.97, initial * 0.94, initial * 0.91]
    estimated_sell_time_days = 15 + 3 * (1 - target_margin) * 100
    estimated_roi = ((predicted_price - ideal_buy_price) / ideal_buy_price) * 100
    return {
        'ideal_buy_price': round(ideal_buy_price),
        'initial_listing_price': round(initial),
        'price_drops': [round(p) for p in price_drops],
        'expected_final_price': round(price_drops[-1]),
        'estimated_sell_time_days': int(estimated_sell_time_days),
        'estimated_roi_percent': round(estimated_roi, 2)
    }

# ----------------------
# Streamlit App
# ----------------------
def run_dashboard():
    st.set_page_config(page_title="Carzami Pricing Tool", layout="centered")
    st.title("🚗 Carzami Pricing Tool")

    model = st.text_input("Car Model (e.g., Kodiaq)")
    year = st.number_input("Model Year", min_value=2000, max_value=2025, value=2022)
    mileage = st.number_input("Mileage (km)", min_value=0, value=60000, step=1000)
    trim = st.text_input("Trim (e.g., Style, Comfort, etc.)", value="Style")

    if st.button("Analyze Price"):
        st.markdown("""
        **Pricing Prompt Being Applied:**
        _Please price this used car for Carzami. Our goal is to buy and resell used cars in Egypt with a 15% target margin (do not over price for the sell price to compensate, adjust for the sell price only). Please note that Carzami pays about 0.0695% interest for every day since the amount has been paid, keep in consideration as it eats up from the margin. Carzami Pricing Rules: Start by checking the official brand new 2025 price in Egypt (if available) and compare it with the asking price. Run a pricing analysis using at least 6 comparable listings from Dubizzle, Hatla2ee, and ContactCars, posted in the last 30 days. Note that listing prices are not real transaction prices. Adjust comparables for trim, model year, mileage, and condition. Listing prices are always not real transaction prices. Adjust accordingly. Carzami does not list with inflated prices or allow negotiation. Our listings are real transaction prices—final and non-negotiable. Output should include: Recommended buy price range. Recommended listing price range. Ideal buy price. Ideal listing price. Estimated sell-through time. Clear negotiation points for Carzami employees to convince the seller. A sell price strategy: starting price, price drop interval, expected final price._
        """)
        scraper = ListingScraper()
        listings = scraper.aggregate_listings(model, year)
        used_model = listings[0].get('matched_model', model) if listings else model
        if not listings:
            st.error("❌ No listings found. Try a different model or spelling.")
            return

        df = preprocess_listings(listings, year, mileage, trim)
        dev_price, dev_source = fetch_developer_price(model)
            predictor = PricePredictor(max_price_cap=dev_price)
        predictor.train(df)

        if predictor.trained:
            predicted_price = predictor.predict_price({'year': year, 'mileage': mileage, 'days_listed': 10})
            strategy = generate_listing_strategy(predicted_price)

            st.success(f"✅ Predicted Sell Price: {round(predicted_price):,} EGP (using model match: '{used_model}')")
            if dev_price:
                st.info(f"ℹ️ 2025 New Car Price from {dev_source}: {dev_price:,} EGP")
            st.write("---")
            st.metric("Ideal Buy Price", f"{strategy['ideal_buy_price']:,} EGP")
            st.metric("Expected Final Listing Price", f"{strategy['expected_final_price']:,} EGP")
            st.metric("Estimated Time to Sell", f"{strategy['estimated_sell_time_days']} days")
            st.metric("Estimated ROI", f"{strategy['estimated_roi_percent']}%")

            st.write("### Listing Price Strategy")
            st.write(f"Start at: {strategy['initial_listing_price']:,} EGP")
            st.write(f"Price drops: {strategy['price_drops']}")
        else:
            st.error("❌ Model training failed. Try again with a different car.")

if __name__ == "__main__":
    run_dashboard()
