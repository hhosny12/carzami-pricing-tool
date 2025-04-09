# Carzami Used Car Pricing Tool – Core Logic Only (No Streamlit in Sandbox)

"""
Carzami Pricing Instructions:
-----------------------------
[... instructions remain unchanged ...]
"""

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
    return 0, 'Not found'  # Placeholder

# ----------------------
# 1. Listing Scraper (Expanded with multiple sources)
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
            listings = soup.select('article')[:10]
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
            listings = soup.select('div.usedCar_card__details__price')[:10]
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
        listings = []
        listings += self.scrape_hatla2ee(model, year)
        listings += self.scrape_dubizzle(model, year)
        listings += self.scrape_contactcars(model, year)
        return listings

# ----------------------
# 2. Preprocessing with Adjustment Logic
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
# 3. Model
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
# 4. Strategy (Includes ROI & Sell Time Estimate)
# ----------------------
def generate_listing_strategy(predicted_price, target_margin=0.15):
    ideal_buy_price = predicted_price * (1 - target_margin)
    initial = predicted_price * 1.0
    price_drops = [initial * 0.97, initial * 0.94, initial * 0.91]
    estimated_sell_time_days = 15 + 3 * (1 - target_margin) * 100  # heuristic: faster if margin is lower
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
# Test Case (CLI with Debug)
# ----------------------
if __name__ == '__main__':
    scraper = ListingScraper()
    model = "kodiaq"
    year = 2022

    print("Scraping Hatla2ee...")
    hatla2ee = scraper.scrape_hatla2ee(model, year)
    print(f"Hatla2ee listings: {len(hatla2ee)}")

    print("Scraping Dubizzle...")
    dubizzle = scraper.scrape_dubizzle(model, year)
    print(f"Dubizzle listings: {len(dubizzle)}")

    print("Scraping ContactCars...")
    contact = scraper.scrape_contactcars(model, year)
    print(f"ContactCars listings: {len(contact)}")

    listings = hatla2ee + dubizzle + contact
    print(f"✅ Total listings found: {len(listings)}")

    if not listings:
        print("❌ No listings found. Try another model or check your internet connection.")
    else:
        df = preprocess_listings(listings, year, 60000, 'Style')
        predictor = PricePredictor()
        predictor.train(df)

        if predictor.trained:
            prediction = predictor.predict_price({'year': 2022, 'mileage': 60000, 'days_listed': 10})
            print("✅ Predicted Price:", round(prediction))
            strategy = generate_listing_strategy(prediction)
            print("📈 Strategy:", strategy)
            print(f"📅 Estimated Time to Sell: {strategy['estimated_sell_time_days']} days")
            print(f"💰 Estimated ROI: {strategy['estimated_roi_percent']}%")
        else:
            print("❌ Model could not be trained due to insufficient data.")
