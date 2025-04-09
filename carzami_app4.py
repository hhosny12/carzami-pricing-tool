# Carzami Used Car Pricing Tool with Full Scraping, Adjustment Logic, and Multi-Brand Developer Price Anchor (Streamlit Ready)

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
import streamlit as st
import re

# ----------------------
# Fetch Developer Price (Hatla2ee prioritized, source reported)
# ----------------------
def fetch_developer_price(model_name, trim_name=None):
    try:
        brand_pages = {
            'hatla2ee': 'https://eg.hatla2ee.com/en/new-cars',
            'skoda': 'https://www.aboughaly.com.eg/skoda',
            'kia': 'https://www.eg.kia.com/eg/main.html',
            'hyundai': 'https://www.hyundai.com/eg/en/find-a-car',
            'toyota': 'https://www.toyota.com.eg/new-cars',
            'mg': 'https://mgmotor.com.eg/vehicles',
            'nissan': 'https://www.nissan.com.eg/vehicles/new.html'
        }

        def normalize(text):
            return re.sub(r'[^a-z0-9]', '', text.lower())

        for brand in ['hatla2ee'] + [b for b in brand_pages if b != 'hatla2ee']:
            url = brand_pages[brand]
            try:
                res = requests.get(url, timeout=10)
                soup = BeautifulSoup(res.text, 'html.parser')

                if brand == 'hatla2ee':
                    links = soup.select('a.model-link')
                    for link in links:
                        if normalize(model_name) in normalize(link.text):
                            car_url = 'https://eg.hatla2ee.com' + link['href']
                            car_page = requests.get(car_url)
                            car_soup = BeautifulSoup(car_page.text, 'html.parser')
                            prices = car_soup.select('.price strong')
                            for p in prices:
                                if trim_name and normalize(trim_name) not in normalize(p.text):
                                    continue
                                val = re.findall(r'[\d,]+', p.text)
                                if val:
                                    return int(val[0].replace(',', '')), 'Hatla2ee'

                elif brand == 'skoda':
                    listings = soup.find_all('div', class_='car-block')
                    for item in listings:
                        if normalize(model_name) in normalize(item.text):
                            if trim_name and normalize(trim_name) not in normalize(item.text):
                                continue
                            price_tag = item.find('span', class_='price')
                            if price_tag:
                                return int(price_tag.text.replace(',', '').replace('EGP', '').strip()), 'Abou Ghaly (Skoda)'

                elif brand == 'kia':
                    titles = soup.find_all('div', class_='thumb-title')
                    prices = soup.find_all('div', class_='price')
                    for title, price in zip(titles, prices):
                        if normalize(model_name) in normalize(title.text):
                            if trim_name and normalize(trim_name) not in normalize(title.text):
                                continue
                            return int(price.text.replace(',', '').replace('EGP', '').strip()), 'Kia Egypt'

                elif brand == 'hyundai':
                    models = soup.select('.model-list-item')
                    for model in models:
                        if normalize(model_name) in normalize(model.text):
                            if trim_name and normalize(trim_name) not in normalize(model.text):
                                continue
                            price = model.find('span', class_='price')
                            if price:
                                return int(price.text.replace(',', '').replace('EGP', '').strip()), 'Hyundai Egypt'

                elif brand == 'toyota':
                    models = soup.find_all('div', class_='model-card')
                    for model in models:
                        if normalize(model_name) in normalize(model.text):
                            if trim_name and normalize(trim_name) not in normalize(model.text):
                                continue
                            price_tag = model.find('div', class_='price')
                            if price_tag:
                                return int(price_tag.text.replace(',', '').replace('EGP', '').strip()), 'Toyota Egypt'

                elif brand == 'mg':
                    models = soup.find_all('div', class_='vehicle-card')
                    for model in models:
                        if normalize(model_name) in normalize(model.text):
                            if trim_name and normalize(trim_name) not in normalize(model.text):
                                continue
                            price_tag = model.find('div', class_='price')
                            if price_tag:
                                return int(price_tag.text.replace(',', '').replace('EGP', '').strip()), 'MG Egypt'

                elif brand == 'nissan':
                    models = soup.select('.vehicle-card')
                    for model in models:
                        if normalize(model_name) in normalize(model.text):
                            if trim_name and normalize(trim_name) not in normalize(model.text):
                                continue
                            price_tag = model.find('div', class_='starting-price')
                            if price_tag:
                                return int(price_tag.text.replace(',', '').replace('EGP', '').strip()), 'Nissan Egypt'

            except:
                continue

    except:
        pass
    return 0, 'Not found'

# ----------------------
# 5. Streamlit App
# ----------------------
def run_dashboard():
    st.set_page_config(page_title="Carzami Pricing Tool", layout="centered")
    st.title("🚗 Carzami Pricing Tool")

    model = st.text_input("Car Model (e.g., Kodiaq)")
    year = st.number_input("Year", value=2022, min_value=2005, step=1)
    mileage = st.number_input("Mileage (KM)", value=50000, step=1000)
    trim = st.selectbox("Trim", ["Unknown", "Base", "Comfort", "Highline", "Luxury", "Style", "Ambition"])
    days_listed = st.number_input("Estimated Holding Days", value=10, step=1)

    auto_fetch = st.checkbox("Auto-fetch Official New Price", value=True)
    dev_price, dev_source = 0, "Not provided"
    if auto_fetch and model:
        dev_price, dev_source = fetch_developer_price(model, trim)
    else:
        dev_price = st.number_input("Official Developer Price (EGP)", value=0, step=10000)

    if st.button("Analyze Price"):
        car = {'year': year, 'mileage': mileage, 'days_listed': days_listed}

        scraper = ListingScraper()
        listings = scraper.aggregate_listings(model, year)
        df = preprocess_listings(listings, year, mileage, trim)

        max_price_cap = dev_price if dev_price > 0 else None
        predictor = PricePredictor(max_price_cap)

        if not df.empty:
            predictor.train(df)
            predicted_price = predictor.predict_price(car)
        else:
            predicted_price = 1_000_000

        strategy = generate_listing_strategy(predicted_price)
        holding_cost = calculate_holding_cost(predicted_price, days_listed)

        st.subheader("💰 Predicted Selling Price")
        st.metric(label="Expected Value", value=f"{round(predicted_price):,} EGP")

        if dev_price > 0 and predicted_price >= dev_price * 0.95:
            st.warning("⚠️ This car is too close in price to the brand-new version.")
        st.write(f"New Car Price ({dev_source}): {dev_price:,} EGP")

        st.subheader("🛒 Suggested Buy Price")
        st.write(f"{strategy['ideal_buy_price']:,} EGP")

        st.subheader("📉 Listing Strategy")
        st.write(f"Initial: {strategy['initial_listing_price']:,} EGP")
        st.write(f"Drops: {strategy['price_drops']}")
        st.write(f"Expected Final Price: {strategy['expected_final_price']:,} EGP")

        st.subheader("📦 Holding Cost")
        st.write(f"{round(holding_cost):,} EGP for {days_listed} days")

        st.subheader("📊 Comparison with New Car Price")
        if dev_price > 0:
            st.write(f"New Car Price: {dev_price:,} EGP")
            st.write(f"Used Price: {round(predicted_price):,} EGP")
            st.write(f"Difference: {round(dev_price - predicted_price):,} EGP")
            st.write(f"Used is {round((predicted_price / dev_price) * 100, 1)}% of New")

if __name__ == '__main__':
    run_dashboard()
