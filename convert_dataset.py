"""
Dataset Converter - Converts Amazon reviews dataset to required format
"""
import pandas as pd
import numpy as np
import json
import re

def parse_price(price_str):
    """Extract price from JSON-like string."""
    if pd.isna(price_str):
        return 100.0  # default price
    try:
        # Try to parse as JSON
        if isinstance(price_str, str) and price_str.startswith('['):
            prices = json.loads(price_str.replace("'", '"'))
            if prices and isinstance(prices, list):
                return float(prices[0].get('amountMin', 100))
        # Try to extract number
        match = re.search(r'[\d.]+', str(price_str))
        if match:
            return float(match.group())
    except:
        pass
    return 100.0

def convert_amazon_to_tokopedia(input_file, output_file='converted_dataset.csv'):
    """
    Convert Amazon reviews dataset to Tokopedia format.
    """
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Create converted dataframe
    converted = pd.DataFrame()
    
    # Map columns
    converted['product_id'] = df['id'].fillna('').astype(str).apply(lambda x: hash(x) % 10000000)
    converted['name'] = df['name'].fillna('Unknown Product')
    converted['category'] = df['categories'].fillna('General').apply(lambda x: str(x).split(',')[0] if pd.notna(x) else 'General')
    
    # Price - parse from JSON or extract number
    converted['price'] = df['prices'].apply(parse_price)
    converted['discounted_price'] = converted['price'] * 0.9  # Assume 10% discount
    
    # Simulated sales based on helpful reviews
    if 'reviews.numHelpful' in df.columns:
        converted['count_sold'] = df['reviews.numHelpful'].fillna(0).astype(int) * 10
    else:
        converted['count_sold'] = np.random.randint(10, 1000, len(df))
    
    # Stock
    converted['stock'] = np.random.randint(10, 500, len(df))
    converted['preorder'] = False
    
    # Shop info (simulated)
    converted['gold_merchant'] = np.random.choice([True, False], len(df), p=[0.3, 0.7])
    converted['is_official'] = np.random.choice([True, False], len(df), p=[0.2, 0.8])
    converted['is_topads'] = False
    converted['shop_id'] = np.random.randint(1000, 9999, len(df))
    converted['shop_location'] = df.get('reviews.userProvince', pd.Series(['Unknown'] * len(df))).fillna('Unknown')
    converted['warehouse_id'] = converted['shop_id']
    converted['product_url'] = df.get('reviews.sourceURLs', '').fillna('')
    
    # Rating
    if 'reviews.rating' in df.columns:
        converted['rating_average'] = df['reviews.rating'].fillna(4.0).astype(float)
        converted['review_rating'] = df['reviews.rating'].fillna(4).astype(int).apply(lambda x: f"[{x}]")
    else:
        converted['rating_average'] = 4.0
        converted['review_rating'] = "[4]"
    
    # Reviews - wrap in list format
    converted['review_id'] = range(1, len(df) + 1)
    
    if 'reviews.text' in df.columns:
        converted['message'] = df['reviews.text'].fillna('Good product').apply(
            lambda x: f"['{str(x).replace(chr(39), chr(39)+chr(39))}']" if pd.notna(x) else "['Good product']"
        )
    else:
        converted['message'] = "['Good product']"
    
    if 'reviews.date' in df.columns:
        converted['review_time'] = df['reviews.date'].fillna('').apply(lambda x: f"['{x}']")
        converted['review_timestamp'] = df['reviews.date'].fillna('').apply(lambda x: f"['{x}']")
    else:
        converted['review_time'] = "['2024-01-01']"
        converted['review_timestamp'] = "['2024-01-01']"
    
    converted['review_response'] = "['']"
    converted['review_like'] = "[0]"
    converted['bad_rating_reason'] = "['']"
    converted['variant_name'] = "['Default']"
    
    # Save
    print(f"Saving to {output_file}...")
    converted.to_csv(output_file, index=False)
    print(f"✓ Conversion complete! {len(converted)} products saved to {output_file}")
    
    return converted

if __name__ == '__main__':
    convert_amazon_to_tokopedia('7817_1.csv', 'converted_dataset.csv')
