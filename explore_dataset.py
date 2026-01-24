"""Explore the Tokopedia dataset."""
import pandas as pd

df = pd.read_csv('tokopedia_products_with_review.csv')

print("=" * 60)
print("TOKOPEDIA DATASET OVERVIEW")
print("=" * 60)
print(f"Total Products: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")

print("\n" + "=" * 60)
print(f"CATEGORIES ({df['category'].nunique()} unique)")
print("=" * 60)
print(df['category'].value_counts().to_string())

print("\n" + "=" * 60)
print("SAMPLE PRODUCTS BY CATEGORY")
print("=" * 60)

for cat in df['category'].unique():
    print(f"\n📁 {cat}:")
    products = df[df['category'] == cat]['name'].head(5).tolist()
    for i, p in enumerate(products, 1):
        print(f"   {i}. {p[:60]}...")

print("\n" + "=" * 60)
print("PRICE RANGE BY CATEGORY")
print("=" * 60)
price_stats = df.groupby('category')['price'].agg(['min', 'max', 'mean'])
print(price_stats.to_string())
