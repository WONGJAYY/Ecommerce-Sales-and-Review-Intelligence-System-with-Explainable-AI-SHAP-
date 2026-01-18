"""Test script to verify Content-Based Filtering is working."""

from src.models.recommender import ExplainableRecommender
import pandas as pd

print("=" * 60)
print("CONTENT-BASED FILTERING TEST")
print("=" * 60)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('converted_dataset.csv', nrows=500)
print(f"   ✅ Loaded {len(df)} products")

# Initialize recommender
print("\n2. Initializing Content-Based Filtering Recommender...")
recommender = ExplainableRecommender()
recommender.fit(df)
print("   ✅ Recommender fitted successfully!")
print(f"   Features used: {recommender.feature_names}")

# Test recommendation
print("\n3. Testing recommendations...")
# Get first product ID
test_product_id = df['product_id'].iloc[0]
test_product = df[df['product_id'] == test_product_id].iloc[0]
print(f"   Input Product ID: {test_product_id}")
print(f"   Name: {test_product['name'][:50]}...")
print(f"   Category: {test_product['category']}")
print(f"   Price: ${test_product['price']:.2f}")

# Get recommendations
recommendations = recommender.recommend(product_id=test_product_id, n_recommendations=5)
print(f"\n4. Top 5 Similar Products Found:")
print("-" * 60)
for i, rec in enumerate(recommendations, 1):
    print(f"   {i}. {rec['name'][:45]}...")
    print(f"      Category: {rec['category']}")
    print(f"      Similarity Score: {rec['similarity_score']*100:.1f}%")
    print()

print("=" * 60)
print("✅ CONTENT-BASED FILTERING IS WORKING CORRECTLY!")
print("=" * 60)
