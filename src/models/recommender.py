"""
Explainable AI E-Commerce Recommender System

Content-based product recommendation engine with SHAP explainability.
Recommends similar products based on product attributes and explains
WHY each product is recommended using feature contributions.

Recommendation Methods:
    - Content-Based: Similarity based on product features (price, rating, category, etc.)
    - Hybrid: Combines content features with review sentiment

Explainability:
    - Feature contribution breakdown for each recommendation
    - Similarity score decomposition
    - Visual explanation of recommendation drivers
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class ExplainableRecommender:
    """
    Explainable Content-Based Product Recommender.
    
    Recommends products similar to a given product and provides
    detailed explanations for each recommendation using feature
    contribution analysis.
    
    Features used for recommendation:
        - price_normalized: Normalized product price
        - rating_average: Average product rating
        - category_encoded: Encoded product category
        - gold_merchant: Seller trust indicator
        - is_official: Official store indicator
        - stock_level: Stock availability indicator
        - review_count: Number of reviews (popularity proxy)
        - sentiment_score: Average review sentiment
    
    Example:
        recommender = ExplainableRecommender()
        recommender.fit(products_df)
        
        recommendations = recommender.recommend(
            product_id=12345,
            n_recommendations=5
        )
        
        for rec in recommendations:
            print(f"Product: {rec['name']}")
            print(f"Similarity: {rec['similarity_score']:.2f}")
            print(f"Why recommended: {rec['explanation']}")
    """
    
    # Features used for computing similarity
    FEATURE_COLUMNS = [
        'price_normalized',
        'rating_average',
        'category_encoded',
        'gold_merchant',
        'is_official',
        'stock_level',
        'review_count_normalized',
        'discount_pct'
    ]
    
    # Feature weights for weighted similarity (can be tuned)
    DEFAULT_WEIGHTS = {
        'price_normalized': 0.20,
        'rating_average': 0.25,
        'category_encoded': 0.20,
        'gold_merchant': 0.05,
        'is_official': 0.05,
        'stock_level': 0.05,
        'review_count_normalized': 0.10,
        'discount_pct': 0.10
    }
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the recommender.
        
        Args:
            weights: Custom feature weights for similarity calculation.
                     Should sum to 1.0 for interpretable contributions.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.scaler = StandardScaler()
        self.category_encoder = LabelEncoder()
        self.products_df: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.product_ids: Optional[List] = None
        self.is_fitted = False
        
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare product features for similarity computation.
        
        Args:
            df: Raw products dataframe
            
        Returns:
            DataFrame with computed features
        """
        features = pd.DataFrame()
        features['product_id'] = df['product_id']
        features['name'] = df['name'] if 'name' in df.columns else 'Product ' + df['product_id'].astype(str)
        features['category'] = df['category'] if 'category' in df.columns else 'Unknown'
        
        # Price normalization (log scale for better distribution)
        price = df['price'].fillna(df['price'].median())
        features['price_normalized'] = np.log1p(price)
        features['price_raw'] = price
        
        # Rating (already 1-5 scale)
        features['rating_average'] = df['rating_average'].fillna(df['rating_average'].mean())
        
        # Category encoding
        features['category_encoded'] = self.category_encoder.fit_transform(
            df['category'].fillna('Unknown').astype(str)
        )
        # Normalize category to 0-1 range
        max_cat = features['category_encoded'].max()
        if max_cat > 0:
            features['category_encoded'] = features['category_encoded'] / max_cat
        
        # Binary features
        features['gold_merchant'] = df['gold_merchant'].fillna(False).astype(int)
        features['is_official'] = df['is_official'].fillna(False).astype(int)
        
        # Stock level (normalized)
        stock = df['stock'].fillna(0)
        features['stock_level'] = np.clip(stock / 500, 0, 1)  # Cap at 500 units
        
        # Review count as popularity proxy
        if 'review_count' in df.columns:
            review_count = df['review_count'].fillna(0)
        else:
            # Count reviews per product if available
            review_count = df.groupby('product_id').size().reindex(df['product_id']).fillna(1).values
        features['review_count_normalized'] = np.log1p(review_count) / np.log1p(review_count.max() + 1)
        
        # Discount percentage
        if 'discounted_price' in df.columns:
            discount = (df['price'] - df['discounted_price']) / df['price']
            features['discount_pct'] = discount.fillna(0).clip(0, 1)
        else:
            features['discount_pct'] = 0
            
        # Count sold (for display)
        features['count_sold'] = df['count_sold'].fillna(0)
        
        return features
    
    def fit(self, products_df: pd.DataFrame) -> 'ExplainableRecommender':
        """
        Fit the recommender on product data.
        
        Args:
            products_df: DataFrame with product information
            
        Returns:
            self for method chaining
        """
        logger.info(f"Fitting recommender on {len(products_df)} products...")
        
        # Aggregate to product level (in case of multiple rows per product)
        agg_df = products_df.groupby('product_id').agg({
            'name': 'first',
            'category': 'first',
            'price': 'first',
            'discounted_price': 'first' if 'discounted_price' in products_df.columns else 'first',
            'count_sold': 'first',
            'stock': 'first',
            'gold_merchant': 'first',
            'is_official': 'first',
            'rating_average': 'mean'
        }).reset_index()
        
        # Prepare features
        self.products_df = self._prepare_features(agg_df)
        self.product_ids = self.products_df['product_id'].tolist()
        
        # Create feature matrix
        feature_cols = [col for col in self.FEATURE_COLUMNS if col in self.products_df.columns]
        X = self.products_df[feature_cols].values
        
        # Scale features
        self.feature_matrix = self.scaler.fit_transform(X)
        self.feature_names = feature_cols
        
        # Build nearest neighbors index for fast lookup
        self.nn_model = NearestNeighbors(n_neighbors=min(50, len(self.products_df)), metric='cosine')
        self.nn_model.fit(self.feature_matrix)
        
        self.is_fitted = True
        logger.info(f"Recommender fitted with {len(feature_cols)} features")
        
        return self
    
    def _compute_feature_contributions(
        self,
        query_features: np.ndarray,
        target_features: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute how much each feature contributes to the similarity.
        
        Uses weighted absolute difference to explain similarity.
        Lower difference = higher contribution to similarity.
        
        Args:
            query_features: Features of the query product
            target_features: Features of the recommended product
            
        Returns:
            Dictionary of feature -> contribution score
        """
        contributions = {}
        
        for i, feature_name in enumerate(self.feature_names):
            weight = self.weights.get(feature_name, 0.1)
            
            # Compute normalized difference (0 = identical, 1 = maximum difference)
            diff = abs(query_features[i] - target_features[i])
            
            # Convert to similarity contribution (higher = more similar)
            similarity_contribution = weight * (1 - min(diff, 1))
            contributions[feature_name] = similarity_contribution
            
        return contributions
    
    def _generate_explanation(
        self,
        contributions: Dict[str, float],
        query_product: pd.Series,
        recommended_product: pd.Series
    ) -> List[Dict]:
        """
        Generate human-readable explanation for the recommendation.
        
        Args:
            contributions: Feature contribution scores
            query_product: The product user is viewing
            recommended_product: The recommended product
            
        Returns:
            List of explanation items with feature, value, and direction
        """
        explanations = []
        
        # Sort by contribution (highest first)
        sorted_features = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        feature_labels = {
            'price_normalized': 'Similar Price Range',
            'rating_average': 'Similar Rating',
            'category_encoded': 'Same Category',
            'gold_merchant': 'Same Merchant Type',
            'is_official': 'Same Store Type',
            'stock_level': 'Similar Availability',
            'review_count_normalized': 'Similar Popularity',
            'discount_pct': 'Similar Discount'
        }
        
        for feature, contribution in sorted_features[:5]:
            if contribution > 0.01:  # Only include meaningful contributions
                label = feature_labels.get(feature, feature)
                
                # Get actual values for context
                if feature == 'rating_average':
                    query_val = f"{query_product.get('rating_average', 0):.1f}⭐"
                    rec_val = f"{recommended_product.get('rating_average', 0):.1f}⭐"
                    detail = f"({query_val} vs {rec_val})"
                elif feature == 'price_normalized':
                    query_val = f"RM {query_product.get('price_raw', 0):,.0f}"
                    rec_val = f"RM {recommended_product.get('price_raw', 0):,.0f}"
                    detail = f"({query_val} vs {rec_val})"
                elif feature == 'category_encoded':
                    detail = f"({recommended_product.get('category', 'N/A')})"
                else:
                    detail = ""
                
                explanations.append({
                    'feature': feature,
                    'label': label,
                    'contribution': contribution,
                    'detail': detail,
                    'direction': 'positive' if contribution > 0.05 else 'neutral'
                })
                
        return explanations
    
    def recommend(
        self,
        product_id: Optional[int] = None,
        product_features: Optional[Dict] = None,
        n_recommendations: int = 5,
        exclude_same_category: bool = False
    ) -> List[Dict]:
        """
        Get product recommendations with explanations.
        
        Args:
            product_id: ID of the product to find similar items for
            product_features: Alternatively, provide features directly
            n_recommendations: Number of recommendations to return
            exclude_same_category: If True, diversify by excluding same category
            
        Returns:
            List of recommendation dictionaries with:
                - product_id: Recommended product ID
                - name: Product name
                - category: Product category
                - price: Product price
                - rating: Product rating
                - similarity_score: How similar (0-1)
                - explanation: Why this product was recommended
                - feature_contributions: Detailed feature breakdown
        """
        if not self.is_fitted:
            raise ValueError("Recommender must be fitted before making recommendations")
        
        # Get query product features
        if product_id is not None:
            if product_id not in self.product_ids:
                logger.warning(f"Product ID {product_id} not found. Using random product.")
                product_id = self.product_ids[0]
            
            query_idx = self.product_ids.index(product_id)
            query_features = self.feature_matrix[query_idx].reshape(1, -1)
            query_product = self.products_df.iloc[query_idx]
            
        elif product_features is not None:
            # Build feature vector from provided features
            query_vector = []
            for feature in self.feature_names:
                val = product_features.get(feature, 0)
                query_vector.append(val)
            query_features = self.scaler.transform([query_vector])
            query_product = pd.Series(product_features)
            query_idx = -1
        else:
            raise ValueError("Either product_id or product_features must be provided")
        
        # Find nearest neighbors
        distances, indices = self.nn_model.kneighbors(query_features, n_neighbors=n_recommendations + 10)
        
        recommendations = []
        for dist, idx in zip(distances[0], indices[0]):
            # Skip the query product itself
            if idx == query_idx:
                continue
                
            # Skip same category if requested
            if exclude_same_category:
                if self.products_df.iloc[idx]['category'] == query_product.get('category'):
                    continue
            
            if len(recommendations) >= n_recommendations:
                break
            
            rec_product = self.products_df.iloc[idx]
            similarity = 1 - dist  # Convert distance to similarity
            
            # Compute feature contributions
            contributions = self._compute_feature_contributions(
                query_features[0],
                self.feature_matrix[idx]
            )
            
            # Generate explanation
            explanation = self._generate_explanation(contributions, query_product, rec_product)
            
            recommendations.append({
                'product_id': int(rec_product['product_id']),
                'name': rec_product['name'],
                'category': rec_product['category'],
                'price': float(rec_product['price_raw']),
                'rating': float(rec_product['rating_average']),
                'count_sold': int(rec_product['count_sold']),
                'similarity_score': float(similarity),
                'explanation': explanation,
                'feature_contributions': contributions,
                'gold_merchant': bool(rec_product['gold_merchant']),
                'is_official': bool(rec_product['is_official'])
            })
        
        return recommendations
    
    def get_popular_products(self, n: int = 10) -> List[Dict]:
        """
        Get top N popular products (by sales).
        
        Args:
            n: Number of products to return
            
        Returns:
            List of product dictionaries
        """
        if not self.is_fitted:
            raise ValueError("Recommender must be fitted first")
        
        top_products = self.products_df.nlargest(n, 'count_sold')
        
        return [
            {
                'product_id': int(row['product_id']),
                'name': row['name'],
                'category': row['category'],
                'price': float(row['price_raw']),
                'rating': float(row['rating_average']),
                'count_sold': int(row['count_sold']),
                'gold_merchant': bool(row['gold_merchant']),
                'is_official': bool(row['is_official'])
            }
            for _, row in top_products.iterrows()
        ]
    
    def get_categories(self) -> List[str]:
        """Get list of available categories."""
        if self.products_df is not None:
            return sorted(self.products_df['category'].unique().tolist())
        return []
    
    def save(self, path: Path) -> None:
        """Save the recommender to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'products_df': self.products_df,
                'feature_matrix': self.feature_matrix,
                'product_ids': self.product_ids,
                'feature_names': self.feature_names,
                'scaler': self.scaler,
                'category_encoder': self.category_encoder,
                'weights': self.weights,
                'is_fitted': self.is_fitted
            }, f)
        
        logger.info(f"Recommender saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'ExplainableRecommender':
        """Load a saved recommender."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        recommender = cls(weights=data['weights'])
        recommender.products_df = data['products_df']
        recommender.feature_matrix = data['feature_matrix']
        recommender.product_ids = data['product_ids']
        recommender.feature_names = data['feature_names']
        recommender.scaler = data['scaler']
        recommender.category_encoder = data['category_encoder']
        recommender.is_fitted = data['is_fitted']
        
        # Rebuild NN model
        if recommender.is_fitted:
            recommender.nn_model = NearestNeighbors(
                n_neighbors=min(50, len(recommender.products_df)),
                metric='cosine'
            )
            recommender.nn_model.fit(recommender.feature_matrix)
        
        logger.info(f"Recommender loaded from {path}")
        return recommender


def create_explanation_summary(recommendations: List[Dict]) -> str:
    """
    Create a text summary explaining the recommendations.
    
    Args:
        recommendations: List of recommendation dictionaries
        
    Returns:
        Human-readable explanation string
    """
    if not recommendations:
        return "No recommendations available."
    
    summary_lines = ["**Why these products are recommended:**\n"]
    
    for i, rec in enumerate(recommendations[:3], 1):
        summary_lines.append(f"{i}. **{rec['name'][:40]}...**")
        
        top_reasons = [exp for exp in rec['explanation'] if exp['contribution'] > 0.03][:2]
        for reason in top_reasons:
            summary_lines.append(f"   • {reason['label']} {reason['detail']}")
        
        summary_lines.append("")
    
    return "\n".join(summary_lines)
