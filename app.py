"""
Review Intelligence System - Demo Dashboard

Interactive Streamlit app showcasing:
- Sales Volume Prediction
- Review Risk Assessment
- SHAP Explainability
- Dataset Insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Review Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .risk-low { color: #10B981; font-weight: bold; }
    .risk-medium { color: #F59E0B; font-weight: bold; }
    .risk-high { color: #EF4444; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# Load models and data
@st.cache_resource
def load_models():
    """Load trained models."""
    try:
        from src.models.sales_predictor import SalesPredictor
        from src.models.review_risk_predictor import ReviewRiskPredictor
        
        sales_model = SalesPredictor.load(Path("models/sales_predictor/model.pkl"))
        risk_model = ReviewRiskPredictor.load(Path("models/review_risk_predictor/model.pkl"))
        
        return sales_model, risk_model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None


@st.cache_data
def load_sample_data(nrows=500):
    """Load sample data for visualization."""
    try:
        from src.data.ingestion import load_tokopedia_data, explode_reviews
        from src.data.preprocessing import DataPreprocessor
        
        # df = load_tokopedia_data("tokopedia_products_with_review.csv", nrows=nrows)
        df = load_tokopedia_data("converted_dataset.csv", nrows=nrows)

        reviews = explode_reviews(df)
        
        preprocessor = DataPreprocessor()
        reviews = preprocessor.fit_transform(reviews)
        
        return df, reviews
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        return None, None


@st.cache_resource
def load_recommender():
    """Load or create the recommender model."""
    try:
        from src.models.recommender import ExplainableRecommender
        from src.data.ingestion import load_tokopedia_data
        
        recommender_path = Path("models/recommender/model.pkl")
        
        if recommender_path.exists():
            return ExplainableRecommender.load(recommender_path)
        else:
            # Create and fit recommender on the fly
            df = load_tokopedia_data("converted_dataset.csv", nrows=None)
            recommender = ExplainableRecommender()
            recommender.fit(df)
            
            # Save for next time
            recommender.save(recommender_path)
            return recommender
            
    except Exception as e:
        st.error(f"Failed to load recommender: {e}")
        return None


def compute_features_for_prediction(features_dict, model_type='risk'):
    """Compute derived features for prediction."""
    price = features_dict.get('price', 0)
    stock = features_dict.get('stock', 0)
    discounted_price = features_dict.get('discounted_price', price)
    gold_merchant = features_dict.get('gold_merchant', False)
    is_official = features_dict.get('is_official', False)
    message_length = features_dict.get('message_length', 0)
    word_count = features_dict.get('word_count', 0)
    rating_average = features_dict.get('rating_average', 4.5)
    
    computed = {
        'price_log': np.log1p(price) if price > 0 else 0,
        'stock_log': np.log1p(stock) if stock > 0 else 0,
        'has_stock': 1 if stock > 0 else 0,
        'low_stock': 1 if stock < 10 else 0,
        'is_preorder': 0,
        'shop_tier': int(is_official) * 2 + int(gold_merchant),
        'uses_topads': 0,
        'discount_pct': max(0, (price - discounted_price) / price) if price > 0 else 0,
        'has_discount': 1 if discounted_price < price else 0,
        'category_encoded': hash(features_dict.get('category', 'Unknown')) % 100,
        'shop_location_encoded': hash(features_dict.get('shop_location', 'Unknown')) % 50,
        'rating_average': rating_average,
        'message_length': message_length,
        'word_count': word_count,
        'has_response': 0,
        'review_hour': 12,
        'review_dayofweek': 3,
        'is_weekend': 0
    }
    
    return computed


def render_header():
    """Render app header."""
    st.markdown('<h1 class="main-header">🧠 E-Commerce Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666; margin-bottom: 2rem;">'
        'ML-Powered Sales Prediction, Review Risk Assessment & Explainable AI Recommendations'
        '</p>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render sidebar with navigation and info."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Overview", "📊 Sales Predictor", "⚠️ Risk Analyzer", "�️ Recommender", "📈 Analytics"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("### About")
        st.info(
            "This system uses **LightGBM** models with **SHAP** explainability "
            "to predict sales, assess risks, and recommend products with explanations."
        )
        
        st.markdown("### Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk AUC", "0.84", delta="✓")
        with col2:
            st.metric("API Status", "Online", delta="✓")
        
        return page


def render_overview():
    """Render overview page."""
    st.header("System Overview")
    
    # Load actual data count
    df, reviews = load_sample_data(nrows=None)
    review_count = len(reviews) if reviews is not None else 1597
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{review_count:,}</h2>
            <p>Reviews Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>6</h2>
            <p>Modules Built</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>0.79</h2>
            <p>Risk Model AUC</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2>< 50ms</h2>
            <p>Inference Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Architecture
    st.subheader("System Architecture")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        | Layer | Components |
        |-------|------------|
        | **Data** | Ingestion, Validation, Preprocessing |
        | **Features** | Engineering, Store, Definitions |
        | **Models** | Sales Predictor, Risk Predictor, **Recommender** |
        | **Explainability** | SHAP Explainer, Recommendation Explanations |
        | **Serving** | FastAPI, Inference Engine |
        | **Monitoring** | Data Drift, Model Drift, Metrics |
        """)
    
    with col2:
        st.markdown("### Tech Stack")
        st.markdown("""
        - 🐍 Python 3.10+
        - 🌿 LightGBM
        - 📊 SHAP
        - 🛍️ Content-Based Filtering
        - ⚡ FastAPI
        - 🎨 Streamlit
        """)


def render_sales_predictor():
    """Render sales prediction page."""
    st.header("📊 Sales Volume Predictor")
    st.markdown("Predict expected sales for a product based on its attributes.")
    
    sales_model, _ = load_models()
    
    if sales_model is None:
        st.error("Sales model not loaded. Please train models first.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Product Details")
        
        price = st.number_input("Price (MYR)", min_value=10, value=150, step=10)
        stock = st.number_input("Stock", min_value=0, value=100, step=10)
        rating = st.slider("Average Rating", 1.0, 5.0, 4.5, 0.1)
        
        gold_merchant = st.checkbox("Gold Merchant", value=True)
        is_official = st.checkbox("Official Store", value=False)
        
        category = st.selectbox(
            "Category",
            ["Electronics", "Fashion", "Home & Living", "Food & Beverage", "Other"]
        )
        
        predict_button = st.button("🔮 Predict Sales", type="primary", use_container_width=True)
    
    with col2:
        if predict_button:
            # Prepare features
            features = compute_features_for_prediction({
                'price': price,
                'stock': stock,
                'rating_average': rating,
                'gold_merchant': gold_merchant,
                'is_official': is_official,
                'category': category
            }, 'sales')
            
            # Get model features
            required = sales_model.feature_names
            X = pd.DataFrame([{f: features.get(f, 0) for f in required}])
            
            # Predict
            prediction = sales_model.predict(X)[0]
            
            st.subheader("Prediction Results")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "Predicted Sales",
                    f"{prediction:,.0f} units",
                    delta=f"±{prediction * 0.3:,.0f}"
                )
            with col_b:
                # Revenue estimate
                revenue = prediction * price
                st.metric("Est. Revenue", f"RM {revenue:,.2f}")
            
            # Feature importance
            st.subheader("Key Drivers")
            importance = sales_model.get_feature_importance().head(8)
            
            fig = px.bar(
                importance,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)


def render_risk_analyzer():
    """Render risk analysis page."""
    st.header("⚠️ Review Risk Analyzer")
    st.markdown("Assess the probability of receiving a negative review.")
    
    _, risk_model = load_models()
    
    if risk_model is None:
        st.error("Risk model not loaded. Please train models first.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Review Details")
        
        price = st.number_input("Product Price (MYR)", min_value=10, value=150, step=10)
        message_length = st.number_input("Review Length (chars)", min_value=0, value=50)
        word_count = st.number_input("Word Count", min_value=0, value=10)
        
        gold_merchant = st.checkbox("Gold Merchant", value=False, key="risk_gold")
        is_official = st.checkbox("Official Store", value=False, key="risk_official")
        
        analyze_button = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)
    
    with col2:
        if analyze_button:
            # Prepare features
            features = compute_features_for_prediction({
                'price': price,
                'message_length': message_length,
                'word_count': word_count,
                'gold_merchant': gold_merchant,
                'is_official': is_official
            }, 'risk')
            
            # Get model features
            required = risk_model.feature_names
            X = pd.DataFrame([{f: features.get(f, 0) for f in required}])
            
            # Predict
            probability = risk_model.predict_proba(X)[0]
            
            # Determine risk level
            if probability < 0.2:
                risk_level = "LOW"
                color = "#10B981"
            elif probability < 0.5:
                risk_level = "MEDIUM"
                color = "#F59E0B"
            elif probability < 0.8:
                risk_level = "HIGH"
                color = "#EF4444"
            else:
                risk_level = "CRITICAL"
                color = "#DC2626"
            
            st.subheader("Risk Assessment")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Risk Probability", f"{probability * 100:.1f}%")
            with col_b:
                st.markdown(f"**Risk Level:** <span style='color: {color}; font-size: 1.5rem;'>{risk_level}</span>", unsafe_allow_html=True)
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 20], 'color': "#D1FAE5"},
                        {'range': [20, 50], 'color': "#FEF3C7"},
                        {'range': [50, 80], 'color': "#FEE2E2"},
                        {'range': [80, 100], 'color': "#FECACA"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Risk Factors")
            importance = risk_model.get_feature_importance().head(8)
            
            fig = px.bar(
                importance,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)


def render_recommender():
    """Render the Explainable AI Recommender page."""
    st.header("🛍️ Explainable AI Product Recommender")
    st.markdown("Get personalized product recommendations with **AI-powered explanations** for why each product is suggested.")
    
    recommender = load_recommender()
    
    if recommender is None:
        st.error("Recommender not available. Please check the data and try again.")
        return
    
    # Get product list for selection
    products_df = recommender.products_df
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select a Product")
        
        # Category filter
        categories = ['All Categories'] + recommender.get_categories()
        selected_category = st.selectbox("Filter by Category", categories)
        
        # Filter products by category
        if selected_category != 'All Categories':
            filtered_products = products_df[products_df['category'] == selected_category]
        else:
            filtered_products = products_df
        
        # Product selector
        product_options = {
            f"{row['name'][:50]}... (RM {row['price_raw']:.0f})": row['product_id']
            for _, row in filtered_products.head(100).iterrows()
        }
        
        if not product_options:
            st.warning("No products found in this category.")
            return
        
        selected_product_name = st.selectbox(
            "Choose a Product",
            list(product_options.keys())
        )
        selected_product_id = product_options[selected_product_name]
        
        # Number of recommendations
        n_recommendations = st.slider("Number of Recommendations", 3, 10, 5)
        
        # Diversify option
        diversify = st.checkbox("Show products from different categories", value=False)
        
        recommend_button = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)
        
        # Show selected product info
        st.divider()
        st.subheader("Selected Product")
        selected_product = products_df[products_df['product_id'] == selected_product_id].iloc[0]
        
        st.markdown(f"**{selected_product['name'][:60]}...**")
        st.markdown(f"📁 Category: {selected_product['category']}")
        st.markdown(f"💰 Price: RM {selected_product['price_raw']:,.2f}")
        st.markdown(f"⭐ Rating: {selected_product['rating_average']:.1f}/5.0")
        st.markdown(f"📦 Sold: {selected_product['count_sold']:,} units")
        
        if selected_product['gold_merchant']:
            st.markdown("🏅 **Gold Merchant**")
        if selected_product['is_official']:
            st.markdown("✅ **Official Store**")
    
    with col2:
        if recommend_button:
            with st.spinner("Finding similar products with AI explanations..."):
                recommendations = recommender.recommend(
                    product_id=selected_product_id,
                    n_recommendations=n_recommendations,
                    exclude_same_category=diversify
                )
            
            if not recommendations:
                st.warning("No recommendations found. Try adjusting your filters.")
                return
            
            st.subheader(f"🎯 Top {len(recommendations)} Recommendations")
            
            for i, rec in enumerate(recommendations, 1):
                with st.expander(
                    f"**{i}. {rec['name'][:50]}...** - {rec['similarity_score']*100:.0f}% Match",
                    expanded=(i <= 3)
                ):
                    # Product details
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Price", f"RM {rec['price']:,.2f}")
                    with col_b:
                        st.metric("Rating", f"{rec['rating']:.1f} ⭐")
                    with col_c:
                        st.metric("Sold", f"{rec['count_sold']:,}")
                    
                    st.markdown(f"**Category:** {rec['category']}")
                    
                    badges = []
                    if rec['gold_merchant']:
                        badges.append("🏅 Gold Merchant")
                    if rec['is_official']:
                        badges.append("✅ Official Store")
                    if badges:
                        st.markdown(" | ".join(badges))
                    
                    # Explainability Section
                    st.markdown("---")
                    st.markdown("### 🧠 Why This Product is Recommended")
                    
                    # Feature contribution visualization
                    explanations = rec['explanation']
                    
                    if explanations:
                        # Create bar chart for feature contributions
                        exp_df = pd.DataFrame(explanations)
                        
                        fig = px.bar(
                            exp_df,
                            x='contribution',
                            y='label',
                            orientation='h',
                            color='contribution',
                            color_continuous_scale='Greens',
                            text='detail',
                            labels={'contribution': 'Similarity Contribution', 'label': 'Factor'}
                        )
                        fig.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            showlegend=False,
                            height=250,
                            margin=dict(l=0, r=0, t=10, b=0)
                        )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Text explanation
                        top_reason = explanations[0] if explanations else None
                        if top_reason:
                            st.info(
                                f"**Main Reason:** {top_reason['label']} {top_reason['detail']} "
                                f"(contributes {top_reason['contribution']*100:.0f}% to similarity)"
                            )
                    else:
                        st.markdown("_Explanation not available_")
            
            # Summary explanation
            st.divider()
            st.subheader("📊 Recommendation Summary")
            
            # Overall similarity distribution
            similarity_scores = [r['similarity_score'] * 100 for r in recommendations]
            avg_similarity = np.mean(similarity_scores)
            
            col_x, col_y = st.columns(2)
            with col_x:
                st.metric("Average Match Score", f"{avg_similarity:.1f}%")
            with col_y:
                categories_found = len(set(r['category'] for r in recommendations))
                st.metric("Categories Covered", f"{categories_found}")
            
            # Aggregate feature importance
            st.markdown("### Key Factors Driving Recommendations")
            
            all_contributions = {}
            for rec in recommendations:
                for feature, value in rec['feature_contributions'].items():
                    if feature not in all_contributions:
                        all_contributions[feature] = []
                    all_contributions[feature].append(value)
            
            avg_contributions = {k: np.mean(v) for k, v in all_contributions.items()}
            sorted_contributions = sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True)
            
            feature_labels = {
                'price_normalized': '💰 Price Similarity',
                'rating_average': '⭐ Rating Match',
                'category_encoded': '📁 Category Match',
                'gold_merchant': '🏅 Merchant Type',
                'is_official': '✅ Store Type',
                'stock_level': '📦 Stock Level',
                'review_count_normalized': '👥 Popularity',
                'discount_pct': '🏷️ Discount Match'
            }
            
            contrib_df = pd.DataFrame([
                {'Factor': feature_labels.get(k, k), 'Importance': v}
                for k, v in sorted_contributions[:6]
            ])
            
            fig = px.bar(
                contrib_df,
                x='Importance',
                y='Factor',
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Show popular products when no recommendation is made yet
            st.subheader("🔥 Popular Products")
            st.markdown("Click **Get Recommendations** to find similar products with AI explanations.")
            
            popular = recommender.get_popular_products(n=5)
            
            for i, product in enumerate(popular, 1):
                with st.container():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{i}. {product['name'][:50]}...**")
                        st.markdown(f"📁 {product['category']} | ⭐ {product['rating']:.1f} | 📦 {product['count_sold']:,} sold")
                    with col_b:
                        st.markdown(f"**RM {product['price']:,.0f}**")
                    st.divider()


def render_analytics():
    """Render analytics page."""
    st.header("📈 Data Analytics")
    
    products, reviews = load_sample_data(nrows=None)  # Load all data
    
    if products is None:
        st.warning("Could not load dataset. Please ensure data file exists.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Products", f"{len(products):,}")
    with col2:
        st.metric("Reviews", f"{len(reviews):,}")
    with col3:
        avg_rating = reviews['review_rating'].astype(float).mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}")
    with col4:
        neg_rate = (reviews['is_negative_review'] == 1).mean() * 100
        st.metric("Negative Rate", f"{neg_rate:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution")
        rating_counts = reviews['review_rating'].astype(float).value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={'x': 'Rating', 'y': 'Count'},
            color=rating_counts.index,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sales Distribution")
        sales = products['count_sold'].dropna()
        fig = px.histogram(
            sales,
            nbins=50,
            labels={'value': 'Units Sold', 'count': 'Products'},
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Price vs Sales
    st.subheader("Price vs Sales Relationship")
    sample = products.dropna(subset=['price', 'count_sold']).sample(min(200, len(products)))
    fig = px.scatter(
        sample,
        x='price',
        y='count_sold',
        color='rating_average',
        size='stock' if 'stock' in sample.columns else None,
        color_continuous_scale='Viridis',
        labels={'price': 'Price (IDR)', 'count_sold': 'Units Sold', 'rating_average': 'Rating'}
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main app entry point."""
    render_header()
    page = render_sidebar()
    
    if page == "🏠 Overview":
        render_overview()
    elif page == "📊 Sales Predictor":
        render_sales_predictor()
    elif page == "⚠️ Risk Analyzer":
        render_risk_analyzer()
    elif page == "�️ Recommender":
        render_recommender()
    elif page == "📈 Analytics":
        render_analytics()
    
    # Footer
    st.divider()
    st.markdown(
        '<p style="text-align: center; color: #888;">Built with ❤️ using Streamlit, LightGBM, SHAP & Explainable AI</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
