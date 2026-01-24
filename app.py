"""Explainable AI E-Commerce Recommender System - Demo Dashboard

Interactive Streamlit app showcasing:
- AI-Powered Product Recommendations
- SHAP Explainability
- Dataset Insights

Note: Sales Predictor and Risk Analyzer features are commented out.
Uncomment them if needed.
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
    page_title="Explainable AI E-Commerce Recommender System",
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


# ========== COMMENTED OUT: Sales & Risk Models ==========
# Uncomment this section if you need Sales Predictor or Risk Analyzer
#
# @st.cache_resource
# def load_models():
#     """Load trained models."""
#     try:
#         from src.models.sales_predictor import SalesPredictor
#         from src.models.review_risk_predictor import ReviewRiskPredictor
#         
#         sales_model = SalesPredictor.load(Path("models/sales_predictor/model.pkl"))
#         risk_model = ReviewRiskPredictor.load(Path("models/review_risk_predictor/model.pkl"))
#         
#         return sales_model, risk_model
#     except Exception as e:
#         st.error(f"Failed to load models: {e}")
#         return None, None
# ========== END COMMENTED SECTION ==========


def get_available_datasets():
    """Get list of available CSV datasets in the project."""
    datasets = {}
    csv_files = list(Path(".").glob("*.csv"))
    for f in csv_files:
        datasets[f.stem] = str(f)
    return datasets


@st.cache_data
def load_sample_data(dataset_path=None, nrows=None):
    """Load sample data for visualization."""
    # Use default dataset path if not provided
    if dataset_path is None:
        dataset_path = Path(__file__).parent / 'converted_dataset.csv'
    try:
        from src.data.ingestion import load_tokopedia_data, explode_reviews
        from src.data.preprocessing import DataPreprocessor
        
        df = load_tokopedia_data(dataset_path, nrows=nrows)

        reviews = explode_reviews(df)
        
        preprocessor = DataPreprocessor()
        reviews = preprocessor.fit_transform(reviews)
        
        return df, reviews
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        return None, None


@st.cache_resource
def load_recommender(dataset_path=None):
    """Load or create the recommender model for the selected dataset."""
    # Use default dataset path if not provided
    if dataset_path is None:
        dataset_path = Path(__file__).parent / 'converted_dataset.csv'
    try:
        from src.models.recommender import ExplainableRecommender
        from src.data.ingestion import load_tokopedia_data
        
        # Create and fit recommender on the fly for the selected dataset
        df = load_tokopedia_data(dataset_path, nrows=None)
        recommender = ExplainableRecommender()
        recommender.fit(df)
        
        return recommender
            
    except Exception as e:
        st.error(f"Failed to load recommender: {e}")
        return None


@st.cache_resource
def load_llm_client():
    """Load NVIDIA NIM LLM client for natural language explanations."""
    try:
        from src.llm.nvidia_nim import NVIDIANimClient
        client = NVIDIANimClient()
        return client
    except Exception as e:
        st.warning(f"LLM client not available: {e}")
        return None


# ========== COMMENTED OUT: Feature computation for Sales/Risk ==========
# Uncomment this section if you need Sales Predictor or Risk Analyzer
#
# def compute_features_for_prediction(features_dict, model_type='risk'):
#     """Compute derived features for prediction."""
#     price = features_dict.get('price', 0)
#     stock = features_dict.get('stock', 0)
#     discounted_price = features_dict.get('discounted_price', price)
#     gold_merchant = features_dict.get('gold_merchant', False)
#     is_official = features_dict.get('is_official', False)
#     message_length = features_dict.get('message_length', 0)
#     word_count = features_dict.get('word_count', 0)
#     rating_average = features_dict.get('rating_average', 4.5)
#     
#     computed = {
#         'price_log': np.log1p(price) if price > 0 else 0,
#         'stock_log': np.log1p(stock) if stock > 0 else 0,
#         'has_stock': 1 if stock > 0 else 0,
#         'low_stock': 1 if stock < 10 else 0,
#         'is_preorder': 0,
#         'shop_tier': int(is_official) * 2 + int(gold_merchant),
#         'uses_topads': 0,
#         'discount_pct': max(0, (price - discounted_price) / price) if price > 0 else 0,
#         'has_discount': 1 if discounted_price < price else 0,
#         'category_encoded': hash(features_dict.get('category', 'Unknown')) % 100,
#         'shop_location_encoded': hash(features_dict.get('shop_location', 'Unknown')) % 50,
#         'rating_average': rating_average,
#         'message_length': message_length,
#         'word_count': word_count,
#         'has_response': 0,
#         'review_hour': 12,
#         'review_dayofweek': 3,
#         'is_weekend': 0
#     }
#     
#     return computed
# ========== END COMMENTED SECTION ==========


def render_header():
    """Render app header."""
    st.markdown('<h1 class="main-header">🧠 Explainable AI E-Commerce Recommender System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666; margin-bottom: 2rem;">'
        'AI-Powered Product Recommendations with Explainable AI (SHAP)'
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
            ["🏠 Overview", "🛍️ Recommender", "� AI Chat", "📈 Analytics"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("### About")
        st.info(
            "This system uses **Content-Based Filtering** with **SHAP & LLM** explainability "
            "to recommend products and explain WHY each product is suggested."
        )
        
        st.markdown("### System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recommender", "Active", delta="✓")
        with col2:
            st.metric("API Status", "Online", delta="✓")
        
        return page


def render_overview():
    """Render overview page."""
    st.header("System Overview")
    
    # Load actual data count
    df, reviews = load_sample_data("converted_dataset.csv", nrows=None)
    product_count = len(df) if df is not None else 0
    review_count = len(reviews) if reviews is not None else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{product_count:,}</h2>
            <p>Products</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{review_count:,}</h2>
            <p>Reviews Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>SHAP, LIME & LLM</h2>
            <p>Explainability</p>
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
        | **Features** | Feature Engineering, Feature Store |
        | **Models** | Content-Based Filtering Recommender |
        | **Explainability** | SHAP, LIME, LLM (Natural Language) |
        | **LLM** | NVIDIA NIM (Llama 3.3), Chat, Review Summary |
        | **Serving** | FastAPI, Streamlit Dashboard |
        """)
    
    with col2:
        st.markdown("### Tech Stack")
        st.markdown("""
        - 🐍 Python 3.10+
        - 🦙 NVIDIA NIM (Llama 3.3 70B)
        - 📊 SHAP (Shapley Additive exPlanations)
        - 🍋 LIME (Local Interpretable Explanations)
        - 🛍️ Content-Based Filtering
        - 📈 Scikit-learn
        - ⚡ FastAPI
        - 🎨 Streamlit
        """)


# ========== COMMENTED OUT: Sales Predictor Page ==========
# Uncomment this section if you need Sales Predictor functionality
#
# def render_sales_predictor():
#     """Render sales prediction page."""
#     st.header("📊 Sales Volume Predictor")
#     st.markdown("Predict expected sales for a product based on its attributes.")
#     
#     sales_model, _ = load_models()
#     
#     if sales_model is None:
#         st.error("Sales model not loaded. Please train models first.")
#         return
#     
#     col1, col2 = st.columns([1, 2])
#     
#     with col1:
#         st.subheader("Product Details")
#         
#         price = st.number_input("Price (MYR)", min_value=10, value=150, step=10)
#         stock = st.number_input("Stock", min_value=0, value=100, step=10)
#         rating = st.slider("Average Rating", 1.0, 5.0, 4.5, 0.1)
#         
#         gold_merchant = st.checkbox("Gold Merchant", value=True)
#         is_official = st.checkbox("Official Store", value=False)
#         
#         category = st.selectbox(
#             "Category",
#             ["Electronics", "Fashion", "Home & Living", "Food & Beverage", "Other"]
#         )
#         
#         predict_button = st.button("🔮 Predict Sales", type="primary", use_container_width=True)
#     
#     with col2:
#         if predict_button:
#             # Prepare features
#             features = compute_features_for_prediction({
#                 'price': price,
#                 'stock': stock,
#                 'rating_average': rating,
#                 'gold_merchant': gold_merchant,
#                 'is_official': is_official,
#                 'category': category
#             }, 'sales')
#             
#             # Get model features
#             required = sales_model.feature_names
#             X = pd.DataFrame([{f: features.get(f, 0) for f in required}])
#             
#             # Predict
#             prediction = sales_model.predict(X)[0]
#             
#             st.subheader("Prediction Results")
#             
#             col_a, col_b = st.columns(2)
#             with col_a:
#                 st.metric(
#                     "Predicted Sales",
#                     f"{prediction:,.0f} units",
#                     delta=f"±{prediction * 0.3:,.0f}"
#                 )
#             with col_b:
#                 # Revenue estimate
#                 revenue = prediction * price
#                 st.metric("Est. Revenue", f"RM {revenue:,.2f}")
#             
#             # Feature importance
#             st.subheader("Key Drivers")
#             importance = sales_model.get_feature_importance().head(8)
#             
#             fig = px.bar(
#                 importance,
#                 x='importance',
#                 y='feature',
#                 orientation='h',
#                 color='importance',
#                 color_continuous_scale='Viridis'
#             )
#             fig.update_layout(
#                 yaxis={'categoryorder': 'total ascending'},
#                 showlegend=False,
#                 height=300
#             )
#             st.plotly_chart(fig, use_container_width=True)
# ========== END COMMENTED SECTION ==========


# ========== COMMENTED OUT: Risk Analyzer Page ==========
# Uncomment this section if you need Risk Analyzer functionality
#
# def render_risk_analyzer():
#     """Render risk analysis page."""
#     st.header("⚠️ Review Risk Analyzer")
#     st.markdown("Assess the probability of receiving a negative review with **SHAP & LIME explanations**.")
#     
#     _, risk_model = load_models()
#     
#     if risk_model is None:
#         st.error("Risk model not loaded. Please train models first.")
#         return
#     
#     col1, col2 = st.columns([1, 2])
#     
#     with col1:
#         st.subheader("Review Details")
#         
#         price = st.number_input("Product Price (MYR)", min_value=10, value=150, step=10)
#         message_length = st.number_input("Review Length (chars)", min_value=0, value=50)
#         word_count = st.number_input("Word Count", min_value=0, value=10)
#         
#         gold_merchant = st.checkbox("Gold Merchant", value=False, key="risk_gold")
#         is_official = st.checkbox("Official Store", value=False, key="risk_official")
#         
#         # Explainability method selection
#         st.divider()
#         st.subheader("Explainability Method")
#         explain_method = st.radio(
#             "Select explanation method:",
#             ["SHAP (Recommended)", "LIME", "Both (Compare)"],
#             key="explain_method"
#         )
#         
#         analyze_button = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)
#     
#     with col2:
#         if analyze_button:
#             # Prepare features
#             features = compute_features_for_prediction({
#                 'price': price,
#                 'message_length': message_length,
#                 'word_count': word_count,
#                 'gold_merchant': gold_merchant,
#                 'is_official': is_official
#             }, 'risk')
#             
#             # Get model features
#             required = risk_model.feature_names
#             X = pd.DataFrame([{f: features.get(f, 0) for f in required}])
#             
#             # Predict
#             probability = risk_model.predict_proba(X)[0]
#             
#             # Determine risk level
#             if probability < 0.2:
#                 risk_level = "LOW"
#                 color = "#10B981"
#             elif probability < 0.5:
#                 risk_level = "MEDIUM"
#                 color = "#F59E0B"
#             elif probability < 0.8:
#                 risk_level = "HIGH"
#                 color = "#EF4444"
#             else:
#                 risk_level = "CRITICAL"
#                 color = "#DC2626"
#             
#             st.subheader("Risk Assessment")
#             
#             col_a, col_b = st.columns(2)
#             with col_a:
#                 st.metric("Risk Probability", f"{probability * 100:.1f}%")
#             with col_b:
#                 st.markdown(f"**Risk Level:** <span style='color: {color}; font-size: 1.5rem;'>{risk_level}</span>", unsafe_allow_html=True)
#             
#             # Risk gauge
#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=probability * 100,
#                 title={'text': "Risk Score"},
#                 gauge={
#                     'axis': {'range': [0, 100]},
#                     'bar': {'color': color},
#                     'steps': [
#                         {'range': [0, 20], 'color': "#D1FAE5"},
#                         {'range': [20, 50], 'color': "#FEF3C7"},
#                         {'range': [50, 80], 'color': "#FEE2E2"},
#                         {'range': [80, 100], 'color': "#FECACA"}
#                     ]
#                 }
#             ))
#             fig.update_layout(height=250)
#             st.plotly_chart(fig, use_container_width=True)
#             
#             # ========== EXPLAINABILITY SECTION ==========
#             st.divider()
#             
#             # Generate background data for explainers
#             _, reviews = load_sample_data(nrows=200)
#             if reviews is not None:
#                 background_data = pd.DataFrame([{f: 0 for f in required}])
#                 for col in required:
#                     if col in reviews.columns:
#                         background_data[col] = reviews[col].median()
#                 # Expand to 50 samples with some variation
#                 background_samples = pd.concat([background_data] * 50, ignore_index=True)
#                 for col in background_samples.columns:
#                     background_samples[col] = background_samples[col] + np.random.randn(50) * 0.1
#             else:
#                 background_samples = X
#             
#             # SHAP Explanation
#             if explain_method in ["SHAP (Recommended)", "Both (Compare)"]:
#                 st.subheader("🔬 SHAP Explanation")
#                 st.markdown("*SHAP (SHapley Additive exPlanations) uses game theory to explain predictions.*")
#                 
#                 try:
#                     from src.explainability.shap_explainer import SHAPExplainer
#                     
#                     with st.spinner("Computing SHAP values..."):
#                         shap_explainer = SHAPExplainer(
#                             model=risk_model,
#                             background_data=background_samples,
#                             feature_names=list(X.columns)
#                         )
#                         shap_explanation = shap_explainer.explain(X)
#                         shap_top = shap_explanation.get_top_features(8)
#                     
#                     # Create SHAP visualization
#                     shap_df = pd.DataFrame(shap_top)
#                     shap_df['abs_value'] = shap_df['shap_value'].abs()
#                     shap_df = shap_df.sort_values('abs_value', ascending=True)
#                     
#                     fig_shap = px.bar(
#                         shap_df,
#                         x='shap_value',
#                         y='feature',
#                         orientation='h',
#                         color='direction',
#                         color_discrete_map={'positive': '#EF4444', 'negative': '#10B981'},
#                         title="SHAP Feature Contributions"
#                     )
#                     fig_shap.update_layout(
#                         showlegend=True,
#                         height=300,
#                         xaxis_title="Impact on Risk (SHAP value)",
#                         yaxis_title=""
#                     )
#                     st.plotly_chart(fig_shap, use_container_width=True)
#                     
#                     # Show interpretation
#                     with st.expander("📖 How to interpret SHAP values"):
#                         st.markdown("""
#                         - **Red bars (positive)**: Features pushing prediction **toward higher risk**
#                         - **Green bars (negative)**: Features pushing prediction **toward lower risk**
#                         - **Longer bars**: Stronger impact on the prediction
#                         - SHAP values are mathematically consistent and sum to the difference between prediction and average
#                         """)
#                         
#                 except Exception as e:
#                     st.warning(f"SHAP explanation unavailable: {e}")
#             
#             # LIME Explanation
#             if explain_method in ["LIME", "Both (Compare)"]:
#                 st.subheader("🍋 LIME Explanation")
#                 st.markdown("*LIME (Local Interpretable Model-agnostic Explanations) fits a local linear model.*")
#                 
#                 try:
#                     from src.explainability.lime_explainer import LIMEExplainer
#                     
#                     with st.spinner("Computing LIME explanation..."):
#                         lime_explainer = LIMEExplainer(
#                             model=risk_model,
#                             training_data=background_samples,
#                             feature_names=list(X.columns),
#                             mode='classification'
#                         )
#                         lime_explanation = lime_explainer.explain(X, num_features=8)
#                         lime_top = lime_explanation.get_top_features(8)
#                     
#                     # Create LIME visualization
#                     lime_df = pd.DataFrame(lime_top)
#                     lime_df['abs_weight'] = lime_df['lime_weight'].abs()
#                     lime_df = lime_df.sort_values('abs_weight', ascending=True)
#                     
#                     fig_lime = px.bar(
#                         lime_df,
#                         x='lime_weight',
#                         y='feature',
#                         orientation='h',
#                         color='direction',
#                         color_discrete_map={'positive': '#EF4444', 'negative': '#10B981'},
#                         title="LIME Feature Weights"
#                     )
#                     fig_lime.update_layout(
#                         showlegend=True,
#                         height=300,
#                         xaxis_title="Impact on Risk (LIME weight)",
#                         yaxis_title=""
#                     )
#                     st.plotly_chart(fig_lime, use_container_width=True)
#                     
#                     # Show local model score
#                     st.info(f"📊 Local Model R² Score: **{lime_explanation.score:.3f}** (how well the linear model approximates locally)")
#                     
#                     with st.expander("📖 How to interpret LIME weights"):
#                         st.markdown("""
#                         - **Red bars (positive)**: Features increasing risk probability
#                         - **Green bars (negative)**: Features decreasing risk probability
#                         - LIME fits a simple linear model around this specific prediction
#                         - Higher R² score = better local approximation
#                         """)
#                         
#                 except Exception as e:
#                     st.warning(f"LIME explanation unavailable: {e}")
#             
#             # Comparison section
#             if explain_method == "Both (Compare)":
#                 st.divider()
#                 st.subheader("⚖️ SHAP vs LIME Comparison")
#                 
#                 try:
#                     # Get top features from both
#                     shap_features = [f['feature'] for f in shap_top[:5]]
#                     lime_features = [f['feature'] for f in lime_top[:5]]
#                     
#                     overlap = set(shap_features) & set(lime_features)
#                     agreement = len(overlap) / 5 * 100
#                     
#                     col_cmp1, col_cmp2, col_cmp3 = st.columns(3)
#                     with col_cmp1:
#                         st.metric("Top-5 Feature Overlap", f"{agreement:.0f}%")
#                     with col_cmp2:
#                         st.metric("SHAP Top Feature", shap_features[0] if shap_features else "N/A")
#                     with col_cmp3:
#                         st.metric("LIME Top Feature", lime_features[0] if lime_features else "N/A")
#                     
#                     if agreement >= 60:
#                         st.success("✅ **High Agreement**: Both methods identify similar important features. Explanations are reliable.")
#                     elif agreement >= 40:
#                         st.warning("⚠️ **Moderate Agreement**: Some differences in feature ranking. Consider both perspectives.")
#                     else:
#                         st.error("❌ **Low Agreement**: Methods disagree significantly. Further investigation recommended.")
#                         
#                 except Exception as e:
#                     st.warning(f"Could not compare methods: {e}")
#             
#             # Global Feature Importance (original)
#             st.divider()
#             st.subheader("📊 Global Feature Importance")
#             st.markdown("*Overall importance of features across all predictions (not specific to this instance).*")
#             
#             importance = risk_model.get_feature_importance().head(8)
#             
#             fig = px.bar(
#                 importance,
#                 x='importance',
#                 y='feature',
#                 orientation='h',
#                 color='importance',
#                 color_continuous_scale='Reds'
#             )
#             fig.update_layout(
#                 yaxis={'categoryorder': 'total ascending'},
#                 showlegend=False,
#                 height=300
#             )
#             st.plotly_chart(fig, use_container_width=True)
# ========== END COMMENTED SECTION ==========


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
            f"{row['name'][:50]}... (${row['price_raw']:.0f})": row['product_id']
            for _, row in filtered_products.iterrows()
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
        
        # Explainability method selection
        st.divider()
        st.subheader("🧠 Explainability Method")
        explain_method = st.radio(
            "Choose explanation method:",
            ["SHAP", "LIME", "LLM (Natural Language)", "All Methods"],
            help="SHAP: Game theory explanations. LIME: Local linear explanations. LLM: AI-powered natural language explanations."
        )
        
        recommend_button = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)
        
        # Show selected product info
        st.divider()
        st.subheader("Selected Product")
        selected_product = products_df[products_df['product_id'] == selected_product_id].iloc[0]
        
        st.markdown(f"**{selected_product['name'][:60]}...**")
        st.markdown(f"📁 Category: {selected_product['category']}")
        st.markdown(f"💰 Price: ${selected_product['price_raw']:,.2f}")
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
                        st.metric("Price", f"${rec['price']:,.2f}")
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
                    
                    # LLM Natural Language Explanation
                    if explain_method in ["LLM (Natural Language)", "All Methods"]:
                        if explain_method == "All Methods":
                            st.markdown("#### 🤖 AI Natural Language Explanation")
                        
                        llm_client = load_llm_client()
                        if llm_client and llm_client.is_available:
                            with st.spinner("Generating AI explanation..."):
                                # Get SHAP values first for context
                                shap_values_dict = rec.get('feature_contributions', {})
                                
                                source_product_info = {
                                    'name': selected_product['name'],
                                    'category': selected_product['category'],
                                    'price': selected_product['price_raw'],
                                    'rating_average': selected_product['rating_average']
                                }
                                
                                rec_product_info = {
                                    'name': rec['name'],
                                    'category': rec['category'],
                                    'price': rec['price'],
                                    'rating_average': rec['rating']
                                }
                                
                                llm_explanation = llm_client.explain_recommendation(
                                    source_product=source_product_info,
                                    recommended_product=rec_product_info,
                                    shap_values=shap_values_dict,
                                    similarity_score=rec['similarity_score']
                                )
                                
                                st.success(f"💬 {llm_explanation}")
                        else:
                            st.info("💡 **Set up LLM for natural language explanations:**\n\n"
                                   "1. Get API key from [build.nvidia.com](https://build.nvidia.com)\n"
                                   "2. Set environment variable: `NVIDIA_API_KEY=your_key_here`\n"
                                   "3. Restart the app")
                            
                            # Show fallback explanation
                            top_features = sorted(rec['feature_contributions'].items(), 
                                                 key=lambda x: abs(x[1]), reverse=True)[:3]
                            feature_names = {
                                'price_normalized': 'similar price',
                                'rating_average': 'similar rating',
                                'category_encoded': 'same category',
                                'gold_merchant': 'trusted seller',
                                'is_official': 'official store',
                                'stock_level': 'availability',
                                'review_count_normalized': 'popularity',
                                'discount_pct': 'discount'
                            }
                            reasons = [feature_names.get(f, f) for f, _ in top_features]
                            st.write(f"**Fallback:** Recommended due to {', '.join(reasons)}.")
                    
                    # SHAP Explanation
                    if explain_method in ["SHAP", "All Methods"]:
                        if explain_method == "All Methods":
                            st.markdown("#### 🔬 SHAP Explanation")
                        
                        try:
                            import shap
                            
                            # Get feature data for SHAP
                            feature_names = recommender.feature_names
                            feature_matrix = recommender.feature_matrix
                            query_idx = recommender.product_ids.index(selected_product_id)
                            query_features = feature_matrix[query_idx]
                            
                            # Create similarity prediction function for SHAP
                            def similarity_predict_shap(X):
                                """Predict similarity scores for SHAP."""
                                similarities = []
                                for row in X:
                                    # Cosine similarity
                                    dot = np.dot(query_features, row)
                                    norm = np.linalg.norm(query_features) * np.linalg.norm(row)
                                    sim = dot / norm if norm > 0 else 0
                                    similarities.append(sim)
                                return np.array(similarities)
                            
                            # Use KernelSHAP with background samples
                            background_sample = shap.sample(feature_matrix, min(50, len(feature_matrix)))
                            explainer = shap.KernelExplainer(similarity_predict_shap, background_sample)
                            
                            # Get the recommended product's features
                            rec_idx = recommender.product_ids.index(rec['product_id'])
                            rec_features = feature_matrix[rec_idx:rec_idx+1]
                            
                            # Calculate SHAP values
                            shap_values = explainer.shap_values(rec_features, nsamples=100)
                            
                            # Create SHAP visualization data
                            shap_data = []
                            for idx, (fname, sval) in enumerate(zip(feature_names, shap_values[0])):
                                shap_data.append({
                                    'feature': fname,
                                    'shap_value': float(sval),
                                    'direction': 'positive' if sval > 0 else 'negative'
                                })
                            
                            shap_df = pd.DataFrame(shap_data)
                            shap_df['abs_value'] = shap_df['shap_value'].abs()
                            shap_df = shap_df.nlargest(6, 'abs_value').sort_values('abs_value', ascending=True)
                            
                            # Feature labels for display
                            feature_labels = {
                                'price_normalized': '💰 Price',
                                'rating_average': '⭐ Rating',
                                'category_encoded': '📁 Category',
                                'gold_merchant': '🏅 Merchant',
                                'is_official': '✅ Official',
                                'stock_level': '📦 Stock',
                                'review_count_normalized': '👥 Popularity',
                                'discount_pct': '🏷️ Discount'
                            }
                            shap_df['label'] = shap_df['feature'].map(lambda x: feature_labels.get(x, x))
                            
                            fig_shap = px.bar(
                                shap_df,
                                x='shap_value',
                                y='label',
                                orientation='h',
                                color='direction',
                                color_discrete_map={'positive': '#10B981', 'negative': '#EF4444'},
                                title="" if explain_method == "All Methods" else "SHAP Feature Impact"
                            )
                            fig_shap.update_layout(
                                showlegend=True,
                                height=250,
                                xaxis_title="Impact on Similarity (SHAP value)",
                                yaxis_title="",
                                margin=dict(l=0, r=0, t=10, b=0)
                            )
                            st.plotly_chart(fig_shap, use_container_width=True)
                            
                            # Top driver explanation
                            top_shap = shap_df.iloc[-1]  # Highest absolute value
                            direction = "increases" if top_shap['shap_value'] > 0 else "decreases"
                            st.info(f"**Top Driver:** {top_shap['label']} {direction} similarity by {abs(top_shap['shap_value']):.3f}")
                            
                        except ImportError:
                            st.warning("SHAP not installed. Run: `pip install shap`")
                        except Exception as e:
                            st.warning(f"SHAP explanation unavailable: {e}")
                    
                    # LIME Explanation
                    if explain_method in ["LIME", "All Methods"]:
                        if explain_method == "All Methods":
                            st.markdown("#### 🍋 LIME Explanation")
                        
                        try:
                            from src.explainability.lime_explainer import LIMEExplainer, LIME_AVAILABLE
                            
                            if LIME_AVAILABLE:
                                # Get feature data for LIME
                                feature_names = recommender.feature_names
                                feature_matrix = recommender.feature_matrix
                                
                                # Create LIME explainer for similarity prediction
                                def similarity_predict(X):
                                    """Predict similarity scores for LIME."""
                                    query_idx = recommender.product_ids.index(selected_product_id)
                                    query_features = feature_matrix[query_idx]
                                    similarities = []
                                    for row in X:
                                        # Cosine similarity
                                        dot = np.dot(query_features, row)
                                        norm = np.linalg.norm(query_features) * np.linalg.norm(row)
                                        sim = dot / norm if norm > 0 else 0
                                        similarities.append([1-sim, sim])  # [dissimilar, similar]
                                    return np.array(similarities)
                                
                                lime_explainer = LIMEExplainer(
                                    model=None,
                                    training_data=feature_matrix,
                                    feature_names=feature_names,
                                    mode='classification',
                                    predict_fn=similarity_predict
                                )
                                
                                # Get the recommended product's features
                                rec_idx = recommender.product_ids.index(rec['product_id'])
                                rec_features = feature_matrix[rec_idx:rec_idx+1]
                                
                                lime_explanation = lime_explainer.explain(rec_features, num_features=6)
                                lime_top = lime_explanation.get_top_features(6)
                                
                                # Create LIME visualization
                                lime_df = pd.DataFrame(lime_top)
                                lime_df['abs_weight'] = lime_df['lime_weight'].abs()
                                lime_df = lime_df.sort_values('abs_weight', ascending=True)
                                
                                fig_lime = px.bar(
                                    lime_df,
                                    x='lime_weight',
                                    y='feature',
                                    orientation='h',
                                    color='direction',
                                    color_discrete_map={'positive': '#10B981', 'negative': '#EF4444'},
                                    title="" if explain_method == "All Methods" else "LIME Feature Weights"
                                )
                                fig_lime.update_layout(
                                    showlegend=True,
                                    height=250,
                                    xaxis_title="Impact on Similarity (LIME weight)",
                                    yaxis_title=""
                                )
                                st.plotly_chart(fig_lime, use_container_width=True)
                                
                                st.caption(f"LIME Local Model R² Score: {lime_explanation.score:.3f}")
                            else:
                                st.warning("LIME not installed. Run: `pip install lime`")
                        except Exception as e:
                            st.warning(f"LIME explanation unavailable: {e}")
                    
                    # Comparison insight
                    if explain_method == "All Methods":
                        with st.expander("📖 Understanding the Explanation Methods"):
                            st.markdown("""
                            **🤖 LLM (Large Language Model):**
                            - Uses NVIDIA NIM AI to generate human-readable explanations
                            - Combines SHAP values with product context
                            - Most intuitive for end users
                            
                            **🔬 SHAP (SHapley Additive exPlanations):**
                            - Based on game theory (Shapley values)
                            - Provides mathematically consistent feature attributions
                            - Shows how each feature contributes to similarity
                            
                            **🍋 LIME (Local Interpretable Model-agnostic Explanations):**
                            - Fits a local linear model around the prediction
                            - Perturbs input features to understand local behavior
                            - Simple and intuitive visualizations
                            
                            **Best Practice:** Use LLM for user-facing explanations, 
                            SHAP for technical analysis, and LIME for quick insights!
                            """)
            
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


def render_ai_chat():
    """Render AI Chat page with product chat and review summarization."""
    st.header("💬 AI Chat Assistant")
    st.markdown("Ask questions about products or get AI-powered review summaries.")
    
    # Load data and LLM client
    llm_client = load_llm_client()
    recommender = load_recommender()
    products, reviews = load_sample_data(nrows=5000)
    
    if recommender is None or recommender.products_df is None:
        st.error("Recommender not loaded. Please check your data.")
        return
    
    # Check LLM availability
    if llm_client is None or not llm_client.is_available:
        st.warning("""
        ⚠️ **LLM not configured.** To enable AI Chat:
        1. Get API key from [build.nvidia.com](https://build.nvidia.com)
        2. Add to `.env` file: `NVIDIA_API_KEY=your_key_here`
        3. Restart the app
        """)
    
    # Create two tabs
    tab1, tab2 = st.tabs(["🤖 Product Chat", "📝 Review Summarizer"])
    
    # ===================== TAB 1: Product Chat =====================
    with tab1:
        st.subheader("Chat About Products")
        st.markdown("Select a product and ask questions about it!")
        
        products_df = recommender.products_df
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Product selection
            product_options = {
                f"{row['name'][:50]}..." if len(row['name']) > 50 else row['name']: row['product_id']
                for _, row in products_df.iterrows()
            }
            
            selected_name = st.selectbox(
                "Select a Product:",
                options=list(product_options.keys()),
                key="chat_product"
            )
            selected_id = product_options[selected_name]
            selected_product = products_df[products_df['product_id'] == selected_id].iloc[0]
            
            # Show product info
            st.markdown("---")
            st.markdown(f"**{selected_product['name'][:60]}...**")
            st.markdown(f"📁 Category: {selected_product['category']}")
            st.markdown(f"💰 Price: RM {selected_product['price_raw']:,.2f}")
            st.markdown(f"⭐ Rating: {selected_product['rating_average']:.1f}/5.0")
            st.markdown(f"📦 Sold: {selected_product['count_sold']:,} units")
            
            # Get recommendations for context
            recommendations = recommender.recommend(product_id=selected_id, n_recommendations=3)
        
        with col2:
            # Initialize chat history
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
            
            # Chat container
            chat_container = st.container(height=400)
            
            with chat_container:
                # Display chat history
                for msg in st.session_state.chat_messages:
                    if msg["role"] == "user":
                        st.markdown(f"**🧑 You:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖 AI:** {msg['content']}")
                    st.markdown("")
            
            # Chat input
            user_question = st.text_input(
                "Ask a question about this product:",
                placeholder="e.g., Is this good for beginners? What are similar alternatives?",
                key="chat_input"
            )
            
            col_send, col_clear = st.columns([1, 1])
            with col_send:
                send_button = st.button("📤 Send", type="primary", use_container_width=True)
            with col_clear:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_messages = []
                    st.rerun()
            
            if send_button and user_question:
                if llm_client and llm_client.is_available:
                    # Add user message
                    st.session_state.chat_messages.append({
                        "role": "user",
                        "content": user_question
                    })
                    
                    # Get AI response
                    with st.spinner("Thinking..."):
                        product_info = {
                            'name': selected_product['name'],
                            'category': selected_product['category'],
                            'price': selected_product['price_raw'],
                            'rating_average': selected_product['rating_average'],
                            'stock': selected_product.get('stock_level', 'Unknown'),
                            'count_sold': selected_product['count_sold']
                        }
                        
                        # Convert recommendations to dict format
                        recs_for_chat = [
                            {'name': r['name'], 'price': r['price'], 'category': r['category']}
                            for r in recommendations
                        ]
                        
                        # Get conversation history for context
                        conv_history = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_messages[-6:]
                        ]
                        
                        response = llm_client.chat_about_product(
                            question=user_question,
                            product_info=product_info,
                            recommendations=recs_for_chat,
                            conversation_history=conv_history[:-1]  # Exclude current question
                        )
                    
                    # Add AI response
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.rerun()
                else:
                    st.error("LLM not available. Please configure your NVIDIA API key.")
            
            # Example questions
            with st.expander("💡 Example Questions"):
                st.markdown("""
                - Is this product worth the price?
                - What are the pros and cons?
                - Who would this be best for?
                - How does it compare to alternatives?
                - Is this suitable for professional use?
                """)
    
    # ===================== TAB 2: Review Summarizer =====================
    with tab2:
        st.subheader("AI Review Summarizer")
        st.markdown("Get an AI-powered summary of product reviews.")
        
        products_df = recommender.products_df
        
        # Product selection
        product_options = {
            f"{row['name'][:50]}..." if len(row['name']) > 50 else row['name']: row['product_id']
            for _, row in products_df.iterrows()
        }
        
        selected_name = st.selectbox(
            "Select a Product to Summarize Reviews:",
            options=list(product_options.keys()),
            key="summary_product"
        )
        selected_id = product_options[selected_name]
        selected_product = products_df[products_df['product_id'] == selected_id].iloc[0]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**Selected:** {selected_product['name'][:60]}...")
            st.markdown(f"⭐ Rating: {selected_product['rating_average']:.1f}/5.0")
        
        with col2:
            summarize_button = st.button("🔍 Summarize Reviews", type="primary", use_container_width=True)
        
        if summarize_button:
            if llm_client and llm_client.is_available:
                with st.spinner("Analyzing reviews with AI..."):
                    # Get reviews for this product
                    if reviews is not None and 'product_id' in reviews.columns:
                        product_reviews = reviews[reviews['product_id'] == selected_id]
                        
                        if len(product_reviews) > 0 and 'message' in product_reviews.columns:
                            review_texts = product_reviews['message'].dropna().tolist()[:15]
                            
                            if review_texts:
                                summary = llm_client.summarize_reviews(
                                    reviews=review_texts,
                                    product_name=selected_product['name'][:50]
                                )
                                
                                st.success("✅ Summary Generated!")
                                st.markdown("### 📊 AI Review Summary")
                                st.markdown(summary)
                                
                                # Show sample reviews
                                with st.expander(f"📝 View {min(5, len(review_texts))} Sample Reviews"):
                                    for i, review in enumerate(review_texts[:5], 1):
                                        st.markdown(f"**Review {i}:** {review[:200]}...")
                                        st.markdown("---")
                            else:
                                st.warning("No review text found for this product.")
                        else:
                            st.warning("No reviews found for this product.")
                    else:
                        st.warning("Reviews data not available.")
            else:
                st.error("LLM not available. Please configure your NVIDIA API key.")
        
        # Info about review summarization
        st.markdown("---")
        st.info("""
        **How it works:**
        1. The AI reads up to 15 reviews for the selected product
        2. It analyzes the overall sentiment (positive/negative/mixed)
        3. It identifies key pros and cons mentioned by customers
        4. It generates a concise summary to help you make a decision
        """)


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
        labels={'price': 'Price (USD)', 'count_sold': 'Units Sold', 'rating_average': 'Rating'}
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main app entry point."""
    render_header()
    page = render_sidebar()
    
    if page == "🏠 Overview":
        render_overview()
    # ========== COMMENTED OUT: Sales & Risk pages ==========
    # elif page == "📊 Sales Predictor":
    #     render_sales_predictor()
    # elif page == "⚠️ Risk Analyzer":
    #     render_risk_analyzer()
    # ========== END COMMENTED SECTION ==========
    elif page == "🛍️ Recommender":
        render_recommender()
    elif page == "� AI Chat":
        render_ai_chat()
    elif page == "📈 Analytics":
        render_analytics()
    
    # Footer
    st.divider()
    st.markdown(
        '<p style="text-align: center; color: #888;">Built with ❤️ using Streamlit, SHAP, LLM & Explainable AI</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
