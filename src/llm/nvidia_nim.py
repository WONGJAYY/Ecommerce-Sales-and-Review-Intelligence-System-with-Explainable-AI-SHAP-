"""
NVIDIA NIM Cloud API Client

Provides LLM-powered natural language explanations for product recommendations.
Uses NVIDIA's cloud API which is OpenAI-compatible.

Setup:
    1. Go to https://build.nvidia.com
    2. Sign up and get your API key
    3. Set environment variable: NVIDIA_API_KEY=your_key_here
    
Example:
    from src.llm.nvidia_nim import NVIDIANimClient
    
    client = NVIDIANimClient()
    explanation = client.explain_recommendation(
        source_product={"name": "iPhone 15", "price": 999},
        recommended_product={"name": "iPhone 15 Pro", "price": 1199},
        shap_values={"price": 0.3, "category": 0.5, "rating": 0.2}
    )
    print(explanation)
"""

import os
import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env file if exists
def load_env_file():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env_file()

# Check for OpenAI library (used for NVIDIA NIM compatibility)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not installed. Run: pip install openai")


class NVIDIANimClient:
    """
    NVIDIA NIM Cloud API Client for LLM-powered explanations.
    
    Uses OpenAI-compatible API to communicate with NVIDIA's LLM endpoints.
    """
    
    # NVIDIA NIM API endpoint
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    
    # Available models on NVIDIA NIM
    AVAILABLE_MODELS = {
        "llama-3.3-70b": "meta/llama-3.3-70b-instruct",
        "llama-3.1-8b": "meta/llama-3.1-8b-instruct",
        "llama-3.1-70b": "meta/llama-3.1-70b-instruct", 
        "llama-3.1-405b": "meta/llama-3.1-405b-instruct",
        "mistral-7b": "mistralai/mistral-7b-instruct-v0.3",
        "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
        "gemma-2-9b": "google/gemma-2-9b-it",
        "phi-3-mini": "microsoft/phi-3-mini-128k-instruct",
    }
    
    DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"  # Using Llama 3.3 70B
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b",
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        """
        Initialize NVIDIA NIM client.
        
        Args:
            api_key: NVIDIA API key. If None, reads from NVIDIA_API_KEY env var.
            model: Model to use (see AVAILABLE_MODELS).
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in response.
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Resolve model name
        if model in self.AVAILABLE_MODELS:
            self.model = self.AVAILABLE_MODELS[model]
        else:
            self.model = model
            
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the OpenAI-compatible client."""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI library not available")
            return
            
        if not self.api_key:
            logger.warning("NVIDIA API key not set. Set NVIDIA_API_KEY environment variable.")
            return
            
        try:
            self.client = OpenAI(
                base_url=self.BASE_URL,
                api_key=self.api_key
            )
            logger.info(f"NVIDIA NIM client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA NIM client: {e}")
            
    @property
    def is_available(self) -> bool:
        """Check if the LLM client is available."""
        return self.client is not None
    
    def _create_recommendation_prompt(
        self,
        source_product: Dict[str, Any],
        recommended_product: Dict[str, Any],
        shap_values: Dict[str, float],
        similarity_score: float
    ) -> str:
        """Create a prompt for explaining a recommendation."""
        
        # Sort SHAP values by importance
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:5]  # Top 5 features
        
        feature_explanations = "\n".join([
            f"- {feat}: {val:.3f} ({'positive' if val > 0 else 'negative'} influence)"
            for feat, val in top_features
        ])
        
        prompt = f"""You are an AI shopping assistant. Explain why a product is recommended to a customer in a friendly, conversational way.

SOURCE PRODUCT (what customer is viewing):
- Name: {source_product.get('name', 'Unknown')}
- Category: {source_product.get('category', 'Unknown')}
- Price: ${source_product.get('price', 0):.2f}
- Rating: {source_product.get('rating_average', 'N/A')}

RECOMMENDED PRODUCT:
- Name: {recommended_product.get('name', 'Unknown')}
- Category: {recommended_product.get('category', 'Unknown')}
- Price: ${recommended_product.get('price', 0):.2f}
- Rating: {recommended_product.get('rating_average', 'N/A')}
- Similarity Score: {similarity_score*100:.1f}%

KEY FACTORS (from AI analysis):
{feature_explanations}

Write a 2-3 sentence explanation for why this product is recommended. Be specific about the features that make it similar. Don't mention technical terms like "SHAP" or "similarity score"."""

        return prompt
    
    def explain_recommendation(
        self,
        source_product: Dict[str, Any],
        recommended_product: Dict[str, Any],
        shap_values: Dict[str, float],
        similarity_score: float = 0.0
    ) -> str:
        """
        Generate a natural language explanation for a recommendation.
        
        Args:
            source_product: The product the user is viewing.
            recommended_product: The recommended product.
            shap_values: SHAP feature contributions.
            similarity_score: Cosine similarity score (0-1).
            
        Returns:
            Natural language explanation string.
        """
        if not self.is_available:
            return self._fallback_explanation(shap_values, similarity_score)
            
        try:
            prompt = self._create_recommendation_prompt(
                source_product, 
                recommended_product, 
                shap_values,
                similarity_score
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful e-commerce assistant that explains product recommendations clearly and concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            logger.error(f"LLM explanation failed: {e}")
            return self._fallback_explanation(shap_values, similarity_score)
    
    def _fallback_explanation(
        self, 
        shap_values: Dict[str, float], 
        similarity_score: float
    ) -> str:
        """Generate a fallback explanation without LLM."""
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        top_3 = sorted_features[:3]
        
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
        
        reasons = [feature_names.get(f, f) for f, _ in top_3]
        return f"Recommended because of {', '.join(reasons)} ({similarity_score*100:.0f}% match)."
    
    def summarize_reviews(
        self,
        reviews: List[str],
        product_name: str = "this product"
    ) -> str:
        """
        Summarize multiple product reviews into key points.
        
        Args:
            reviews: List of review texts.
            product_name: Name of the product.
            
        Returns:
            Summary of reviews.
        """
        if not self.is_available:
            return "Review summary not available (LLM not configured)."
            
        if not reviews:
            return "No reviews available."
            
        # Take first 10 reviews to avoid token limits
        sample_reviews = reviews[:10]
        reviews_text = "\n---\n".join(sample_reviews)
        
        prompt = f"""Summarize the following customer reviews for {product_name}. 
Provide:
1. Overall sentiment (positive/negative/mixed)
2. Key pros mentioned
3. Key cons mentioned
4. A brief 1-sentence summary

REVIEWS:
{reviews_text}

Keep your response concise and helpful for a potential buyer."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes product reviews objectively."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Review summarization failed: {e}")
            return "Review summary not available."
    
    def chat_about_product(
        self,
        question: str,
        product_info: Dict[str, Any],
        recommendations: List[Dict[str, Any]] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Answer questions about a product or recommendations.
        
        Args:
            question: User's question.
            product_info: Information about the current product.
            recommendations: List of recommended products (optional).
            conversation_history: Previous messages in conversation.
            
        Returns:
            AI response.
        """
        if not self.is_available:
            return "Chat not available. Please configure NVIDIA API key."
        
        # Build context
        context = f"""PRODUCT INFORMATION:
Name: {product_info.get('name', 'Unknown')}
Category: {product_info.get('category', 'Unknown')}
Price: ${product_info.get('price', 0):.2f}
Rating: {product_info.get('rating_average', 'N/A')} stars
Stock: {product_info.get('stock', 'Unknown')} units
"""
        
        if recommendations:
            context += "\nRECOMMENDED SIMILAR PRODUCTS:\n"
            for i, rec in enumerate(recommendations[:5], 1):
                context += f"{i}. {rec.get('name', 'Unknown')} - ${rec.get('price', 0):.2f}\n"
        
        system_prompt = f"""You are a helpful e-commerce shopping assistant. Answer customer questions about products based on the provided information. Be helpful, concise, and friendly.

{context}

If you don't have enough information to answer, say so politely."""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Keep last 6 messages
            
        messages.append({"role": "user", "content": question})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=256
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return "I'm sorry, I couldn't process your question. Please try again."
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to NVIDIA NIM API."""
        if not self.is_available:
            return {
                "status": "error",
                "message": "Client not initialized. Check API key.",
                "model": self.model
            }
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say 'Hello' in one word."}],
                max_tokens=10
            )
            
            return {
                "status": "success",
                "message": "Connection successful!",
                "model": self.model,
                "response": response.choices[0].message.content.strip()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "model": self.model
            }


# Convenience function
def get_llm_client(api_key: Optional[str] = None) -> NVIDIANimClient:
    """Get a configured NVIDIA NIM client."""
    return NVIDIANimClient(api_key=api_key)
