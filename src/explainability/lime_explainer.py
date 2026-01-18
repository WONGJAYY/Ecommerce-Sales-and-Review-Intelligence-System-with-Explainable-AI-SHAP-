"""
LIME Explainer Module

Local Interpretable Model-agnostic Explanations (LIME) for model explainability.
Provides local explanations by fitting interpretable models around predictions.

Methods:
    - LimeTabularExplainer: For tabular data (numerical and categorical)
    - Supports both regression and classification models

Comparison with SHAP:
    - LIME: Fast, local perturbation-based, may vary between runs
    - SHAP: Theoretically grounded, consistent, based on game theory
    
Use both for cross-validation of explanations.
"""

import logging
from typing import Optional, Dict, List, Union, Any, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None

logger = logging.getLogger(__name__)


@dataclass
class LIMEExplanation:
    """
    Container for LIME explanation results.
    
    Attributes:
        feature_weights: LIME weights for each feature (local linear model coefficients)
        intercept: Intercept of the local linear model
        prediction: Model's prediction for this instance
        local_prediction: Local model's prediction
        feature_names: Names of features
        feature_values: Actual feature values for the instance
        score: R² score of the local model (how well it approximates locally)
    """
    feature_weights: Dict[str, float]
    intercept: float
    prediction: float
    local_prediction: float
    feature_names: List[str]
    feature_values: np.ndarray
    score: float
    
    def get_top_features(self, n: int = 5) -> List[Dict]:
        """Get top N most important features for this prediction."""
        # Sort by absolute weight
        sorted_features = sorted(
            self.feature_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:n]
        
        feature_idx = {name: i for i, name in enumerate(self.feature_names)}
        
        return [
            {
                'feature': name,
                'lime_weight': float(weight),
                'feature_value': float(self.feature_values[feature_idx[name]]) 
                    if name in feature_idx and np.isfinite(self.feature_values[feature_idx[name]]) 
                    else None,
                'direction': 'positive' if weight > 0 else 'negative'
            }
            for name, weight in sorted_features
        ]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            'prediction': float(self.prediction),
            'local_prediction': float(self.local_prediction),
            'intercept': float(self.intercept),
            'local_model_score': float(self.score),
            'feature_weights': {
                name: float(weight) 
                for name, weight in self.feature_weights.items()
            },
            'top_drivers': self.get_top_features(5)
        }


class LIMEExplainer:
    """
    Main explainer interface for generating LIME explanations.
    
    Provides local interpretable explanations by fitting a linear model
    around each prediction.
    
    Example:
        explainer = LIMEExplainer(model, X_train, feature_names)
        explanation = explainer.explain(X.iloc[[0]])
        print(explanation.get_top_features())
    """
    
    def __init__(
        self,
        model: Any,
        training_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        mode: str = 'regression',
        kernel_width: Optional[float] = None,
        num_samples: int = 5000,
        random_state: int = 42,
        predict_fn: Optional[Callable] = None
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model (SalesPredictor, ReviewRiskPredictor, or any sklearn-like model)
            training_data: Training dataset for determining feature statistics
            feature_names: Feature column names
            categorical_features: Indices of categorical features
            mode: 'regression' or 'classification'
            kernel_width: Width of exponential kernel (None = auto)
            num_samples: Number of samples to generate for local approximation
            random_state: Random seed for reproducibility
            predict_fn: Optional custom predict function (overrides model's predict)
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required. Install with: pip install lime")
        
        self.model = model
        self.mode = mode
        self.num_samples = num_samples
        self.random_state = random_state
        
        # Extract feature names
        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif hasattr(training_data, 'columns'):
            self.feature_names = list(training_data.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(training_data.shape[1])]
        
        # Use custom predict function if provided
        if predict_fn is not None:
            self._predict_fn = predict_fn
            self._model = None
        else:
            # Extract the underlying model's predict function
            if hasattr(model, 'model'):
                self._model = model.model
            else:
                self._model = model
            
            # Determine predict function based on mode
            if mode == 'classification':
                # For classification, LIME needs predict_proba
                if hasattr(self._model, 'predict_proba'):
                    self._predict_fn = self._model.predict_proba
                else:
                    # Wrap predict to return probability-like output
                    self._predict_fn = self._wrap_predict_for_classification()
            else:
                self._predict_fn = self._model.predict
        
        # Convert training data to numpy if needed
        if isinstance(training_data, pd.DataFrame):
            training_data_np = training_data.values
        else:
            training_data_np = training_data
        
        # Create LIME explainer
        self._explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data_np,
            feature_names=self.feature_names,
            categorical_features=categorical_features,
            mode=mode,
            kernel_width=kernel_width,
            random_state=random_state,
            verbose=False
        )
        
        logger.info(f"Initialized LIME explainer in {mode} mode with {len(self.feature_names)} features")
    
    def _wrap_predict_for_classification(self) -> Callable:
        """Wrap predict function for classification mode."""
        def predict_proba(X):
            preds = self._model.predict(X)
            # Convert to probability format [P(0), P(1)]
            if len(preds.shape) == 1:
                return np.column_stack([1 - preds, preds])
            return preds
        return predict_proba
    
    def explain(
        self,
        X: pd.DataFrame,
        num_features: int = 10,
        num_samples: Optional[int] = None
    ) -> LIMEExplanation:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            X: Single row DataFrame or numpy array
            num_features: Number of top features to include in explanation
            num_samples: Number of samples for local approximation (overrides default)
            
        Returns:
            LIMEExplanation object
        """
        if isinstance(X, pd.DataFrame):
            if len(X) > 1:
                logger.warning("Multiple rows provided, using first row only")
                X = X.iloc[[0]]
            instance = X.values.flatten()
        else:
            instance = np.array(X).flatten()
        
        samples = num_samples or self.num_samples
        
        # Generate LIME explanation
        lime_exp = self._explainer.explain_instance(
            data_row=instance,
            predict_fn=self._predict_fn,
            num_features=num_features,
            num_samples=samples
        )
        
        # Extract feature weights
        feature_weights = dict(lime_exp.as_list())
        
        # Map discretized feature names back to original names
        # LIME may add conditions like "feature > 0.5", extract base feature name
        cleaned_weights = {}
        for feat_condition, weight in feature_weights.items():
            # Find which original feature this corresponds to
            for orig_name in self.feature_names:
                if orig_name in feat_condition:
                    # Aggregate weights if multiple conditions for same feature
                    if orig_name in cleaned_weights:
                        cleaned_weights[orig_name] += weight
                    else:
                        cleaned_weights[orig_name] = weight
                    break
            else:
                # If no match found, keep original
                cleaned_weights[feat_condition] = weight
        
        # Get prediction
        if self.mode == 'classification':
            proba = self._predict_fn(instance.reshape(1, -1))
            prediction = proba[0][1] if proba.shape[1] > 1 else proba[0][0]
        else:
            prediction = float(self._model.predict(instance.reshape(1, -1))[0])
        
        # Get local prediction and score
        local_pred = lime_exp.local_pred[0] if hasattr(lime_exp, 'local_pred') else prediction
        score = lime_exp.score if hasattr(lime_exp, 'score') else 0.0
        intercept = lime_exp.intercept[1] if self.mode == 'classification' else lime_exp.intercept
        
        return LIMEExplanation(
            feature_weights=cleaned_weights,
            intercept=float(intercept) if isinstance(intercept, (int, float, np.number)) else float(intercept[0]),
            prediction=float(prediction),
            local_prediction=float(local_pred),
            feature_names=self.feature_names,
            feature_values=instance,
            score=float(score)
        )
    
    def explain_batch(
        self,
        X: pd.DataFrame,
        num_features: int = 10,
        num_samples: Optional[int] = None
    ) -> List[LIMEExplanation]:
        """
        Generate LIME explanations for multiple instances.
        
        Note: LIME is inherently local, so each instance is explained separately.
        This is slower than batch SHAP but provides diverse local approximations.
        
        Args:
            X: Multi-row DataFrame
            num_features: Number of features per explanation
            num_samples: Samples per explanation
            
        Returns:
            List of LIMEExplanation objects
        """
        explanations = []
        for i in range(len(X)):
            if isinstance(X, pd.DataFrame):
                instance = X.iloc[[i]]
            else:
                instance = X[i:i+1]
            
            exp = self.explain(instance, num_features, num_samples)
            explanations.append(exp)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Explained {i + 1}/{len(X)} instances")
        
        return explanations
    
    def get_feature_importance(
        self,
        X: pd.DataFrame,
        num_features: int = 10,
        num_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute average feature importance across multiple instances.
        
        Unlike SHAP global importance, LIME importance is aggregated from
        local explanations, so it may vary with the sample set.
        
        Args:
            X: Dataset to compute importance over
            num_features: Features per local explanation
            num_samples: Samples per explanation
            
        Returns:
            DataFrame with feature importance
        """
        explanations = self.explain_batch(X, num_features, num_samples)
        
        # Aggregate weights across all explanations
        importance_sum = {name: 0.0 for name in self.feature_names}
        
        for exp in explanations:
            for name in self.feature_names:
                if name in exp.feature_weights:
                    importance_sum[name] += abs(exp.feature_weights[name])
        
        # Average
        n = len(explanations)
        importance = {name: val / n for name, val in importance_sum.items()}
        
        return pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)


class CombinedExplainer:
    """
    Combined SHAP + LIME explainer for cross-validated explanations.
    
    Uses both methods and provides comparison metrics to validate
    explanation consistency.
    
    Example:
        explainer = CombinedExplainer(model, X_train)
        result = explainer.explain(X.iloc[[0]])
        print(result['comparison'])  # Agreement metrics
    """
    
    def __init__(
        self,
        model: Any,
        training_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        mode: str = 'regression'
    ):
        """
        Initialize combined explainer.
        
        Args:
            model: Trained model
            training_data: Background/training data
            feature_names: Feature column names
            mode: 'regression' or 'classification'
        """
        self.feature_names = feature_names or list(training_data.columns)
        self.mode = mode
        
        # Initialize SHAP explainer
        try:
            from src.explainability.shap_explainer import SHAPExplainer
            self.shap_explainer = SHAPExplainer(
                model=model,
                background_data=training_data,
                feature_names=self.feature_names
            )
            self.shap_available = True
            logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"SHAP explainer not available: {e}")
            self.shap_explainer = None
            self.shap_available = False
        
        # Initialize LIME explainer
        try:
            self.lime_explainer = LIMEExplainer(
                model=model,
                training_data=training_data,
                feature_names=self.feature_names,
                mode=mode
            )
            self.lime_available = True
            logger.info("LIME explainer initialized")
        except Exception as e:
            logger.warning(f"LIME explainer not available: {e}")
            self.lime_explainer = None
            self.lime_available = False
    
    def explain(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate explanations using both SHAP and LIME.
        
        Args:
            X: Single row DataFrame
            
        Returns:
            Dictionary with SHAP, LIME, and comparison results
        """
        result = {
            'shap': None,
            'lime': None,
            'comparison': None
        }
        
        # Get SHAP explanation
        if self.shap_available:
            try:
                shap_exp = self.shap_explainer.explain(X)
                result['shap'] = shap_exp.to_dict()
            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}")
        
        # Get LIME explanation
        if self.lime_available:
            try:
                lime_exp = self.lime_explainer.explain(X)
                result['lime'] = lime_exp.to_dict()
            except Exception as e:
                logger.error(f"LIME explanation failed: {e}")
        
        # Compare explanations
        if result['shap'] and result['lime']:
            result['comparison'] = self._compare_explanations(
                result['shap'], result['lime']
            )
        
        return result
    
    def _compare_explanations(
        self,
        shap_result: Dict,
        lime_result: Dict
    ) -> Dict[str, Any]:
        """
        Compare SHAP and LIME explanations.
        
        Returns metrics on how well the two methods agree.
        """
        shap_top = [d['feature'] for d in shap_result['top_drivers']]
        lime_top = [d['feature'] for d in lime_result['top_drivers']]
        
        # Top-k overlap
        top_3_overlap = len(set(shap_top[:3]) & set(lime_top[:3])) / 3
        top_5_overlap = len(set(shap_top[:5]) & set(lime_top[:5])) / 5
        
        # Direction agreement for top features
        shap_directions = {
            d['feature']: d['direction'] for d in shap_result['top_drivers']
        }
        lime_directions = {
            d['feature']: d['direction'] for d in lime_result['top_drivers']
        }
        
        common_features = set(shap_directions.keys()) & set(lime_directions.keys())
        if common_features:
            direction_agreement = sum(
                1 for f in common_features
                if shap_directions[f] == lime_directions[f]
            ) / len(common_features)
        else:
            direction_agreement = 0.0
        
        # Rank correlation (Spearman) for common features
        shap_vals = shap_result.get('shap_values', {})
        lime_vals = lime_result.get('feature_weights', {})
        
        common = set(shap_vals.keys()) & set(lime_vals.keys())
        if len(common) >= 3:
            from scipy.stats import spearmanr
            shap_ranks = [abs(shap_vals[f]) for f in common]
            lime_ranks = [abs(lime_vals[f]) for f in common]
            rank_corr, _ = spearmanr(shap_ranks, lime_ranks)
        else:
            rank_corr = None
        
        return {
            'top_3_overlap': top_3_overlap,
            'top_5_overlap': top_5_overlap,
            'direction_agreement': direction_agreement,
            'rank_correlation': rank_corr,
            'agreement_level': self._get_agreement_level(top_5_overlap, direction_agreement)
        }
    
    def _get_agreement_level(self, overlap: float, direction: float) -> str:
        """Interpret agreement metrics."""
        avg = (overlap + direction) / 2
        if avg >= 0.8:
            return 'high'
        elif avg >= 0.5:
            return 'moderate'
        else:
            return 'low'


def compute_lime_weights(
    model: Any,
    X: pd.DataFrame,
    training_data: pd.DataFrame,
    mode: str = 'regression'
) -> np.ndarray:
    """
    Convenience function to compute LIME weights.
    
    Args:
        model: Trained model
        X: Features to explain
        training_data: Background data
        mode: 'regression' or 'classification'
        
    Returns:
        Array of LIME weights
    """
    explainer = LIMEExplainer(model, training_data, mode=mode)
    explanations = explainer.explain_batch(X)
    
    # Extract weights as array
    feature_names = explainer.feature_names
    weights = np.zeros((len(X), len(feature_names)))
    
    for i, exp in enumerate(explanations):
        for j, name in enumerate(feature_names):
            if name in exp.feature_weights:
                weights[i, j] = exp.feature_weights[name]
    
    return weights


def get_lime_feature_importance(
    model: Any,
    X: pd.DataFrame,
    training_data: pd.DataFrame,
    mode: str = 'regression'
) -> pd.DataFrame:
    """
    Get feature importance from LIME.
    
    Args:
        model: Trained model
        X: Features
        training_data: Background data
        mode: 'regression' or 'classification'
        
    Returns:
        Feature importance DataFrame
    """
    explainer = LIMEExplainer(model, training_data, mode=mode)
    return explainer.get_feature_importance(X)
