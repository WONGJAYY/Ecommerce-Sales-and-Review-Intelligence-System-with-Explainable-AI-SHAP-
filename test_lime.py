"""
Test script to verify LIME installation and functionality.
Run this script to check if LIME is working correctly.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

def test_lime_installation():
    """Test if LIME is installed."""
    print("=" * 50)
    print("TEST 1: Checking LIME Installation")
    print("=" * 50)
    
    try:
        import lime
        import lime.lime_tabular
        print("✓ LIME is installed!")
        return True
    except ImportError as e:
        print(f"✗ LIME is NOT installed: {e}")
        print("  Run: pip install lime")
        return False


def test_lime_explainer():
    """Test LIME explainer with dummy data."""
    print("\n" + "=" * 50)
    print("TEST 2: Testing LIME Explainer")
    print("=" * 50)
    
    import lime.lime_tabular
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] * 0.5 + np.random.rand(100) * 0.1
    feature_names = ['important_feat', 'medium_feat', 'noise_1', 'noise_2', 'noise_3']
    
    # Train model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    print("✓ Model trained successfully")
    
    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode='regression'
    )
    print("✓ LIME explainer created")
    
    # Explain a prediction
    test_instance = X_train[0]
    explanation = explainer.explain_instance(
        test_instance, 
        model.predict, 
        num_features=5
    )
    print("✓ Explanation generated")
    
    # Show results
    print("\nTop contributing features:")
    for feat, weight in explanation.as_list():
        direction = "↑" if weight > 0 else "↓"
        print(f"  {direction} {feat}: {weight:+.4f}")
    
    return True


def test_custom_lime_module():
    """Test the custom LIME explainer module."""
    print("\n" + "=" * 50)
    print("TEST 3: Testing Custom LIME Module")
    print("=" * 50)
    
    try:
        from src.explainability.lime_explainer import (
            LIMEExplainer, 
            LIMEExplanation,
            CombinedExplainer
        )
        print("✓ Custom LIME module imported successfully")
        
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        
        # Create dummy data
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.rand(100, 5),
            columns=['price', 'rating', 'reviews', 'stock', 'age']
        )
        y_train = X_train['price'] * 2 + X_train['rating'] * 0.5
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Create custom LIME explainer
        lime_explainer = LIMEExplainer(
            model=model,
            training_data=X_train,
            mode='regression'
        )
        print("✓ Custom LIMEExplainer created")
        
        # Generate explanation
        explanation = lime_explainer.explain(X_train.iloc[[0]])
        print("✓ Explanation generated")
        
        # Show top features
        print("\nTop features (custom module):")
        for feat in explanation.get_top_features(5):
            direction = "↑" if feat['direction'] == 'positive' else "↓"
            print(f"  {direction} {feat['feature']}: {feat['lime_weight']:+.4f}")
        
        print(f"\nLocal model R² score: {explanation.score:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Custom module import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("   LIME INSTALLATION & FEATURE TEST")
    print("=" * 50)
    
    results = []
    
    # Test 1: Installation
    results.append(("LIME Installation", test_lime_installation()))
    
    # Test 2: Basic functionality
    if results[-1][1]:
        results.append(("LIME Explainer", test_lime_explainer()))
    
    # Test 3: Custom module
    if results[-1][1]:
        results.append(("Custom LIME Module", test_custom_lime_module()))
    
    # Summary
    print("\n" + "=" * 50)
    print("   TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("All tests passed! LIME is ready to use.")
    else:
        print("Some tests failed. Check the errors above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
