# 🧠 E-Commerce Sales and Review Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-ML-green.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-orange.svg)](https://shap.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit]([https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://review-intelligence-system.streamlit.app/)

A production-grade **MLOps platform** for predicting product sales volume and identifying negative review risks with **SHAP-based explainability**. Built with LightGBM for high-performance inference and FastAPI for real-time serving.

---

## 🎯 Problem Statement

E-commerce platforms face two critical challenges:

1. **Sales Forecasting**: Predicting which products will sell well to optimize inventory and marketing
2. **Review Risk Management**: Identifying products likely to receive negative reviews before they impact brand reputation

This system addresses both challenges with interpretable ML models that provide actionable insights.

---

## ✨ Key Features

| Feature                      | Description                                                 |
| ---------------------------- | ----------------------------------------------------------- |
| 🔮 **Sales Prediction**      | Predict expected sales volume with confidence intervals     |
| ⚠️ **Risk Assessment**       | Identify products with high probability of negative reviews |
| 📊 **SHAP Explainability**   | Transparent explanations for every prediction               |
| ⚡ **Real-time API**         | FastAPI endpoints with <50ms inference time                 |
| 📈 **Interactive Dashboard** | Streamlit app for visualization and demos                   |
| 🔄 **Drift Detection**       | Monitor data and model drift in production                  |
| 📦 **Feature Store**         | Versioned feature management with Parquet persistence       |

---

## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Raw Data   │───▶│  Ingestion   │───▶│  Validation  │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                    ┌──────────────┐    ┌──────▼───────┐
                    │   Training   │◀───│ Preprocessing│
                    └──────────────┘    └──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│Sales Predictor│   │Risk Predictor│    │   Registry   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                    ┌──────────────┐
                    │   FastAPI    │
                    │   Serving    │
                    └──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Dashboard   │    │    SHAP      │    │  Monitoring  │
│ (Streamlit)  │    │ Explanations │    │(Drift/Metrics)│
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## 📁 Project Structure

```
├── app.py                    # Streamlit dashboard
├── configs/
│   ├── config.yaml           # Main configuration
│   ├── features.yaml         # Feature definitions
│   └── model_config.yaml     # Model hyperparameters
├── data/
│   ├── raw/                  # Raw dataset storage
│   ├── processed/            # Preprocessed data
│   └── features/             # Feature store
├── docker/
│   └── Dockerfile            # Container definition
├── logs/                     # Application logs
├── models/                   # Saved model artifacts
│   ├── sales_predictor/
│   └── review_risk_predictor/
├── scripts/
│   └── train.py              # Training pipeline script
├── src/
│   ├── data/                 # Data layer
│   │   ├── ingestion.py      # Data loading & parsing
│   │   ├── validation.py     # Data validation checks
│   │   └── preprocessing.py  # Cleaning & transformation
│   ├── features/             # Feature engineering
│   │   ├── feature_engineering.py
│   │   ├── feature_definitions.py
│   │   └── feature_store.py
│   ├── models/               # ML models
│   │   ├── sales_predictor.py
│   │   ├── review_risk_predictor.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── registry.py
│   ├── explainability/       # SHAP module
│   │   ├── shap_explainer.py
│   │   ├── explanations.py
│   │   └── visualization.py
│   ├── serving/              # API layer
│   │   ├── api.py            # FastAPI endpoints
│   │   ├── inference.py      # Inference engine
│   │   └── schemas.py        # Pydantic models
│   ├── monitoring/           # Production monitoring
│   │   ├── data_drift.py
│   │   ├── model_drift.py
│   │   └── metrics.py
│   └── utils/                # Utilities
│       ├── config.py
│       └── logging.py
└── tests/                    # Unit tests
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/MuneebMM/Ecommerce-Sales-and-Review-Intelligence-System-with-Explainable-AI-SHAP-.git
cd Ecommerce-Sales-and-Review-Intelligence-System-with-Explainable-AI-SHAP-

# Install dependencies
pip install -r requirements.txt

# Place your dataset in the root directory
# tokopedia_products_with_review.csv
```

### Train Models

```bash
# Train both models
python scripts/train.py --model all --nrows 5000

# Train specific model
python scripts/train.py --model risk --nrows 1000
```

### Start the API Server

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

### Launch Dashboard

```bash
streamlit run app.py --server.port 8501
```

---

## 📡 API Endpoints

| Endpoint                | Method | Description                    |
| ----------------------- | ------ | ------------------------------ |
| `/health`               | GET    | Health check with model status |
| `/v1/predict/sales`     | POST   | Sales volume prediction        |
| `/v1/predict/risk`      | POST   | Review risk assessment         |
| `/v1/model/info/{type}` | GET    | Model information              |
| `/docs`                 | GET    | Swagger documentation          |

### Example: Risk Prediction

```bash
curl -X POST "http://localhost:8000/v1/predict/risk" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "SKU123",
    "price": 150000,
    "message_length": 50,
    "word_count": 10
  }'
```

**Response:**

```json
{
  "product_id": "SKU123",
  "risk_probability": 0.15,
  "risk_level": "low",
  "is_high_risk": false,
  "model_version": "v1.0"
}
```

---

## 📊 Model Performance

| Model                     | Metric   | Score      |
| ------------------------- | -------- | ---------- |
| **Review Risk Predictor** | AUC-ROC  | 0.84       |
| **Review Risk Predictor** | F1 Score | 0.72       |
| **Sales Predictor**       | R² Score | 0.68       |
| **Sales Predictor**       | MAE      | 45.2 units |

---

## 🔍 SHAP Explainability

Every prediction includes interpretable explanations powered by **TreeSHAP**:

- **Feature Importance**: Which factors drive each prediction
- **Waterfall Charts**: Visual breakdown of feature contributions
- **Counterfactuals**: Actionable recommendations for improvement

---

## 🛡️ Production Monitoring

The system includes comprehensive monitoring:

- **Data Drift Detection**: KS-test and Chi-squared for distribution shifts
- **Model Drift Detection**: PSI (Population Stability Index) tracking
- **Performance Monitoring**: Real-time metric collection (Prometheus-compatible)

---

## 🧪 Dataset

This project uses the **Tokopedia Products with Reviews** dataset containing:

- Product information (price, stock, category, shop details)
- Customer reviews (ratings, messages, timestamps)
- ~30,000+ reviews from ~500 products

**Note**: The dataset is not included in the repository due to size. Place `tokopedia_products_with_review.csv` in the root directory before training.

---

## 🛠️ Tech Stack

| Category           | Technologies                |
| ------------------ | --------------------------- |
| **ML Framework**   | LightGBM, scikit-learn      |
| **Explainability** | SHAP                        |
| **API**            | FastAPI, Pydantic           |
| **Dashboard**      | Streamlit, Plotly           |
| **Data**           | Pandas, NumPy               |
| **Monitoring**     | Custom metrics, scipy stats |

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

**Muneeb MM**

- GitHub: [@MuneebMM](https://github.com/MuneebMM)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
