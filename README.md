# 🏙️ PropSense · Islamabad Property Price Predictor

A machine learning web application that predicts house prices in Islamabad using real estate data scraped from Zameen.com. Built as a final project for an ML lab course.

---

## ✨ Features

- **Price Prediction** — Enter location, area, bedrooms, and bathrooms to get an instant price estimate in PKR
- **Confidence Range** — Displays a low–high price band alongside the predicted value
- **Market Intelligence** — Browse sector-wise price per marla, luxury sector multipliers, and area category breakdowns
- **Location Insights** — Per-location stats with data richness badges (Sparse / Moderate / Rich)
- **Model Comparison** — View accuracy, R², MAE, and RMSE across 9 trained models
- **Dark UI** — Styled Streamlit dashboard with custom CSS (DM Sans + Playfair Display)

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Web App | Streamlit |
| ML Models | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Data | Pandas, NumPy |
| Scraping | Selenium, BeautifulSoup4, Requests |
| Serialization | Joblib (`.pkl` files) |
| Data Source | Zameen.com (Islamabad Houses) |

---

## 📁 Project Structure

```
final_project/
├── app/
│   ├── app.py                      # Streamlit app (main entry point)
│   ├── best_model.pkl              # Best performing model (LightGBM+XGBoost Blend)
│   ├── model_metrics.json          # Accuracy metrics for all 9 models
│   ├── pricing_intelligence.json   # Sector PPM, luxury multipliers, area categories
│   ├── knn_imputer.pkl             # KNN imputer for missing values
│   ├── loc_mean_map.pkl            # Per-location mean price map
│   ├── sector_ppm.pkl              # Sector price-per-marla lookup
│   ├── top_locs_ohe.pkl            # One-hot encoder for top locations
│   ├── expected_cols.pkl           # Feature column names for the model
│   └── charts/                     # Pre-generated analysis charts
│       ├── actual_vs_predicted.png
│       ├── error_distribution.png
│       ├── feature_importance.png
│       ├── model_comparison.png
│       └── sector_ppm.png
├── data/
│   └── zameen_islamabad.csv        # Scraped dataset (~5000 house listings)
├── ml/
│   ├── train_model.ipynb           # Full training pipeline notebook
│   └── models/                     # All trained model files
│       ├── Linear Regression.pkl
│       ├── Ridge Regression.pkl
│       ├── Decision Tree.pkl
│       ├── Random Forest.pkl
│       ├── Gradient Boosting.pkl
│       ├── CatBoost.pkl
│       ├── LightGBM.pkl
│       ├── XGBoost.pkl
│       └── best_model.pkl
└── scraper/
    └── scraper.py                  # Zameen.com multi-threaded scraper
```

---

## 📊 Model Performance

| Model | R² | Accuracy | MAE |
|---|---|---|---|
| Gradient Boosting | 0.891 | 92.3% | 20.8M PKR |
| Random Forest | 0.891 | 92.1% | 20.9M PKR |
| LightGBM+XGBoost Blend | 0.886 | 91.9% | 20.6M PKR |
| XGBoost | 0.888 | 91.7% | 20.4M PKR |
| CatBoost | 0.881 | 91.8% | 20.5M PKR |
| LightGBM | 0.873 | 91.5% | 21.3M PKR |
| Ridge Regression | 0.870 | 89.0% | 22.0M PKR |
| Linear Regression | 0.866 | 89.6% | 22.0M PKR |
| Decision Tree | 0.840 | 87.8% | 24.6M PKR |

> Best deployed model: **Gradient Boosting** (highest accuracy at 92.3%)

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit scikit-learn xgboost lightgbm catboost pandas numpy joblib selenium webdriver-manager beautifulsoup4 requests
```

### Run the App

```bash
cd final_project/app
streamlit run app.py
```

### Re-scrape Data (optional)

```bash
cd final_project/scraper
python scraper.py
```

> Requires Google Chrome installed. The scraper uses Selenium with ChromeDriver (auto-managed via `webdriver-manager`). Targets up to 5,000 house listings from Zameen.com Islamabad.

### Retrain Models (optional)

Open and run `final_project/ml/train_model.ipynb` in Jupyter after collecting fresh data.

---

## 🏘️ Coverage

The app covers **100+ sectors and housing societies** in Islamabad including DHA, Bahria Town, F-sectors, G-sectors, E-7, NUST area, Park View City, and more. Global average price per marla is **PKR 54.74 Lakh** with a city-wide mean house price of **PKR 11.99 Crore**.

---

## 👥 Authors

Developed as an ML Lab Final Project — FA24-BAI-011.

---

## 📄 License

This project is for academic/educational use only.
