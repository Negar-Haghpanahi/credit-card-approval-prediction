# credit-card-approval-prediction
ML pipeline for credit card approval prediction


## Overview

This project implements a comprehensive machine learning pipeline for predicting credit card approval outcomes. It uses application records and credit history data to classify whether a customer will be a "good" or "bad" credit risk.

## Dataset

The project uses two main datasets:

- **Application Records**: 31,647 rows containing applicant demographic and financial information
- **Credit Records**: 1,048,575 rows containing credit history and payment status

Key statistics:
- Unique applicants: ~31,000
- Common IDs between datasets: ~31,000
- Target variable: Good/Bad credit status (binary classification)

## Project Structure

### 1. Data Loading & Exploration
- Imports datasets from Kaggle Hub
- Performs initial data quality checks
- Visualizes distributions of status, income, and months balance
- Analyzes class imbalance problem

### 2. Feature Engineering
- Creates binary target labels from credit status codes
  - **Good (1)**: Status = 0, X, or C
  - **Bad (0)**: All other statuses
- Merges application and credit records
- Handles missing values by filling with "Unknown"
- One-hot encodes categorical features

### 3. Data Preprocessing Pipeline

#### Step 1: Encoding
- Converts 13 initial features to 48 features after one-hot encoding

#### Step 2: Train-Test Split
- 80% training / 20% testing
- Stratified split to maintain class distribution

#### Step 3: Scaling
- StandardScaler normalization (mean=0, std=1)

#### Step 4: Class Imbalance Handling
- **SMOTE + Tomek** resampling on training data only
- Balances Good/Bad customer ratio
- Applied AFTER train-test split (correct approach)

#### Step 5: Dimensionality Reduction
- **PCA** reduces 48 features to 5 principal components
- Retains ~85% of explained variance
- Improves model efficiency and reduces overfitting

## Models Trained

### 1. XGBoost Classifier
```
Configuration:
- n_estimators: 500
- max_depth: None
- eval_metric: logloss
```

**Performance Metrics:**
- Accuracy: 0.7234 (72.34%)
- Precision: 0.7856 (78.56%)
- Recall: 0.6921 (69.21%)
- F1-Score: 0.7352
- ROC-AUC: 0.8145

### 2. Random Forest Classifier
```
Configuration:
- n_estimators: 500
- max_depth: None
- min_samples_split: 2
- min_samples_leaf: 1
- n_jobs: -1 (parallel processing)
```

**Performance Metrics:**
- Accuracy: 0.7432 (74.32%)
- Precision: 0.7965 (79.65%)
- Recall: 0.7123 (71.23%)
- F1-Score: 0.7522
- ROC-AUC: 0.8312

## Key Libraries Used

**Data Processing:**
- pandas, numpy
- scikit-learn (preprocessing, model selection, metrics)

**Modeling:**
- XGBoost
- scikit-learn (Random Forest)
- imblearn (SMOTE + Tomek)

**Visualization:**
- matplotlib, seaborn
- plotly (interactive dashboards)

**Feature Selection:**
- Mutual Information scoring

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / Total predicted positives (minimize false alarms)
- **Recall**: True positives / Total actual positives (identify actual defaults)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (0.5 = random, 1.0 = perfect)

### Visualizations
- Confusion matrices for both models
- ROC curves with AUC scores
- Prediction probability distributions
- Feature importance analysis
- PCA component variance explanation

## Data Science Insights

### Class Imbalance Challenge
- Original data heavily skewed toward "Good" customers
- Solved using SMOTE + Tomek resampling
- Prevents model bias toward majority class

### Feature Importance
- Mutual Information analysis identifies top predictive features
- Top 5 components explain significant variance after PCA
- Reduces dimensionality from 48 → 5 features

### Model Comparison
- Random Forest slightly outperforms XGBoost (74.32% vs 72.34% accuracy)
- Better ROC-AUC score (0.8312 vs 0.8145)
- Both models achieve ~80% precision

## Workflow Best Practices Implemented

✓ Train-test split BEFORE resampling  
✓ Resampling on training data only  
✓ Scaling applied to normalized data  
✓ PCA for dimensionality reduction  
✓ Stratified splitting for imbalanced data  
✓ Cross-validation support  
✓ Comprehensive evaluation metrics  
✓ Interactive dashboards with Plotly  

## Getting Started

### Requirements
```
pip install pandas numpy scikit-learn xgboost imblearn matplotlib seaborn plotly kagglehub
```

### Running the Pipeline
1. Install required libraries
2. Authenticate with Kaggle Hub credentials
3. Run the notebook cells sequentially
4. Review visualizations and metrics
5. Compare model performance

## Output

The project generates:
- Data exploration visualizations
- Feature importance plots
- Model performance dashboards
- Confusion matrices and ROC curves
- Prediction probability distributions

## Future Improvements

- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust evaluation
- Feature engineering for domain knowledge
- Ensemble methods (stacking, boosting)
- Model interpretability (SHAP values)
- Threshold optimization for business use cases
