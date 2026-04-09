# Learning Gap Prediction (Student Dropout Analysis)

This repository contains an end-to-end machine learning pipeline designed to predict student outcomes—specifically classifying whether a student is likely to **Graduate**, **Dropout**, or remain **Enrolled**. By identifying at-risk students early, educational institutions can intervene and provide targeted support to close the learning gap.

## Dataset

The project uses a dataset consisting of **4,424 records and 35 initial features**. The target variable (`Target`) consists of three classes:
- **Graduate** (~49.9%)
- **Dropout** (~32.1%)
- **Enrolled** (~17.9%)

Features are categorized into four main buckets:
- **Academic:** Unit performance, credits, evaluations, grades.
- **Demographic:** Age, gender, nationality, marital status, displaced status.
- **Financial:** Debtor status, tuition fee status, scholarship holder.
- **Macroeconomic:** Unemployment rate, inflation rate, GDP.

## Project Pipeline

The workflow is broken down into three main Jupyter notebooks:

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Explores class distributions and identifies class imbalance.
- Analyzes feature relationships and key predictors of student dropouts.
- Groups features into logical buckets (Academic, Demographic, Financial, Macroeconomic).

### 2. Preprocessing & Feature Engineering (`02_preprocess_featureeng.ipynb`)
- **Target Encoding:** Creates both Multi-class labels and Binary labels (Dropout vs. Graduate, with 'Enrolled' held out for pseudo-labeling).
- **Feature Engineering:** Adds powerful derived features:
  - `zero_approved_units_flag`
  - `financial_stress_flag`
  - `semester_performance_delta`
  - `engagement_ratio_1st_sem` & `engagement_ratio_2nd_sem`
- **Scaling:** Standardizes continuous variables for model consumption.

### 3. Model Training & Evaluation (`03_modeltraining.ipynb`)
- **Label Propagation:** Uses KNN on the hard-labeled set to propagate probabilities to 'Enrolled' students, utilizing high-confidence predictions as pseudo-labeled training samples.
- **Handling Imbalance:** Replaces plain SMOTE with **SMOTETomek** for better decision boundaries.
- **Feature Selection:** RF-based `SelectFromModel` step.
- **Hyperparameter Optimization (HPO):** Uses **Optuna** to tune parameters dynamically (including `scale_pos_weight`).
- **Models Evaluated:** Logistic Regression, Random Forest, XGBoost, and LightGBM.
- **Explainability:** Employs **SHAP** for global and local interpretability.

##  Results & Performance

The models were optimized for `f1_macro` with a soft recall floor (≥ 0.85) to ensure at-risk students are reliably identified. 

**Best Model: XGBoost**
- **Recall:** ~0.85 
- **Precision:** ~0.84 
- **F1-macro:** ~0.878
- **Accuracy:** ~0.887

**Top Predictors (SHAP Global Importance):**
1. `Curricular units 2nd sem (approved)`
2. `Curricular units 1st sem (approved)`
3. `Tuition fees up to date`
4. `Age at enrollment`

