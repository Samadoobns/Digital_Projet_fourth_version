# ⚙️ Digital Project – Machine Learning for Electric Motor Torque Prediction - 

This project aims to **predict the electromagnetic torque (Cmoy)** of small electric motors using **supervised machine learning techniques**. It leverages experimental datasets and implements a **complete pipeline**: **data preprocessing, feature engineering, feature selection, model training, and evaluation**. This is an **improved version** of the original project, featuring enhanced models, refined feature selection methods, and more robust evaluation metrics.

---

## 📁 Project Structure

```
.
├── Dataset_numerique_20000_petites_machines.csv      # Training dataset
├── Dataset_numerique_10000_petites_machines.csv      # Test dataset
├── all_functions.py                                  # Utility functions
├── main_script.py                                    # Main script
├── results/                                          # Outputs and plots
├── Models/                                           # Trained models
└── README.md                                         # This file
```

---

## 🎯 Objectives

- Clean and preprocess raw data
- Select the most relevant features using multiple methods
- Train and evaluate several regression models
- Compare model performance with and without feature engineering
- Visualize scores and feature importances

---

## ⚙️ How the Main Script Works

### 1. Load Data
Training and test datasets are loaded using a custom `load_data()` function. Each dataset includes measurements from small electric motors.

### 2. Feature Engineering
The `engineer_features()` function creates meaningful features from the raw dataset.

### 3. Feature Selection

- **Statistical selection** using SelectKBest
- **Correlation analysis** to remove redundant features
- (Optionally: Recursive Feature Elimination, Random Forest importance)

### 4. Missing Value Imputation
Missing values are filled using `SimpleImputer` with a mean strategy to maintain dimensional consistency.

### 5. Noise Injection
Small Gaussian noise is added to simulate real-world uncertainty and improve generalization.

### 6. Models Trained

The following models are trained using GridSearchCV for hyperparameter tuning:
- Ridge Regression
- Lasso Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- XGBoost
- LightGBM

### 7. Evaluation
Models are evaluated on both **feature-engineered** and **raw** datasets using R² score and feature importance metrics.

### 8. Visualization
Comparison of models is saved in the `results/` directory for analysis and interpretation.

---

## 📊 Output

- ✅ R² scores with and without feature engineering
- 📈 Visual feature importances
- 📁 Saved best pipelines
- 📂 Everything stored in `results/` and `Models/` folders

---

## 📌 Notes

- Set the environment variable `TF_ENABLE_ONEDNN_OPTS=0` to ensure compatibility with TensorFlow-based backends.
- The project is modular and easy to expand for more datasets or models.

---

## 🧠 Skills Involved

- Machine Learning (Sklearn, XGBoost, LightGBM)
- Feature Engineering & Selection
- Hyperparameter Tuning
- Model Evaluation
- Pipeline Design

---

## 🏁 Final Message

> All analysis is complete! Results are automatically saved in the `results/` directory. Happy modeling!
