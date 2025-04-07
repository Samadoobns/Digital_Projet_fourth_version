from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.impute import SimpleImputer
import os
import pandas as pd
from all_functions import *
# outputs
os.makedirs("results", exist_ok=True)
os.makedirs("Models", exist_ok=True)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  

# data paths 
train_data_path = 'Dataset_numerique_20000_petites_machines.csv'  
test_data_path = 'Dataset_numerique_10000_petites_machines.csv'  


# Obtenir le dataset final avec les features sélectionnées
# Load and preprocess training data
X_raw, y = load_data(train_data_path)
X_without_fe = X_raw.copy()  # Save original features
X = engineer_features(X_raw)  # Apply feature engineering


# Load and preprocess test data
X_test_raw, y_test = load_data(test_data_path)
X_test_without_fe = X_test_raw.copy()
X_test = engineer_features(X_test_raw)
X_fe = X.copy()

# Utilisation
k = 30  # Nombre de features à sélectionner
# Exécuter les différentes méthodes de sélection
print("Analyse statistique (SelectKBest):")
statistical_features, X_statistical = select_features_statistical(X_fe, y, k)

'''print("\nSélection récursive (RFE):")
rfe_features, X_rfe = select_features_rfe(X_fe, y, k)'''

'''print("\nImportance des features (Random Forest):")
importance_features, X_importance = select_features_importance(X_fe, y, k)'''

print("\nAnalyse de corrélation:")
redundant_features = correlation_analysis(X_fe, threshold=0.9)

# Comparaison des méthodes
all_selected = set(statistical_features)
#common_features = set(statistical_features)

print(f"\nNombre total de features uniques sélectionnées: {len(all_selected)}")
#print(f"Features communes à toutes les méthodes: {len(common_features)}")
#print("Features communes:", common_features)

# Créer un ensemble final de features en excluant les redondantes
final_features = [f for f in all_selected if f not in redundant_features]
print(f"\nNombre final de features recommandées: {len(final_features)}")
print("Features recommandées:", final_features)
X = X[final_features]
X_test = X_test[final_features]


# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
X_without_fe = pd.DataFrame(imputer.fit_transform(X_without_fe), columns=X_without_fe.columns, index=X_without_fe.index)
X_test_without_fe = pd.DataFrame(imputer.transform(X_test_without_fe), columns=X_test_without_fe.columns, index=X_test_without_fe.index)

# Print dataset shapes
print("Training data with FE:", X.shape)
print("Training data without FE:", X_without_fe.shape)
print("Test data with FE:", X_test.shape)
print("Test data without FE:", X_test_without_fe.shape)



X_without_fe = add_noise(X_without_fe)
X_test_without_fe = add_noise(X_test_without_fe)
X = add_noise(X)
X_test = add_noise(X_test)

# Define models to test
regressors = {
    "Ridge": {
        "model": Ridge(),
        "params": {
            "alpha": [0.01, 0.1, 1.0, 10.0]
        }
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=0),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=0),
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    },
    "HistGradient Boosting": {
        "model": HistGradientBoostingRegressor(random_state=0),
        "params": {
            "max_iter": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "Extra Trees": {
        "model": ExtraTreesRegressor(random_state=0),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=0),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0]
        }
    },
    
    "LightGBM": {
        "model": LGBMRegressor(random_state=0),
        "params": {
            "n_estimators": [50, 100],
            "num_leaves": [31, 63],
            "learning_rate": [0.01, 0.1],
            "feature_fraction": [0.8, 0.9]
        }
    },
        "Lasso": {
        "model": Lasso(max_iter=50000,tol=0.01),
        "params": {
            "alpha": [0.0001, 0.001, 0.01]
        }
    }
}


# Train models with feature engineering
best_models, scores = train_models(X, y, regressors)


# Create pipelines for both feature sets
pipelines_with_fe = create_pipelines(best_models, X.columns)
pipelines_without_fe = create_pipelines(best_models, X_without_fe.columns)


# Evaluate models with feature engineering
print("\n=== Evaluating models with feature engineering ===")
scores_with_fe, importances_with_fe, preds_with_fe = evaluate_models(
    pipelines_with_fe, X, y, X_test, y_test
)

# Evaluate models without feature engineering
print("\n=== Evaluating models without feature engineering ===")
scores_without_fe, importances_without_fe, preds_without_fe = evaluate_models(
    pipelines_without_fe, X_without_fe, y, X_test_without_fe, y_test
)


# Visualize all results
comparison = visualize_results(
    scores_with_fe, 
    scores_without_fe,
    importances_with_fe,
    importances_without_fe
)

print("\n=== Summary ===")
print(comparison)
print("\n✅ Analysis complete! All results saved to 'results' directory.")



















