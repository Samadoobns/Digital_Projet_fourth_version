from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import json
import joblib
import shap
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance
from IPython.display import display
# Load data
def load_data(filepath):
    """Load data and perform initial preprocessing"""
    data = pd.read_csv(filepath, sep=';')
    
    # Remove specific columns
    for col in ['l_cr', 'l_axe', 'l_cs']:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)
    
    # Extract target variable
    y = data.pop('Cmoy')
    return data, y

# Feature engineering function
def engineer_features(X):
    """Create new features based on existing ones"""
    X_fe = X.copy()
    
    # Basic features
    X_fe['log_Cmax'] = np.log1p(X_fe['Cmax'])
    X_fe['Cdiff'] = X_fe['Cmax'] - X_fe['Cmin']
    X_fe['C_ratio'] = X_fe['Cmax'] / (X_fe['Cmin'] + 1e-6)  # Avoid division by 0
    X_fe['Csumm'] = X_fe['Cmax'] + X_fe['Cmin']
    X_fe['Cmin_to_max_ratio'] = X_fe['Cmin'] / (X_fe['Cmax'] + 1e-6)  # Rapport inverse

# Power transformations
    X_fe['Cmax_squared'] = X_fe['Cmax']**2
    X_fe['Cmin_squared'] = X_fe['Cmin']**2
    X_fe['Cmax_cubed'] = X_fe['Cmax']**3  # Cube de Cmax
    X_fe['Cmin_cubed'] = X_fe['Cmin']**3  # Cube de Cmin
    X_fe['sqrt_Cmax'] = np.sqrt(X_fe['Cmax'] + 1e-6)  # Racine carrée de Cmax

# Interaction features
    X_fe['Cmax_Cmin_product'] = X_fe['Cmax'] * X_fe['Cmin']
    X_fe['Cdiff_squared'] = X_fe['Cdiff']**2  # Carré de la différence
    X_fe['Csumm_squared'] = X_fe['Csumm']**2  # Carré de la somme

# Trigonometric features
    X_fe['Cmax_sin'] = np.sin(X_fe['Cmax'])  # Sinus de Cmax
    X_fe['Cmin_sin'] = np.sin(X_fe['Cmin'])  # Sinus de Cmin
    X_fe['Cmax_cos'] = np.cos(X_fe['Cmax'])  # Cosinus de Cmax
    X_fe['Cmin_cos'] = np.cos(X_fe['Cmin'])  # Cosinus de Cmin

# Statistical features
    X_fe['C_harmonic_mean'] = 2 / ((1/(X_fe['Cmax'] + 1e-6)) + (1/(X_fe['Cmin'] + 1e-6)))  # Moyenne harmonique
    X_fe['C_mean'] = (X_fe['Cmax'] + X_fe['Cmin']) / 2  # Moyenne arithmétique
    X_fe['C_range_to_mean'] = X_fe['Cdiff'] / (X_fe['C_mean'] + 1e-6)  # Plage normalisée

# Combinatorial features
    X_fe['C_max_div_sum'] = X_fe['Cmax'] / (X_fe['Csumm'] + 1e-6)  # Proportion de Cmax dans la somme
    X_fe['C_min_div_sum'] = X_fe['Cmin'] / (X_fe['Csumm'] + 1e-6)  # Proportion de Cmin dans la somme
    X_fe['log_diff'] = np.log1p(np.abs(X_fe['Cdiff']) + 1e-6)  # Log de la différence absolue
    return X_fe
# Supposons que X_fe contient toutes vos features et y est votre variable cible
# 1. Traitement des valeurs manquantes avant la sélection des features
def handle_missing_values(X_fe):
    # Vérifier s'il y a des valeurs manquantes
    na_count = X_fe.isna().sum()
    if na_count.sum() > 0:
        print("🔍 Colonnes avec valeurs manquantes détectées :")
        print(na_count[na_count > 0])
        
        # Affichage progressif de l'imputation
        print("⚙️ Imputation des valeurs manquantes avec la médiane...")
        tqdm.pandas(desc="Imputation médiane")

        # On impute mais on simule une boucle pour la barre de progression
        imputer = SimpleImputer(strategy='median')
        X_imputed_array = imputer.fit_transform(X_fe)

        # Pour simuler la progression colonne par colonne
        X_imputed = pd.DataFrame(columns=X_fe.columns)
        for i, col in enumerate(tqdm(X_fe.columns, desc="🔄 Reconstruction DataFrame")):
            X_imputed[col] = X_imputed_array[:, i]
        
        print("✅ Valeurs manquantes traitées.")
        return X_imputed
    else:
        print("✅ Aucune valeur manquante détectée.")
        return X_fe
# Add random noise to prevent perfect collinearity
def add_noise(df, scale=0.01):
    """Add small random noise to dataframe"""
    if df.shape[0] > 0:
        return df + np.random.normal(0, scale, df.shape)
    else:
        print("❌ DataFrame is empty")
        return df
# Fonction pour la sélection statistique avec gestion des NaN
def select_features_statistical(X_fe, y, k=15):
    # S'assurer qu'il n'y a pas de valeurs manquantes
    X_fe = handle_missing_values(X_fe)
    
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X_fe, y)
    
    # Récupérer les scores et les noms des features
    scores = selector.scores_
    feature_names = X_fe.columns
    feature_scores = pd.DataFrame({'Feature': feature_names, 'Score': scores})
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    # Afficher les meilleurs features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Score', y='Feature', data=feature_scores.head(k))
    plt.title(f'Top {k} Features Based on F-regression')
    plt.tight_layout()
    plt.show()
    
    selected_features = feature_scores.head(k)['Feature'].tolist()
    return selected_features, X_fe[selected_features]

# Fonction pour la sélection RFE avec gestion des NaN
def select_features_rfe(X_fe, y, k=15):
    # S'assurer qu'il n'y a pas de valeurs manquantes
    X_fe = handle_missing_values(X_fe)
    
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=k, step=1)
    selector.fit(X_fe, y)
    
    # Récupérer les rankings et les noms des features
    feature_names = X_fe.columns
    rankings = selector.ranking_
    feature_ranks = pd.DataFrame({'Feature': feature_names, 'Rank': rankings})
    feature_ranks = feature_ranks.sort_values('Rank')
    
    # Afficher les features sélectionnées
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Rank', y='Feature', data=feature_ranks.head(k))
    plt.title(f'Top {k} Features Based on RFE')
    plt.tight_layout()
    plt.show()
    
    selected_features = feature_ranks.head(k)['Feature'].tolist()
    return selected_features, X_fe[selected_features]

# Fonction pour la sélection par importance avec gestion des NaN
def select_features_importance(X_fe, y, k=15):
    # S'assurer qu'il n'y a pas de valeurs manquantes
    X_fe = handle_missing_values(X_fe)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_fe, y)
    
    # Récupérer les importances et les noms des features
    feature_names = X_fe.columns
    importances = rf.feature_importances_
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values('Importance', ascending=False)
    
    # Afficher les features les plus importantes
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(k))
    plt.title(f'Top {k} Features Based on Random Forest Importance')
    plt.tight_layout()
    plt.show()
    
    selected_features = feature_importances.head(k)['Feature'].tolist()
    return selected_features, X_fe[selected_features]

# Analyse de corrélation entre features avec gestion des NaN
def correlation_analysis(X_fe, threshold=0.9):
    # S'assurer qu'il n'y a pas de valeurs manquantes pour la corrélation
    X_fe = handle_missing_values(X_fe)
    
    corr_matrix = X_fe.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant_features = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Visualiser la matrice de corrélation
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                linewidths=0.5, vmin=0, vmax=1)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.show()
    
    print(f"Features hautement corrélées (r > {threshold}):")
    for col in redundant_features:
        correlated_features = upper.index[upper[col] > threshold].tolist()
        for feat in correlated_features:
            print(f"{col} ↔ {feat}: {upper.loc[feat, col]:.2f}")
    
    return redundant_features
# Train models with hyperparameter tuning
def train_models(X_train, y_train, model_configs):
    """Train models with GridSearchCV"""
    best_models = {}
    scores = {}
    
    for name, config in model_configs.items():
        print(f"Optimizing {name}...")
        
        grid_search = GridSearchCV(
            config["model"], 
            config["params"], 
            cv=5, 
            scoring='r2', 
            n_jobs=-1, 
            verbose=2
        )
        grid_search.fit(X_train, y_train)
        
        # Save best model and score
        best_models[name] = grid_search.best_estimator_
        scores[name] = grid_search.best_score_
        
        # Save model to disk
        model_filename = f"Models/model_{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(grid_search.best_estimator_, model_filename)
        
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best R² validation score: {grid_search.best_score_:.4f}")
        print(f"✅ Model '{name}' saved to: {model_filename}")
        
        # SHAP analysis
        try:
            explainer = shap.Explainer(grid_search.best_estimator_.predict, X_train)
            shap_values = explainer(X_train.iloc[:100])  # Use a subset for speed
            shap.plots.beeswarm(shap_values)
        except Exception as e:
            print(f"⚠️ SHAP analysis failed for {name}: {e}")
    
    # Save scores
    with open("results/scores.json", "w") as f:
        json.dump(scores, f, indent=4)
        print("✅ Scores saved to results/scores.json")
        
    return best_models, scores
# Create preprocessing pipelines
def create_pipelines(models, X_columns):
    """Create scikit-learn pipelines with preprocessing"""
    preprocessor = ColumnTransformer(
        transformers=[('num', SimpleImputer(strategy='mean'), X_columns)]
    )
    
    pipelines = []
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipelines.append((name, pipe))
    
    return pipelines
# Evaluate models on test set
def evaluate_models(pipelines, X_train, y_train, X_test, y_test):
    """Evaluate models and calculate feature importance"""
    scores = {}
    importances = {}
    predictions = pd.DataFrame({'y_true': y_test})
    
    for name, pipeline in pipelines:
        print(f"\n➡️ Training model: {name}")
        
        with tqdm(total=1, desc=f"Fitting {name}", unit="model") as pbar:
            pipeline.fit(X_train, y_train)
            pbar.update(1)
        
        # Make predictions
        preds = pipeline.predict(X_test)
        predictions[name] = preds
        
        # Calculate R² score
        score = r2_score(y_test, preds)
        scores[name] = score
        print(f"✅ Test R² score for {name}: {score:.4f}")
        
        # Calculate feature importance
        try:
            # Get preprocessed data
            X_test_preprocessed = pipeline.named_steps['preprocessor'].transform(X_test)
            
            # Create permutation importance
            perm = PermutationImportance(pipeline.named_steps['model'], random_state=42)
            perm.fit(X_test_preprocessed, y_test)
            
            # Get feature importance weights
            weights = eli5.explain_weights_df(perm, feature_names=X_train.columns.tolist())
            importances[name] = weights
            display(weights.head(10))  # Show top 10 features
        except Exception as e:
            print(f"⚠️ Importance calculation failed for {name}: {e}")
    
    # Save predictions
    predictions.to_csv("results/predictions.csv", index=False)
    print("✅ Predictions saved")
    
    return scores, importances, predictions
# Visualize results
def visualize_results(scores_with_fe, scores_without_fe, importances_with_fe, importances_without_fe):
    """Create visualizations of model performance and feature importance"""
    
    # Sort scores
    sorted_scores_with_fe = dict(sorted(scores_with_fe.items(), key=lambda item: item[1], reverse=True))
    sorted_scores_without_fe = dict(sorted(scores_without_fe.items(), key=lambda item: item[1], reverse=True))
    
    # 1. Plot R² scores
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # With feature engineering
    axes[0].barh(list(sorted_scores_with_fe.keys()), list(sorted_scores_with_fe.values()), color='skyblue')
    axes[0].set_xlabel("R² Score")
    axes[0].set_title("Performance with Feature Engineering")
    axes[0].invert_yaxis()  # Best model on top
    for i, (model, score) in enumerate(sorted_scores_with_fe.items()):
        axes[0].text(score + 0.01, i, f"{score:.4f}", va='center')
    
    # Without feature engineering
    axes[1].barh(list(sorted_scores_without_fe.keys()), list(sorted_scores_without_fe.values()), color='skyblue')
    axes[1].set_xlabel("R² Score")
    axes[1].set_title("Performance without Feature Engineering")
    axes[1].invert_yaxis()  # Best model on top
    for i, (model, score) in enumerate(sorted_scores_without_fe.items()):
        axes[1].text(score + 0.01, i, f"{score:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig("results/model_performance.png")
    
    # 2. Plot feature importance
    def plot_importances(importances, title_suffix):
        # Count models with importance data
        n_models = len(importances)
        
        # Create subplot grid
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 5))
        axes = np.array(axes).flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, importance_df) in enumerate(importances.items()):
            ax = axes[idx]
            
            # Get top features
            top_features = importance_df.head(10)
            ax.barh(top_features['feature'], top_features['weight'], color='lightgreen')
            ax.set_title(f"{model_name}")
            ax.invert_yaxis()  # Most important on top
            
            # Add text labels
            for i, (_, row) in enumerate(top_features.iterrows()):
                ax.text(row['weight'] + 0.01, i, f"{row['weight']:.4f}", va='center')
        
        # Remove empty subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])
        
        fig.suptitle(f"Top 10 Features by Model {title_suffix}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"results/feature_importance_{title_suffix.lower().replace(' ', '_')}.png")
    
    # Plot both sets of importance
    if importances_with_fe:
        plot_importances(importances_with_fe, "with Feature Engineering")
    
    if importances_without_fe:
        plot_importances(importances_without_fe, "without Feature Engineering")
    
    # 3. Create comparison DataFrame
    comparison = pd.DataFrame({
        "Model": list(scores_without_fe.keys()),
        "Score_Without_FE": [scores_without_fe.get(model, np.nan) for model in scores_without_fe.keys()],
        "Score_With_FE": [scores_with_fe.get(model, np.nan) for model in scores_without_fe.keys()]
    })
    
    # Calculate improvement
    comparison["Improvement"] = comparison["Score_With_FE"] - comparison["Score_Without_FE"]
    
    # Save comparison
    comparison.to_csv("results/comparison_scores.csv", index=False)
    print("✅ Comparison saved to results/comparison_scores.csv")
    
    return comparison
