# 📦 IMPORTS
# --- General-purpose libraries ---
import os                     # File and directory operations
import pandas as pd            # Data manipulation and analysis
import numpy as np             # Numerical computations

# --- Visualization libraries ---
import seaborn as sns          # Advanced data visualization (heatmaps, boxplots, etc.)
import matplotlib.pyplot as plt  # Plotting library

# --- Scikit-learn: data splitting and preprocessing ---
from sklearn.model_selection import train_test_split  # Split data into train/validation/test sets
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler  # Scaling methods

# --- Scikit-learn: feature selection and regression models ---
from sklearn.feature_selection import RFE             # Recursive Feature Elimination
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV  # Linear, Ridge, and Lasso regressors
from sklearn.ensemble import RandomForestRegressor    # Ensemble-based regressor
from sklearn.tree import DecisionTreeRegressor        # Simple tree-based regressor

# --- Scikit-learn: classification models ---
from sklearn.linear_model import LogisticRegression   # Linear model for classification
from sklearn.tree import DecisionTreeClassifier        # Decision tree classifier
from sklearn.ensemble import RandomForestClassifier    # Random forest classifier

# --- Statistical and diagnostic tools ---
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Variance Inflation Factor (multicollinearity)
from scipy.stats import spearmanr                   # Spearman correlation (non-parametric)

# --- Scikit-learn: feature selection methods ---
from sklearn.feature_selection import mutual_info_regression, f_regression, chi2

# --- Scikit-learn: feature selection with regularization ---
from sklearn.feature_selection import SelectFromModel

# --- Lasso regression (já tem LassoCV, mas precisa do Lasso básico) ---
from sklearn.linear_model import Lasso

# --- Visualization theme ---
sns.set_theme(style="whitegrid", context="notebook")

class NumericalFeatureSelector:
    """
    Classe para seleção de features numéricas em problemas de regressão.
    Usa apenas dados de treino para evitar data leakage.
    """

    def __init__(self, X_train, y_train, numeric_features, X_val=None, y_val=None, vif_threshold=5, corr_threshold=0.7):
        self.X_train = X_train[numeric_features].copy()
        self.y_train = y_train.copy()
        self.X_val = X_val[numeric_features].copy() if X_val is not None else None
        self.y_val = y_val.copy() if y_val is not None else None
        self.numeric_features = numeric_features
        self.vif_threshold = vif_threshold
        self.corr_threshold = corr_threshold

    # Multicolinearidade / Redundância
    def vif_analysis(self):
        X = self.X_train.dropna().copy()
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_data["Accepted"] = vif_data["VIF"] < self.vif_threshold
        return vif_data

    def spearman_redundancy(self):
        corr = self.X_train.corr(method='spearman').abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        redundancy_df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "Max_SpearmanCorr": [upper[col].max(skipna=True) for col in self.X_train.columns],
        })
        redundancy_df["Accepted"] = redundancy_df["Max_SpearmanCorr"] < self.corr_threshold
        return redundancy_df

    # Correlação com o Target (Relevância)
    def spearman_relevance(self, threshold=0.1):
        corr_values = []
        for col in self.X_train.columns:
            corr, _ = spearmanr(self.X_train[col], self.y_train)
            corr_values.append(abs(corr))
        corr_df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "Spearman_TargetCorr": corr_values
        })
        corr_df["Accepted"] = corr_df["Spearman_TargetCorr"] > threshold
        return corr_df

    # Recursive Feature Elimination (RFE)
    def rfe_model(self, model, scaler=None):
        X = self.X_train.copy()
        if scaler:
            X = scaler.fit_transform(X)
        rfe = RFE(model)
        rfe.fit(X, self.y_train)
        results = pd.DataFrame({"Feature": self.X_train.columns, "Accepted": rfe.support_})
        return results

    def rfe_all_models(self):
        models = {
            "RFE_DecisionTree": DecisionTreeRegressor(random_state=42),
            "RFE_RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
            "RFE_LR_MinMax": (LinearRegression(), MinMaxScaler()),
            "RFE_LR_Standard": (LinearRegression(), StandardScaler()),
            "RFE_LR_Robust": (LinearRegression(), RobustScaler())
        }
        results = []
        for name, model in models.items():
            df = self.rfe_model(model[0], model[1]) if isinstance(model, tuple) else self.rfe_model(model)
            df = df.rename(columns={"Accepted": name})
            df = df[["Feature", name]]
            results.append(df)
        return results

    # Regularização Ridge/Lasso
    def regularization_model(self, model_type="ridge", scaler=None, threshold=0.01):
        X = self.X_train.copy()
        if scaler:
            X = scaler.fit_transform(X)
        model = RidgeCV(alphas=np.logspace(-3, 3, 50)) if model_type == "ridge" else LassoCV(alphas=np.logspace(-3, 3, 50), max_iter=10000)
        model.fit(X, self.y_train)
        coefs = np.abs(model.coef_)
        df = pd.DataFrame({
            "Feature": self.X_train.columns,
            f"{model_type.capitalize()}_Coef": coefs,
            "Accepted": coefs > threshold
        })
        return df

    def ridge_all(self):
        scalers = {"Ridge_MinMax": MinMaxScaler(), "Ridge_Standard": StandardScaler(), "Ridge_Robust": RobustScaler()}
        results = []
        for name, scaler in scalers.items():
            df = self.regularization_model("ridge", scaler)
            df = df.rename(columns={"Accepted": name})
            df = df[["Feature", name]]
            results.append(df)
        return results

    def lasso_all(self):
        scalers = {"Lasso_MinMax": MinMaxScaler(), "Lasso_Standard": StandardScaler(), "Lasso_Robust": RobustScaler()}
        results = []
        for name, scaler in scalers.items():
            df = self.regularization_model("lasso", scaler)
            df = df.rename(columns={"Accepted": name})
            df = df[["Feature", name]]
            results.append(df)
        return results

    # Tabela Final
    def compile_results(self):
        results = [
            self.vif_analysis(),
            self.spearman_redundancy(),
            self.spearman_relevance(),
            *self.rfe_all_models(),
            *self.ridge_all(),
            *self.lasso_all()
        ]
        merged = results[0][["Feature"]]
        for df in results:
            merged = merged.merge(df, on="Feature", how="left")

        accept_cols = [c for c in merged.columns if "RFE_" in c or "Ridge_" in c or "Lasso_" in c or "Accepted" in c]
        merged["Total_Accepted"] = merged[accept_cols].sum(axis=1)
        merged["Final_Decision"] = np.where(merged["Total_Accepted"] > len(accept_cols) / 2, "Keep", "Drop")
        return merged


class CategoricalFeatureSelector:

    def __init__(self, X_train, y_train, categorical_encoded_features, X_val=None, y_val=None, 
                 corr_threshold=0.8, importance_threshold=0.01):
        self.X_train = X_train[categorical_encoded_features].copy()
        self.y_train = y_train.copy()
        self.X_val = X_val[categorical_encoded_features].copy() if X_val is not None else None
        self.y_val = y_val.copy() if y_val is not None else None
        self.categorical_encoded_features = categorical_encoded_features
        self.corr_threshold = corr_threshold
        self.importance_threshold = importance_threshold

    # 1️⃣ Multicolinearidade / Redundância (Para Dummy Variables)
    def correlation_redundancy(self):
        """Detecta features dummy altamente correlacionadas"""
        corr = self.X_train.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        
        redundancy_df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "Max_PearsonCorr": [upper[col].max(skipna=True) for col in self.X_train.columns],
        })
        redundancy_df["Accepted"] = redundancy_df["Max_PearsonCorr"] < self.corr_threshold
        return redundancy_df

    # 2️⃣ Relevância com o Target
    def mutual_information_relevance(self, threshold=0.01):
        """Mutual Information é ideal para relações não-lineares com categóricas"""
        mi_scores = mutual_info_regression(self.X_train, self.y_train, random_state=42)
        
        mi_df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "Mutual_Information": mi_scores
        })
        mi_df["Accepted"] = mi_df["Mutual_Information"] > threshold
        return mi_df

    def anova_relevance(self, threshold=0.05):
        """ANOVA F-test para relevância estatística"""
        f_scores, p_values = f_regression(self.X_train, self.y_train)
        
        anova_df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "F_Score": f_scores,
            "P_Value": p_values
        })
        anova_df["Accepted"] = anova_df["P_Value"] < threshold
        return anova_df

    # 3️⃣ Recursive Feature Elimination (RFE)
    def rfe_model(self, model, scaler=None):
        X = self.X_train.copy()
        if scaler:
            X = scaler.fit_transform(X)
        rfe = RFE(model)
        rfe.fit(X, self.y_train)
        results = pd.DataFrame({"Feature": self.X_train.columns, "Accepted": rfe.support_})
        return results

    def rfe_all_models(self):
        models = {
            "RFE_DecisionTree": DecisionTreeRegressor(random_state=42),
            "RFE_RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
            "RFE_LR_MinMax": (LinearRegression(), MinMaxScaler()),
            "RFE_LR_Standard": (LinearRegression(), StandardScaler()),
            "RFE_LR_Robust": (LinearRegression(), RobustScaler())
        }
        results = []
        for name, model in models.items():
            df = self.rfe_model(model[0], model[1]) if isinstance(model, tuple) else self.rfe_model(model)
            df = df.rename(columns={"Accepted": name})
            df = df[["Feature", name]]
            results.append(df)
        return results

    # 4️⃣ Regularização Lasso (Ideal para Dummy Variables)
    def lasso_model(self, scaler=None, threshold=0.01):
        """Lasso é excelente para seleção de features dummy"""
        X = self.X_train.copy()
        if scaler:
            X = scaler.fit_transform(X)
        
        lasso = LassoCV(alphas=np.logspace(-3, 3, 50), max_iter=10000, random_state=42)
        lasso.fit(X, self.y_train)
        
        coefs = np.abs(lasso.coef_)
        df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "Lasso_Coef": coefs,
            "Accepted": coefs > threshold
        })
        return df

    def lasso_all_scalers(self):
        scalers = {
            "Lasso_MinMax": MinMaxScaler(),
            "Lasso_Standard": StandardScaler(), 
            "Lasso_Robust": RobustScaler()
        }
        results = []
        for name, scaler in scalers.items():
            df = self.lasso_model(scaler)
            df = df.rename(columns={"Accepted": name})
            df = df[["Feature", name]]
            results.append(df)
        return results

    # 5️⃣ Feature Importance com Random Forest
    def random_forest_importance(self, threshold=0.01):
        """Feature importance nativo do Random Forest"""
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        
        importance_df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "RF_Importance": rf.feature_importances_
        })
        importance_df["Accepted"] = importance_df["RF_Importance"] > threshold
        return importance_df

    # 6️⃣ Tabela Final Consolidada
    def compile_results(self):
        # Coletar todos os resultados
        results = [
            self.correlation_redundancy()[["Feature", "Accepted"]].rename(columns={"Accepted": "Low_Correlation"}),
            self.mutual_information_relevance()[["Feature", "Accepted"]].rename(columns={"Accepted": "High_MI"}),
            self.anova_relevance()[["Feature", "Accepted"]].rename(columns={"Accepted": "ANOVA_Sig"}),
            self.random_forest_importance()[["Feature", "Accepted"]].rename(columns={"Accepted": "RF_Important"}),
            *self.rfe_all_models(),
            *self.lasso_all_scalers()
        ]

        # Merge todos os resultados
        merged = results[0][["Feature"]]
        for df in results:
            merged = merged.merge(df, on="Feature", how="left")

        # Calcular decisão final
        accept_cols = [c for c in merged.columns if c != "Feature"]
        merged["Total_Accepted"] = merged[accept_cols].sum(axis=1)
        merged["Accept_Rate"] = merged["Total_Accepted"] / len(accept_cols)
        merged["Final_Decision"] = np.where(merged["Accept_Rate"] > 0.6, "Keep", "Drop")
        
        return merged

    # 7️⃣ Método para obter features selecionadas
    def get_selected_features(self):
        """Retorna lista das features selecionadas"""
        results = self.compile_results()
        selected = results[results["Final_Decision"] == "Keep"]["Feature"].tolist()
        
        print(f"🎯 Selected {len(selected)} out of {len(self.categorical_encoded_features)} encoded categorical features")
        print(f"📊 Selection rate: {len(selected)/len(self.categorical_encoded_features):.1%}")
        
        return selected

    # 8️⃣ Plot de Importâncias
    def plot_feature_importance(self, top_n=15):
        """Plot das importâncias das features"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Mutual Information
        mi_df = self.mutual_information_relevance()
        mi_df.sort_values('Mutual_Information', ascending=True).tail(top_n).plot(
            kind='barh', x='Feature', y='Mutual_Information', ax=axes[0,0], 
            color='lightblue', title='Mutual Information Importance'
        )
        
        # ANOVA F-scores
        anova_df = self.anova_relevance()
        anova_df.sort_values('F_Score', ascending=True).tail(top_n).plot(
            kind='barh', x='Feature', y='F_Score', ax=axes[0,1],
            color='lightcoral', title='ANOVA F-Scores'
        )
        
        # Random Forest Importance
        rf_df = self.random_forest_importance()
        rf_df.sort_values('RF_Importance', ascending=True).tail(top_n).plot(
            kind='barh', x='Feature', y='RF_Importance', ax=axes[1,0],
            color='lightgreen', title='Random Forest Importance'
        )
        
        # Lasso Coefficients
        lasso_df = self.lasso_model()
        lasso_df.sort_values('Lasso_Coef', ascending=True).tail(top_n).plot(
            kind='barh', x='Feature', y='Lasso_Coef', ax=axes[1,1],
            color='orange', title='Lasso Coefficients'
        )
        
        plt.tight_layout()
        plt.show()