import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, PowerTransformer, OneHotEncoder

import utils


@dataclass
class FittedParams:
    """Container para todos os parâmetros fitted no treino."""
    
    # Brand/model reference
    valid_brands: List[str] = field(default_factory=list)
    all_models: List[str] = field(default_factory=list)
    models_by_brand: Dict[str, List[str]] = field(default_factory=dict)
    brands_by_model: Dict[str, List[str]] = field(default_factory=dict)
    unique_brand_by_model: Dict[str, str] = field(default_factory=dict)
    
    # Manual fixes
    manual_model_fixes: Dict[tuple, str] = field(default_factory=dict)
    manual_model_corrections: Dict[str, Dict[str, str]] = field(default_factory=dict)
    k_neighbors: int = 5
    
    # Categorical valid values
    transmission_uniques: List[str] = field(default_factory=list)
    fueltype_uniques: List[str] = field(default_factory=list)
    
    # Simple imputation (MCAR)
    brand_mode: str = None
    model_mode_per_brand: Dict[str, str] = field(default_factory=dict)
    model_mode_global: str = None
    
    # MAR variables
    mar_vars: List[str] = field(default_factory=list)
    
    # MICE imputation
    mice_imputer: Any = None
    mice_scaler: Any = None
    numeric_cols: List[str] = field(default_factory=list)
    
    # Rare models
    keep_models_table: pd.DataFrame = None
    
    # Feature engineering
    brand_stats: pd.DataFrame = None
    mileage_q25: float = None
    mileage_q75: float = None
    model_counts: Dict[str, int] = field(default_factory=dict)
    
    # Encoding
    target_encodings: Dict[str, Dict[str, float]] = field(default_factory=dict)
    global_mean_price: float = None
    ohe_encoder: Any = None
    low_cardinality_cols: List[str] = field(default_factory=list)
    
    # Final scaler
    final_scaler: Any = None
    final_feature_names: List[str] = field(default_factory=list)


class CarPreprocessingPipeline:
    
    def __init__(self, reference_year: int = 2020):
        self.reference_year = reference_year
        self.params = FittedParams()
        self.is_fitted = False
        # NOVO: Guardar X_train transformado
        self._X_train_transformed = None
        # NOVO: Guardar carIDs originais do treino
        self._original_train_ids = None
    
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
            brand_model_dic: pd.DataFrame,
            manual_model_fixes: Dict[tuple, str] = None,
            manual_model_corrections: Dict[str, Dict[str, str]] = None) -> 'CarPreprocessingPipeline':
        
        X = X_train.copy()
        y = y_train.copy()
        
        # =====================================================================
        # PRESERVAR carID - extrair antes de qualquer transformação
        # =====================================================================
        if 'carID' in X.columns:
            self._original_train_ids = X['carID'].values.copy()
            X = X.drop(columns=['carID'])
        elif X.index.name == 'carID':
            self._original_train_ids = X.index.values.copy()
            X = X.reset_index(drop=True)
        else:
            self._original_train_ids = None
        
        self.params.manual_model_fixes = manual_model_fixes or {}
        self.params.manual_model_corrections = manual_model_corrections or {}
        
        # -----------------------------------------------------------------
        # 1. Preparar dicionário de referência
        # -----------------------------------------------------------------
        print("\n[1/13] Preparando dicionário de referência...")
        self._fit_brand_model_reference(brand_model_dic)
        
        # -----------------------------------------------------------------
        # 2. Normalizar Brand/model
        # -----------------------------------------------------------------
        print("[2/13] Normalizando Brand e model...")
        X['Brand'] = X['Brand'].apply(utils.norm)
        X['model'] = X['model'].apply(utils.norm)
        
        # -----------------------------------------------------------------
        # 3. Limpar Brand/model
        # -----------------------------------------------------------------
        print("[3/13] Limpando Brand e model...")
        X = self._clean_brand_model(X)
        
        # -----------------------------------------------------------------
        # 4. Imputar model por engineSize
        # -----------------------------------------------------------------
        print("[4/13] Imputando models por engineSize...")
        X = self._impute_model_by_engine_size(X)
        
        # -----------------------------------------------------------------
        # 5. Forçar missing e aplicar correções
        # -----------------------------------------------------------------
        print("[5/13] Forçando missing para needs_review...")
        mask = X['bm_status'] == 'needs_review'
        X.loc[mask, 'model'] = None
        X = X.drop(columns=['bm_status', 'bm_note'], errors='ignore')
        
        # -----------------------------------------------------------------
        # 6. Limpar variáveis numéricas/categóricas
        # -----------------------------------------------------------------
        print("[6/13] Limpando variáveis numéricas e categóricas...")
        X = self._clean_numeric_categorical(X)
        
        # -----------------------------------------------------------------
        # 7. FIT imputação simples (MCAR) - CONDICIONAL POR BRAND
        # -----------------------------------------------------------------
        print("[7/13] Calculando modas para imputação MCAR (model por Brand)...")
        self.params.brand_mode = X['Brand'].mode().iloc[0] if not X['Brand'].mode().empty else 'unknown'
        self.params.model_mode_global = X['model'].mode().iloc[0] if not X['model'].mode().empty else 'unknown'
        
        # IMPUTAÇÃO CONDICIONAL: moda do model POR Brand
        self.params.model_mode_per_brand = (
            X.groupby('Brand')['model']
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown')
            .to_dict()
        )
        
        # Aplicar imputação
        X['Brand'] = X['Brand'].fillna(self.params.brand_mode)
        
        # Imputar model usando a moda DENTRO de cada Brand
        missing_mask = X['model'].isna()
        X.loc[missing_mask, 'model'] = X.loc[missing_mask, 'Brand'].map(self.params.model_mode_per_brand)
        
        # Fallback para model_mode_global se Brand não existir no dict
        still_missing = X['model'].isna()
        X.loc[still_missing, 'model'] = self.params.model_mode_global
        
        # -----------------------------------------------------------------
        # 8. Identificar MAR vars e criar flags
        # -----------------------------------------------------------------
        print("[8/13] Identificando variáveis MAR...")
        self.params.mar_vars = ['mpg', 'tax', 'engineSize', 'year']
        
        for var in self.params.mar_vars:
            X[f"{var}_is_missing"] = X[var].isna().astype(int)
        
        # -----------------------------------------------------------------
        # 9. FIT MICE imputation
        # -----------------------------------------------------------------
        print("[9/13] Fitting MICE imputer...")
        self.params.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.params.mice_imputer = IterativeImputer(max_iter=10, random_state=37)
        self.params.mice_scaler = RobustScaler()
        
        data_numeric = X[self.params.numeric_cols].copy()
        data_imputed = self.params.mice_imputer.fit_transform(data_numeric)
        data_scaled = self.params.mice_scaler.fit_transform(data_imputed)
        
        X[self.params.numeric_cols] = pd.DataFrame(
            data_scaled, index=X.index, columns=self.params.numeric_cols
        )
        
        # -----------------------------------------------------------------
        # 10. FIT rare models collapse
        # -----------------------------------------------------------------
        print("[10/13] Calculando tabela de modelos frequentes...")
        self.params.keep_models_table = self._fit_keep_models_per_brand(X)
        X = self._collapse_rare_models(X)
        
        # Inverse transform para feature engineering
        X[self.params.numeric_cols] = self.params.mice_scaler.inverse_transform(
            X[self.params.numeric_cols]
        )
        
        # Arredondar variáveis discretas
        for col in ['year', 'previousOwners']:
            if col in X.columns:
                X[col] = X[col].round().astype(int, errors='ignore')
        
        # -----------------------------------------------------------------
        # 11. FIT feature engineering params
        # -----------------------------------------------------------------
        print("[11/13] Calculando parâmetros de feature engineering...")
        
        # Brand stats
        y_price = y['price'] if isinstance(y, pd.DataFrame) else y
        self.params.brand_stats = (
            X.join(y_price)
            .groupby('Brand')['price']
            .agg(['mean', 'median', 'std', 'count'])
            .round(2)
            .rename(columns={
                'mean': 'brand_mean_price',
                'median': 'brand_median_price',
                'std': 'brand_price_std',
                'count': 'brand_count'
            })
        )
        
        p33, p67 = self.params.brand_stats['brand_mean_price'].quantile([0.33, 0.67])
        self.params.brand_stats['brand_segment'] = pd.cut(
            self.params.brand_stats['brand_mean_price'],
            bins=[-np.inf, p33, p67, np.inf],
            labels=['budget', 'mid_range', 'luxury']
        )
        
        self.params.mileage_q25 = X['mileage'].quantile(0.25)
        self.params.mileage_q75 = X['mileage'].quantile(0.75)
        self.params.model_counts = X['model'].value_counts().to_dict()
        
        # Aplicar feature engineering
        X = self._apply_feature_engineering(X)
        
        # -----------------------------------------------------------------
        # 12. FIT encoding
        # -----------------------------------------------------------------
        print("[12/13] Fitting encoders...")
        
        # Target encoding
        high_cardinality_cols = ['Brand', 'brand_model']
        self.params.global_mean_price = y_price.mean()
        self.params.target_encodings = {}
        
        for col in high_cardinality_cols:
            if col in X.columns:
                self.params.target_encodings[col] = (
                    y_price.groupby(X[col]).mean().to_dict()
                )
                X[f'{col}_target_enc'] = X[col].map(self.params.target_encodings[col])
        
        # CORREÇÃO: Remover colunas high cardinality ANTES do OHE
        X = X.drop(columns=high_cardinality_cols, errors='ignore')
        
        # OHE
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.params.low_cardinality_cols = [c for c in categorical_cols if c not in high_cardinality_cols]
        
        if self.params.low_cardinality_cols:
            self.params.ohe_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            self.params.ohe_encoder.fit(X[self.params.low_cardinality_cols])
            
            ohe_transformed = pd.DataFrame(
                self.params.ohe_encoder.transform(X[self.params.low_cardinality_cols]),
                columns=self.params.ohe_encoder.get_feature_names_out(),
                index=X.index
            )
            X = X.drop(columns=self.params.low_cardinality_cols)
            X = pd.concat([X, ohe_transformed], axis=1)
        
        # CORREÇÃO: Drop TODAS as colunas que possam ainda ser strings
        cols_to_drop = ['transmission', 'fuelType', 'brand_model', 'model_popularity', 
                        'brand_segment', 'Brand', 'model', 'year']
        X = X.drop(columns=cols_to_drop, errors="ignore")
        
        # CORREÇÃO CRÍTICA: Remover QUALQUER coluna que ainda seja string/object
        remaining_string_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if remaining_string_cols:
            print(f"  ⚠️ Removendo colunas string restantes: {remaining_string_cols}")
            X = X.drop(columns=remaining_string_cols, errors='ignore')
        
        # -----------------------------------------------------------------
        # 13. FIT final scaler
        # -----------------------------------------------------------------
        print("[13/13] Fitting final scaler...")
        
        # Verificação final - só colunas numéricas
        assert X.select_dtypes(include=['object', 'category']).columns.tolist() == [], \
            f"Ainda existem colunas não-numéricas: {X.select_dtypes(include=['object', 'category']).columns.tolist()}"
        
        self.params.final_scaler = PowerTransformer()
        self.params.final_feature_names = X.columns.tolist()
        
        X = pd.DataFrame(
            self.params.final_scaler.fit_transform(X),
            columns=self.params.final_feature_names,
            index=X.index
        )
        
        # =====================================================================
        # RESTAURAR carID como índice
        # =====================================================================
        if self._original_train_ids is not None:
            X.index = self._original_train_ids
            X.index.name = 'carID'
        
        self.is_fitted = True
        
        # NOVO: Guardar X_train transformado
        self._X_train_transformed = X.copy()
        
        print(f"\n✓ Pipeline fitted! Final shape: {X.shape}")
        print(f"  Features: {len(self.params.final_feature_names)}")
        print(f"  ✓ X_train transformado guardado internamente (use get_transformed_train())")
        
        return self
    
    # =========================================================================
    # NOVO: Método para obter X_train transformado
    # =========================================================================
    
    def get_transformed_train(self) -> pd.DataFrame:
        """Retorna X_train transformado após fit().
        
        Returns
        -------
        pd.DataFrame
            X_train com todas as transformações aplicadas
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline não fitted! Chama fit() primeiro.")
        if self._X_train_transformed is None:
            raise RuntimeError("X_train transformado não disponível.")
        return self._X_train_transformed.copy()
    
    # =========================================================================
    # TRANSFORM
    # =========================================================================
    
    def transform(self, X: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Pipeline não fitted! Chama fit() primeiro.")
        
        print(f"\n{'='*60}")
        print(f"TRANSFORMANDO {dataset_name.upper()} ({X.shape[0]} rows)")
        print(f"{'='*60}")
        
        df = X.copy()
        
        # =====================================================================
        # PRESERVAR carID - extrair antes de qualquer transformação
        # =====================================================================
        if 'carID' in df.columns:
            original_car_ids = df['carID'].values.copy()
            df = df.drop(columns=['carID'])
        elif df.index.name == 'carID':
            original_car_ids = df.index.values.copy()
            df = df.reset_index(drop=True)
        else:
            original_car_ids = None
        
        # 1. Normalizar Brand/model
        print("\n[1/11] Normalizando Brand e model...")
        df['Brand'] = df['Brand'].apply(utils.norm)
        df['model'] = df['model'].apply(utils.norm)
        
        # 2. Limpar Brand/model
        print("[2/11] Limpando Brand e model...")
        df = self._clean_brand_model(df)
        
        # 3. Imputar model por engineSize
        print("[3/11] Imputando models por engineSize...")
        df = self._impute_model_by_engine_size(df)
        
        # 4. Forçar missing e correções manuais
        print("[4/11] Aplicando correções manuais...")
        mask = df['bm_status'] == 'needs_review'
        df.loc[mask, 'model'] = None
        
        # Correções manuais específicas por dataset
        if dataset_name in self.params.manual_model_corrections:
            for old_model, new_model in self.params.manual_model_corrections[dataset_name].items():
                df.loc[df['model'] == old_model, 'model'] = new_model
        
        df = df.drop(columns=['bm_status', 'bm_note'], errors='ignore')
        
        # 5. Limpar variáveis
        print("[5/11] Limpando variáveis numéricas e categóricas...")
        df = self._clean_numeric_categorical(df)
        
        # 6. Imputação simples (MCAR) - CONDICIONAL POR BRAND
        print("[6/11] Imputando Brand e model (MCAR condicional)...")
        df['Brand'] = df['Brand'].fillna(self.params.brand_mode)
        
        # Imputar model usando a moda DENTRO de cada Brand
        missing_mask = df['model'].isna()
        df.loc[missing_mask, 'model'] = df.loc[missing_mask, 'Brand'].map(self.params.model_mode_per_brand)
        
        # Fallback para model_mode_global
        still_missing = df['model'].isna()
        df.loc[still_missing, 'model'] = self.params.model_mode_global
        
        # 7. Flags de missing (MAR)
        print("[7/11] Criando flags de missing...")
        for var in self.params.mar_vars:
            df[f"{var}_is_missing"] = df[var].isna().astype(int)
        
        # 8. MICE imputation
        print("[8/11] Aplicando MICE imputation...")
        # Garantir que temos as mesmas colunas numéricas
        for col in self.params.numeric_cols:
            if col not in df.columns:
                df[col] = 0
        
        data_numeric = df[self.params.numeric_cols].copy()
        data_imputed = self.params.mice_imputer.transform(data_numeric)
        data_scaled = self.params.mice_scaler.transform(data_imputed)
        df[self.params.numeric_cols] = pd.DataFrame(
            data_scaled, index=df.index, columns=self.params.numeric_cols
        )
        
        # 9. Collapse rare models
        print("[9/11] Colapsando modelos raros...")
        df = self._collapse_rare_models(df)
        
        # Inverse transform
        df[self.params.numeric_cols] = self.params.mice_scaler.inverse_transform(
            df[self.params.numeric_cols]
        )
        for col in ['year', 'previousOwners']:
            if col in df.columns:
                df[col] = df[col].round().astype(int, errors='ignore')
        
        # 10. Feature engineering
        print("[10/11] Criando features...")
        df = self._apply_feature_engineering(df)
        
        # 11. Encoding e scaling final
        print("[11/11] Aplicando encoding e scaling final...")
        df = self._apply_encoding_and_scaling(df)
        
        # =====================================================================
        # RESTAURAR carID como índice
        # =====================================================================
        if original_car_ids is not None:
            df.index = original_car_ids
            df.index.name = 'carID'
        
        print(f"\n✓ Transformação concluída! Shape final: {df.shape}")
        return df
    
    # =========================================================================
    # FIT_TRANSFORM (convenience method)
    # =========================================================================
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                      brand_model_dic: pd.DataFrame,
                      manual_model_fixes: Dict[tuple, str] = None,
                      manual_model_corrections: Dict[str, Dict[str, str]] = None) -> pd.DataFrame:
        """Fit e transform num único passo.
        
        Returns
        -------
        pd.DataFrame
            X_train transformado
        """
        self.fit(X_train, y_train, brand_model_dic, manual_model_fixes, manual_model_corrections)
        # CORREÇÃO: Retornar o X_train que foi guardado durante o fit
        return self.get_transformed_train()
    
    # =========================================================================
    # MÉTODOS PRIVADOS
    # =========================================================================
    
    def _fit_brand_model_reference(self, brand_model_dic: pd.DataFrame):
        """Prepara lookup tables do dicionário de referência."""
        
        df = brand_model_dic.copy()
        df['Brand'] = df['brand'].apply(utils.norm) if 'brand' in df.columns else df['Brand'].apply(utils.norm)
        df['model'] = df['model'].apply(utils.norm)
        
        self.params.valid_brands = df['Brand'].unique().tolist()
        self.params.all_models = df['model'].tolist()
        
        self.params.models_by_brand = df.groupby('Brand')['model'].apply(list).to_dict()
        self.params.brands_by_model = df.groupby('model')['Brand'].apply(list).to_dict()
        
        self.params.unique_brand_by_model = {
            model: brands[0]
            for model, brands in self.params.brands_by_model.items()
            if len(brands) == 1
        }
        
        self.params.transmission_uniques = [utils.norm(x) for x in ['Manual', 'Automatic', 'Semi-Auto', 'Other']]
        self.params.fueltype_uniques = [utils.norm(x) for x in ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other']]
    
    def _clean_brand_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e corrige Brand/model."""
        
        X = df.copy()
        X['bm_status'] = 'ok'
        X['bm_note'] = ''
        
        def _fix_row(row):
            b = '' if pd.isna(row['Brand']) else str(row['Brand'])
            m = '' if pd.isna(row['model']) else str(row['model'])
            
            if b == '' and m == '':
                row['bm_status'] = 'empty'
                return row
            
            if b == '' and m != '':
                if m == 'corsa':
                    row['Brand'] = 'opel'
                    row['bm_status'] = 'brand_inferred'
                    return row
                
                if m in self.params.unique_brand_by_model:
                    row['Brand'] = self.params.unique_brand_by_model[m]
                    row['bm_status'] = 'brand_inferred'
                    return row
                
                matches = utils.get_best_match(m, self.params.all_models)
                if len(matches) == 1:
                    matched_model = matches[0]
                    brands = self.params.brands_by_model.get(matched_model, [])
                    if len(brands) == 1:
                        row['Brand'] = brands[0]
                        row['model'] = matched_model
                        row['bm_status'] = 'brand_inferred'
                    else:
                        row['model'] = matched_model
                        row['bm_status'] = 'needs_review'
                else:
                    row['bm_status'] = 'needs_review'
                return row
            
            if b not in self.params.valid_brands:
                b_matches = utils.get_best_match(b, self.params.valid_brands)
                if len(b_matches) == 1:
                    row['Brand'] = b_matches[0]
                    b = b_matches[0]
                else:
                    row['bm_status'] = 'needs_review'
                    return row
            
            key = (b, m)
            if key in self.params.manual_model_fixes:
                row['model'] = self.params.manual_model_fixes[key]
                m = self.params.manual_model_fixes[key]
            
            if m == '':
                row['bm_status'] = 'no_model'
                return row
            
            brand_models = self.params.models_by_brand.get(b, [])
            if m in brand_models:
                return row
            
            if brand_models:
                m_matches = utils.get_best_match(m, brand_models)
                if len(m_matches) == 1:
                    row['model'] = m_matches[0]
                    row['bm_status'] = 'model_corrected'
                    return row
            
            m_matches_global = utils.get_best_match(m, self.params.all_models)
            if len(m_matches_global) == 1:
                new_m = m_matches_global[0]
                brands_for_m = self.params.brands_by_model.get(new_m, [])
                if len(brands_for_m) == 1 and brands_for_m[0] != b:
                    row['Brand'] = brands_for_m[0]
                row['model'] = new_m
                row['bm_status'] = 'model_corrected'
                return row
            
            row['bm_status'] = 'needs_review'
            return row
        
        X = X.apply(_fix_row, axis=1)
        return X
    
    def _impute_model_by_engine_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputa model para needs_review usando KNN em engineSize."""
        
        X = df.copy()
        K = self.params.k_neighbors
        
        mask_needs_review = X['bm_status'] == 'needs_review'
        df_reference = X[X['bm_status'] != 'needs_review'].copy()
        
        for car_id in X[mask_needs_review].index:
            row = X.loc[car_id]
            current_brand = row['Brand']
            current_model = row['model']
            current_engine_size = row['engineSize']
            
            if pd.isna(current_brand) or pd.isna(current_engine_size):
                continue
            
            ref_by_brand = df_reference[df_reference['Brand'] == current_brand].copy()
            
            if isinstance(current_model, str) and len(current_model) <= 2:
                ref_by_brand = ref_by_brand[
                    ref_by_brand['model'].str.startswith(current_model, na=False)
                ]
            
            ref_by_brand = ref_by_brand.dropna(subset=['engineSize'])
            if ref_by_brand.empty:
                continue
            
            distance = np.abs(ref_by_brand['engineSize'] - current_engine_size)
            nearest_neighbors = distance.nsmallest(K).index
            
            mode_result = ref_by_brand.loc[nearest_neighbors, 'model'].mode()
            if not mode_result.empty:
                X.loc[car_id, 'model'] = mode_result.iloc[0]
                X.loc[car_id, 'bm_status'] = 'model_imputed_by_esize'
        
        return X
    
    def _clean_numeric_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa variáveis numéricas e categóricas."""
        
        X = df.copy()
        
        X["year"] = X["year"].apply(lambda x: abs(int(x)) if pd.notnull(x) else x)
        X.loc[X["year"] > self.reference_year, "year"] = self.reference_year
        
        X['transmission'] = X['transmission'].apply(
            lambda x: self._correct_categorical_value(x, self.params.transmission_uniques)
        )
        
        X["mileage"] = X["mileage"].apply(
            lambda x: float(x) if pd.notnull(x) and x >= 0 else (abs(float(x)) if pd.notnull(x) else np.nan)
        )
        
        X['fuelType'] = X['fuelType'].apply(
            lambda x: self._correct_categorical_value(x, self.params.fueltype_uniques)
        )
        
        X["tax"] = X["tax"].apply(
            lambda x: float(x) if pd.notnull(x) and x >= 0 else (abs(float(x)) if pd.notnull(x) else np.nan)
        )
        
        X["mpg"] = X["mpg"].apply(
            lambda x: float(x) if pd.notnull(x) and x >= 0 else (abs(float(x)) if pd.notnull(x) else np.nan)
        )
        
        X.loc[X['engineSize'] == 0, 'engineSize'] = np.nan
        X["engineSize"] = X["engineSize"].apply(
            lambda x: float(x) if pd.notnull(x) and x >= 0 else (abs(float(x)) if pd.notnull(x) else np.nan)
        )
        
        X["paintQuality%"] = pd.to_numeric(X["paintQuality%"], errors="coerce").clip(lower=0, upper=100)
        
        X["previousOwners"] = X["previousOwners"].apply(
            lambda x: int(x) if pd.notnull(x) and x >= 0 else (abs(int(x)) if pd.notnull(x) else np.nan)
        )
        
        return X
    
    def _correct_categorical_value(self, input_value, valid_values, min_score=0.6, fallback_value="unknown"):
        """Corrige valor categórico usando fuzzy matching."""
        if pd.isna(input_value):
            return fallback_value
        
        normalized_input = utils.norm(input_value)
        best_matches = utils.get_best_match(normalized_input, valid_values, min_score=min_score)
        
        if isinstance(best_matches, list):
            if len(best_matches) == 1:
                return best_matches[0]
            elif len(best_matches) > 1:
                return None
            else:
                return fallback_value
        elif isinstance(best_matches, tuple):
            return best_matches[0]
        return fallback_value
    
    def _fit_keep_models_per_brand(self, df: pd.DataFrame, min_count=20, min_freq=0.01) -> pd.DataFrame:
        """Calcula tabela de modelos frequentes."""
        
        tmp = df[['Brand', 'model']].copy()
        tmp = tmp[~tmp['Brand'].isin(["unknown"])]
        tmp = tmp[~tmp['model'].isin(["unknown"])]
        
        g = tmp.groupby(['Brand', 'model'], dropna=False).size().rename("n").reset_index()
        g["brand_total"] = g.groupby('Brand')["n"].transform("sum")
        g["freq"] = g["n"] / g["brand_total"]
        
        keep = g.loc[
            (g["n"] >= min_count) & (g["freq"] >= min_freq),
            ['Brand', 'model']
        ].copy()
        keep["keep"] = True
        
        return keep
    
    def _collapse_rare_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """Colapsa modelos raros para 'other'."""
        
        original_index = df.index
        X = df.copy()
        
        X = X.merge(self.params.keep_models_table, how="left", on=['Brand', 'model'])
        
        keep_mask = X["keep"].fillna(False)
        mask_to_replace = (
            (~keep_mask) &
            (~X['Brand'].isin(["unknown", None, np.nan])) &
            (~X['model'].isin(["unknown", None, np.nan]))
        )
        
        X.loc[mask_to_replace, 'model'] = 'other'
        X.index = original_index
        X = X.drop(columns=["keep"], errors="ignore")
        
        return X
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria todas as features derivadas."""
        
        X = df.copy()
        
        # Join brand_stats
        X = X.join(self.params.brand_stats, on='Brand', how='left')
        
        # Age features
        X["age"] = (self.reference_year - pd.to_numeric(X["year"], errors="coerce")).astype("Int64")
        X['is_new_car'] = (X['age'] <= 1).astype(int)
        X['is_old_car'] = (X['age'] >= 10).astype(int)
        
        # Mileage features
        X['miles_per_year'] = X['mileage'] / (X['age'] + 1)
        X['high_mileage'] = (X['mileage'] > self.params.mileage_q75).astype(int)
        X['low_mileage'] = (X['mileage'] < self.params.mileage_q25).astype(int)
        
        # Interactions
        X['age_mileage_interaction'] = X['age'] * X['mileage']
        X['premium_brand_engine_size_interaction'] = (X['brand_segment'] == 'luxury').astype(int) * X['engineSize']
        X['tax_per_engine'] = X['tax'] / (X['engineSize'] + 0.1)
        X['mpg_per_liter'] = X['mpg'] / (X['engineSize'] + 0.1)
        
        # Brand_model
        X['brand_model'] = X['Brand'] + '_' + X['model']
        
        # Model popularity
        X['model_popularity'] = X['model'].map(self.params.model_counts).fillna(0).astype(int)
        
        # CORREÇÃO: Drop model e year aqui (Brand vai ser removido depois do target encoding)
        X = X.drop(columns=["model", "year"], errors="ignore")
        
        return X
    
    def _apply_encoding_and_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica target encoding, OHE e scaling final."""
        
        X = df.copy()
        
        # Target encoding
        high_cardinality_cols = ['Brand', 'brand_model']
        for col in high_cardinality_cols:
            if col in X.columns:
                encoding = self.params.target_encodings.get(col, {})
                X[f'{col}_target_enc'] = X[col].map(encoding).fillna(self.params.global_mean_price)
        
        # CORREÇÃO: Remover high cardinality cols imediatamente após encoding
        X = X.drop(columns=high_cardinality_cols, errors='ignore')
        
        # OHE
        if self.params.low_cardinality_cols and self.params.ohe_encoder is not None:
            existing_cols = [c for c in self.params.low_cardinality_cols if c in X.columns]
            if existing_cols:
                # Garantir que todas as colunas esperadas existem
                for col in self.params.low_cardinality_cols:
                    if col not in X.columns:
                        X[col] = 'unknown'
                
                ohe_transformed = pd.DataFrame(
                    self.params.ohe_encoder.transform(X[self.params.low_cardinality_cols]),
                    columns=self.params.ohe_encoder.get_feature_names_out(),
                    index=X.index
                )
                X = X.drop(columns=self.params.low_cardinality_cols, errors='ignore')
                X = pd.concat([X, ohe_transformed], axis=1)
        
        # CORREÇÃO: Drop TODAS as possíveis colunas string restantes
        cols_to_drop = ['transmission', 'fuelType', 'brand_model', 'model_popularity', 
                        'brand_segment', 'Brand', 'model', 'year']
        X = X.drop(columns=cols_to_drop, errors="ignore")
        
        # CORREÇÃO CRÍTICA: Remover QUALQUER coluna string restante
        remaining_string_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if remaining_string_cols:
            print(f"  ⚠️ Removendo colunas string restantes: {remaining_string_cols}")
            X = X.drop(columns=remaining_string_cols, errors='ignore')
        
        # Garantir mesmas features que o treino
        for col in self.params.final_feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Remover colunas extra
        X = X[self.params.final_feature_names]
        
        # Scaling final
        X = pd.DataFrame(
            self.params.final_scaler.transform(X),
            columns=self.params.final_feature_names,
            index=X.index
        )
        
        return X
    
    # =========================================================================
    # SAVE / LOAD
    # =========================================================================
    
    def save(self, filepath: str):
        """Guarda o pipeline em ficheiro."""
        joblib.dump(self, filepath)
        print(f"Pipeline guardado em: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'CarPreprocessingPipeline':
        """Carrega pipeline de ficheiro."""
        pipeline = joblib.load(filepath)
        print(f"Pipeline carregado de: {filepath}")
        return pipeline
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de features finais."""
        return self.params.final_feature_names