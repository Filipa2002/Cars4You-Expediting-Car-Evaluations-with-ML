"""
Car Preprocessing Pipeline
==========================
A comprehensive preprocessing pipeline for car price prediction.
Handles brand/model cleaning, imputation, feature engineering, and encoding.
"""

import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass, field
from typing import Dict, List, Any

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, PowerTransformer, OneHotEncoder

import utils


# =============================================================================
# FITTED PARAMETERS CONTAINER
# =============================================================================

@dataclass
class FittedParams:
    """Container for all fitted parameters from training data."""
    
    # Brand/model reference lookups
    valid_brands: List[str] = field(default_factory=list)
    all_models: List[str] = field(default_factory=list)
    models_by_brand: Dict[str, List[str]] = field(default_factory=dict)
    brands_by_model: Dict[str, List[str]] = field(default_factory=dict)
    unique_brand_by_model: Dict[str, str] = field(default_factory=dict)
    
    # Manual corrections
    manual_model_fixes: Dict[tuple, str] = field(default_factory=dict)
    manual_model_corrections: Dict[str, Dict[str, str]] = field(default_factory=dict)
    k_neighbors: int = 5
    
    # Valid categorical values
    transmission_uniques: List[str] = field(default_factory=list)
    fueltype_uniques: List[str] = field(default_factory=list)
    
    # MCAR imputation parameters
    brand_mode: str = None
    model_mode_per_brand: Dict[str, str] = field(default_factory=dict)
    model_mode_global: str = None
    
    # MAR variables
    mar_vars: List[str] = field(default_factory=list)
    
    # MICE imputation
    mice_imputer: Any = None
    mice_scaler: Any = None
    numeric_cols: List[str] = field(default_factory=list)
    
    # Rare model collapsing
    keep_models_table: pd.DataFrame = None
    
    # Feature engineering parameters
    brand_stats: pd.DataFrame = None
    mileage_q25: float = None
    mileage_q75: float = None
    model_counts: Dict[str, int] = field(default_factory=dict)
    
    # Encoding parameters
    target_encodings: Dict[str, Dict[str, float]] = field(default_factory=dict)
    global_mean_price: float = None
    ohe_encoder: Any = None
    low_cardinality_cols: List[str] = field(default_factory=list)
    
    # Final scaling
    final_scaler: Any = None
    final_feature_names: List[str] = field(default_factory=list)


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class CarPreprocessingPipeline:
    """
    Preprocessing pipeline for car price prediction data.
    
    Parameters
    ----------
    reference_year : int, default=2020
        Reference year for calculating car age.
    use_final_scaling : bool, default=True
        If True, applies PowerTransformer at the end.
        If False, returns features without final scaling.
    verbose : bool, default=True
        If True, prints progress messages.
    """
    
    def __init__(self, reference_year: int = 2020, use_final_scaling: bool = True, verbose: bool = True):
        self.reference_year = reference_year
        self.use_final_scaling = use_final_scaling
        self.verbose = verbose
        self.params = FittedParams()
        self.is_fitted = False
        self._X_train_transformed = None
        self._original_train_ids = None
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    # =========================================================================
    # FIT METHOD
    # =========================================================================
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
            brand_model_dic: pd.DataFrame,
            manual_model_fixes: Dict[tuple, str] = None,
            manual_model_corrections: Dict[str, Dict[str, str]] = None) -> 'CarPreprocessingPipeline':
        """
        Fit the preprocessing pipeline on training data.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.DataFrame
            Training target (must contain 'price' column).
        brand_model_dic : pd.DataFrame
            Reference dictionary with valid brand/model combinations.
        manual_model_fixes : dict, optional
            Manual fixes for specific (brand, model) pairs.
        manual_model_corrections : dict, optional
            Dataset-specific model corrections.
        
        Returns
        -------
        self
        """
        X = X_train.copy()
        y = y_train.copy()
        
        # Preserve carID
        self._original_train_ids = self._extract_car_ids(X)
        if 'carID' in X.columns:
            X = X.drop(columns=['carID'])
        elif X.index.name == 'carID':
            X = X.reset_index(drop=True)
        
        self.params.manual_model_fixes = manual_model_fixes or {}
        self.params.manual_model_corrections = manual_model_corrections or {}
        
        self._log("Fitting preprocessing pipeline...")
        
        # Step 1: Prepare brand/model reference
        self._fit_brand_model_reference(brand_model_dic)
        
        # Step 2: Normalize brand and model
        X['Brand'] = X['Brand'].apply(utils.norm)
        X['model'] = X['model'].apply(utils.norm)
        
        # Step 3: Clean brand and model
        X = self._clean_brand_model(X)
        
        # Step 4: Impute model using engineSize
        X = self._impute_model_by_engine_size(X)
        
        # Step 5: Handle unresolved cases
        mask = X['bm_status'] == 'needs_review'
        X.loc[mask, 'model'] = None
        X = X.drop(columns=['bm_status', 'bm_note'], errors='ignore')
        
        # Step 6: Clean numeric and categorical variables
        X = self._clean_numeric_categorical(X)
        
        # Step 7: Fit MCAR imputation (Brand and model modes)
        self._fit_mcar_imputation(X)
        X['Brand'] = X['Brand'].fillna(self.params.brand_mode)
        missing_mask = X['model'].isna()
        X.loc[missing_mask, 'model'] = X.loc[missing_mask, 'Brand'].map(self.params.model_mode_per_brand)
        
        # Step 8: Create missing value flags for MAR variables
        self.params.mar_vars = ['mpg', 'tax', 'engineSize', 'year']
        for var in self.params.mar_vars:
            X[f"{var}_is_missing"] = X[var].isna().astype(int)
        
        # Step 9: Fit MICE imputation
        X = self._fit_mice_imputation(X)
        
        # Step 10: Fit rare models collapse
        self.params.keep_models_table = self._fit_keep_models_per_brand(X)
        X = self._collapse_rare_models(X)
        
        # Inverse transform for feature engineering (need original scale)
        X[self.params.numeric_cols] = self.params.mice_scaler.inverse_transform(X[self.params.numeric_cols])
        for col in ['year', 'previousOwners']:
            if col in X.columns:
                X[col] = X[col].round().astype(int, errors='ignore')
        
        # Step 11: Fit feature engineering parameters
        y_price = y['price'] if isinstance(y, pd.DataFrame) else y
        self._fit_feature_engineering_params(X, y_price)
        X = self._apply_feature_engineering(X)
        
        # Step 12: Fit encoding
        X = self._fit_encoding(X, y_price)
        
        # Step 13: Fit final scaler (conditional)
        X = self._fit_final_scaling(X)
        
        # Restore carID as index
        if self._original_train_ids is not None:
            X.index = self._original_train_ids
            X.index.name = 'carID'
        
        self.is_fitted = True
        self._X_train_transformed = X.copy()
        
        scaling_status = 'PowerTransformer' if self.use_final_scaling else 'None'
        self._log(f"Pipeline fitted. Shape: {X.shape}, Features: {len(self.params.final_feature_names)}, Scaling: {scaling_status}")
        
        return self
    
    # =========================================================================
    # TRANSFORM METHOD
    # =========================================================================
    
    def transform(self, X: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Transform data using the fitted pipeline.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.
        dataset_name : str, default="dataset"
            Name for logging and dataset-specific corrections.
        
        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        self._log(f"Transforming {dataset_name} ({X.shape[0]} rows)...")
        
        df = X.copy()
        
        # Preserve carID
        original_car_ids = self._extract_car_ids(df)
        if 'carID' in df.columns:
            df = df.drop(columns=['carID'])
        elif df.index.name == 'carID':
            df = df.reset_index(drop=True)
        
        # Step 1-2: Normalize brand and model
        df['Brand'] = df['Brand'].apply(utils.norm)
        df['model'] = df['model'].apply(utils.norm)
        
        # Step 3: Clean brand and model
        df = self._clean_brand_model(df)
        
        # Step 4: Impute model using engineSize
        df = self._impute_model_by_engine_size(df)
        
        # Step 5: Handle unresolved cases and manual corrections
        mask = df['bm_status'] == 'needs_review'
        df.loc[mask, 'model'] = None
        
        if dataset_name in self.params.manual_model_corrections:
            for old_model, new_model in self.params.manual_model_corrections[dataset_name].items():
                df.loc[df['model'] == old_model, 'model'] = new_model
        
        df = df.drop(columns=['bm_status', 'bm_note'], errors='ignore')
        
        # Step 6: Clean numeric and categorical variables
        df = self._clean_numeric_categorical(df)
        
        # Step 7: Apply MCAR imputation
        df['Brand'] = df['Brand'].fillna(self.params.brand_mode)
        missing_mask = df['model'].isna()
        df.loc[missing_mask, 'model'] = df.loc[missing_mask, 'Brand'].map(self.params.model_mode_per_brand)
        still_missing = df['model'].isna()
        df.loc[still_missing, 'model'] = self.params.model_mode_global
        
        # Step 8: Create missing value flags
        for var in self.params.mar_vars:
            df[f"{var}_is_missing"] = df[var].isna().astype(int)
        
        # Step 9: Apply MICE imputation
        for col in self.params.numeric_cols:
            if col not in df.columns:
                df[col] = 0
        
        data_numeric = df[self.params.numeric_cols].copy()
        data_imputed = self.params.mice_imputer.transform(data_numeric)
        data_scaled = self.params.mice_scaler.transform(data_imputed)
        df[self.params.numeric_cols] = pd.DataFrame(data_scaled, index=df.index, columns=self.params.numeric_cols)
        
        # Step 10: Collapse rare models
        df = self._collapse_rare_models(df)
        
        # Inverse transform for feature engineering
        df[self.params.numeric_cols] = self.params.mice_scaler.inverse_transform(df[self.params.numeric_cols])
        for col in ['year', 'previousOwners']:
            if col in df.columns:
                df[col] = df[col].round().astype(int, errors='ignore')
        
        # Step 11: Apply feature engineering
        df = self._apply_feature_engineering(df)
        
        # Step 12: Apply encoding and final scaling
        df = self._apply_encoding_and_scaling(df)
        
        # Restore carID as index
        if original_car_ids is not None:
            df.index = original_car_ids
            df.index.name = 'carID'
        
        self._log(f"Transformation complete. Shape: {df.shape}")
        return df
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                      brand_model_dic: pd.DataFrame,
                      manual_model_fixes: Dict[tuple, str] = None,
                      manual_model_corrections: Dict[str, Dict[str, str]] = None) -> pd.DataFrame:
        """Fit and transform in a single step."""
        self.fit(X_train, y_train, brand_model_dic, manual_model_fixes, manual_model_corrections)
        return self.get_transformed_train()
    
    def get_transformed_train(self) -> pd.DataFrame:
        """Return the transformed training data after fit()."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        if self._X_train_transformed is None:
            raise RuntimeError("Transformed training data not available.")
        return self._X_train_transformed.copy()
    
    def get_feature_names(self) -> List[str]:
        """Return list of final feature names."""
        return self.params.final_feature_names
    
    # =========================================================================
    # SAVE / LOAD
    # =========================================================================
    
    def save(self, filepath: str):
        """Save pipeline to file."""
        joblib.dump(self, filepath)
        self._log(f"Pipeline saved: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'CarPreprocessingPipeline':
        """Load pipeline from file."""
        pipeline = joblib.load(filepath)
        print(f"Pipeline loaded: {filepath}")
        return pipeline
    
    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================
    
    def _extract_car_ids(self, df: pd.DataFrame):
        """Extract carID from DataFrame."""
        if 'carID' in df.columns:
            return df['carID'].values.copy()
        elif df.index.name == 'carID':
            return df.index.values.copy()
        return None
    
    def _fit_brand_model_reference(self, brand_model_dic: pd.DataFrame):
        """Build lookup tables from reference dictionary."""
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
    
    def _fit_mcar_imputation(self, X: pd.DataFrame):
        """Fit MCAR imputation parameters (modes)."""
        self.params.brand_mode = X['Brand'].mode().iloc[0] if not X['Brand'].mode().empty else 'unknown'
        self.params.model_mode_global = X['model'].mode().iloc[0] if not X['model'].mode().empty else 'unknown'
        self.params.model_mode_per_brand = (
            X.groupby('Brand')['model']
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown')
            .to_dict()
        )
    
    def _fit_mice_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit MICE imputer and scale numeric columns."""
        self.params.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.params.mice_imputer = IterativeImputer(max_iter=10, random_state=37)
        self.params.mice_scaler = RobustScaler()
        
        data_numeric = X[self.params.numeric_cols].copy()
        data_imputed = self.params.mice_imputer.fit_transform(data_numeric)
        data_scaled = self.params.mice_scaler.fit_transform(data_imputed)
        
        X[self.params.numeric_cols] = pd.DataFrame(data_scaled, index=X.index, columns=self.params.numeric_cols)
        return X
    
    def _fit_feature_engineering_params(self, X: pd.DataFrame, y_price: pd.Series):
        """Fit parameters needed for feature engineering."""
        # Brand statistics
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
        
        # Brand segments based on price percentiles
        p33, p67 = self.params.brand_stats['brand_mean_price'].quantile([0.33, 0.67])
        self.params.brand_stats['brand_segment'] = pd.cut(
            self.params.brand_stats['brand_mean_price'],
            bins=[-np.inf, p33, p67, np.inf],
            labels=['budget', 'mid_range', 'luxury']
        )
        
        # Mileage quartiles
        self.params.mileage_q25 = X['mileage'].quantile(0.25)
        self.params.mileage_q75 = X['mileage'].quantile(0.75)
        
        # Model counts
        self.params.model_counts = X['model'].value_counts().to_dict()
    
    def _fit_encoding(self, X: pd.DataFrame, y_price: pd.Series) -> pd.DataFrame:
        """Fit target encoding and OHE."""
        high_cardinality_cols = ['Brand', 'brand_model']
        self.params.global_mean_price = y_price.mean()
        self.params.target_encodings = {}
        
        # Target encoding for high cardinality columns
        for col in high_cardinality_cols:
            if col in X.columns:
                self.params.target_encodings[col] = y_price.groupby(X[col]).mean().to_dict()
                X[f'{col}_target_enc'] = X[col].map(self.params.target_encodings[col])
        
        X = X.drop(columns=high_cardinality_cols, errors='ignore')
        
        # OHE for low cardinality columns
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
        
        # Remove any remaining string columns
        cols_to_drop = ['transmission', 'fuelType', 'brand_model', 'model_popularity', 
                        'brand_segment', 'Brand', 'model', 'year']
        X = X.drop(columns=cols_to_drop, errors="ignore")
        
        remaining_string_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if remaining_string_cols:
            X = X.drop(columns=remaining_string_cols, errors='ignore')
        
        return X
    
    def _fit_final_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit final scaler if enabled."""
        if self.use_final_scaling:
            self.params.final_scaler = PowerTransformer()
            self.params.final_feature_names = X.columns.tolist()
            X = pd.DataFrame(
                self.params.final_scaler.fit_transform(X),
                columns=self.params.final_feature_names,
                index=X.index
            )
        else:
            self.params.final_scaler = None
            self.params.final_feature_names = X.columns.tolist()
        return X
    
    # =========================================================================
    # PRIVATE TRANSFORMATION METHODS
    # =========================================================================
    
    def _clean_brand_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and correct Brand/model using fuzzy matching."""
        X = df.copy()
        X['bm_status'] = 'ok'
        X['bm_note'] = ''
        
        def _fix_row(row):
            b = '' if pd.isna(row['Brand']) else str(row['Brand'])
            m = '' if pd.isna(row['model']) else str(row['model'])
            
            # Both empty
            if b == '' and m == '':
                row['bm_status'] = 'empty'
                return row
            
            # Brand missing, model present
            if b == '' and m != '':
                # Special case: corsa -> opel
                if m == 'corsa':
                    row['Brand'] = 'opel'
                    row['bm_status'] = 'brand_inferred'
                    return row
                
                # Check if model uniquely identifies brand
                if m in self.params.unique_brand_by_model:
                    row['Brand'] = self.params.unique_brand_by_model[m]
                    row['bm_status'] = 'brand_inferred'
                    return row
                
                # Try fuzzy matching
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
            
            # Validate brand
            if b not in self.params.valid_brands:
                b_matches = utils.get_best_match(b, self.params.valid_brands)
                if len(b_matches) == 1:
                    row['Brand'] = b_matches[0]
                    b = b_matches[0]
                else:
                    row['bm_status'] = 'needs_review'
                    return row
            
            # Apply manual fixes
            key = (b, m)
            if key in self.params.manual_model_fixes:
                row['model'] = self.params.manual_model_fixes[key]
                m = self.params.manual_model_fixes[key]
            
            # Model missing
            if m == '':
                row['bm_status'] = 'no_model'
                return row
            
            # Check if model exists for brand
            brand_models = self.params.models_by_brand.get(b, [])
            if m in brand_models:
                return row
            
            # Try fuzzy matching within brand
            if brand_models:
                m_matches = utils.get_best_match(m, brand_models)
                if len(m_matches) == 1:
                    row['model'] = m_matches[0]
                    row['bm_status'] = 'model_corrected'
                    return row
            
            # Try global fuzzy matching
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
        """Impute model for unresolved cases using KNN on engineSize."""
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
            
            # Filter reference by brand
            ref_by_brand = df_reference[df_reference['Brand'] == current_brand].copy()
            
            # If model is 1-2 chars, filter by prefix
            if isinstance(current_model, str) and len(current_model) <= 2:
                ref_by_brand = ref_by_brand[
                    ref_by_brand['model'].str.startswith(current_model, na=False)
                ]
            
            ref_by_brand = ref_by_brand.dropna(subset=['engineSize'])
            if ref_by_brand.empty:
                continue
            
            # Find K nearest neighbors by engineSize
            distance = np.abs(ref_by_brand['engineSize'] - current_engine_size)
            nearest_neighbors = distance.nsmallest(K).index
            
            # Use mode of neighbors' models
            mode_result = ref_by_brand.loc[nearest_neighbors, 'model'].mode()
            if not mode_result.empty:
                X.loc[car_id, 'model'] = mode_result.iloc[0]
                X.loc[car_id, 'bm_status'] = 'model_imputed_by_esize'
        
        return X
    
    def _clean_numeric_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric and categorical variables."""
        X = df.copy()
        
        # Year: absolute value, cap at reference year
        X["year"] = X["year"].apply(lambda x: abs(int(x)) if pd.notnull(x) else x)
        X.loc[X["year"] > self.reference_year, "year"] = self.reference_year
        
        # Transmission: fuzzy match to valid values
        X['transmission'] = X['transmission'].apply(
            lambda x: self._correct_categorical_value(x, self.params.transmission_uniques)
        )
        
        # Mileage: absolute value
        X["mileage"] = X["mileage"].apply(
            lambda x: float(x) if pd.notnull(x) and x >= 0 else (abs(float(x)) if pd.notnull(x) else np.nan)
        )
        
        # Fuel type: fuzzy match to valid values
        X['fuelType'] = X['fuelType'].apply(
            lambda x: self._correct_categorical_value(x, self.params.fueltype_uniques)
        )
        
        # Tax: absolute value
        X["tax"] = X["tax"].apply(
            lambda x: float(x) if pd.notnull(x) and x >= 0 else (abs(float(x)) if pd.notnull(x) else np.nan)
        )
        
        # MPG: absolute value
        X["mpg"] = X["mpg"].apply(
            lambda x: float(x) if pd.notnull(x) and x >= 0 else (abs(float(x)) if pd.notnull(x) else np.nan)
        )
        
        # Engine size: 0 -> NaN, absolute value
        X.loc[X['engineSize'] == 0, 'engineSize'] = np.nan
        X["engineSize"] = X["engineSize"].apply(
            lambda x: float(x) if pd.notnull(x) and x >= 0 else (abs(float(x)) if pd.notnull(x) else np.nan)
        )
        
        # Paint quality: clip to [0, 100]
        X["paintQuality%"] = pd.to_numeric(X["paintQuality%"], errors="coerce").clip(lower=0, upper=100)
        
        # Previous owners: absolute integer
        X["previousOwners"] = X["previousOwners"].apply(
            lambda x: int(x) if pd.notnull(x) and x >= 0 else (abs(int(x)) if pd.notnull(x) else np.nan)
        )
        
        return X
    
    def _correct_categorical_value(self, input_value, valid_values, min_score=0.6, fallback_value="unknown"):
        """Correct categorical value using fuzzy matching."""
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
        """Calculate table of frequent models to keep."""
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
        """Collapse rare models to 'other'."""
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
        """Create derived features."""
        X = df.copy()
        
        # Join brand statistics
        X = X.join(self.params.brand_stats, on='Brand', how='left')
        
        # Age features
        X["age"] = (self.reference_year - pd.to_numeric(X["year"], errors="coerce")).astype("Int64")
        X['is_new_car'] = (X['age'] <= 1).astype(int)
        X['is_old_car'] = (X['age'] >= 10).astype(int)
        
        # Mileage features
        X['miles_per_year'] = X['mileage'] / (X['age'] + 1)
        X['high_mileage'] = (X['mileage'] > self.params.mileage_q75).astype(int)
        X['low_mileage'] = (X['mileage'] < self.params.mileage_q25).astype(int)
        
        # Interaction features
        X['age_mileage_interaction'] = X['age'] * X['mileage']
        X['premium_brand_engine_size_interaction'] = (X['brand_segment'] == 'luxury').astype(int) * X['engineSize']
        X['tax_per_engine'] = X['tax'] / (X['engineSize'] + 0.1)
        X['mpg_per_liter'] = X['mpg'] / (X['engineSize'] + 0.1)
        
        # Brand_model combination
        X['brand_model'] = X['Brand'] + '_' + X['model']
        
        # Model popularity
        X['model_popularity'] = X['model'].map(self.params.model_counts).fillna(0).astype(int)
        
        # Drop intermediate columns
        X = X.drop(columns=["model", "year"], errors="ignore")
        
        return X
    
    def _apply_encoding_and_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply target encoding, OHE, and final scaling."""
        X = df.copy()
        
        # Target encoding
        high_cardinality_cols = ['Brand', 'brand_model']
        for col in high_cardinality_cols:
            if col in X.columns:
                encoding = self.params.target_encodings.get(col, {})
                X[f'{col}_target_enc'] = X[col].map(encoding).fillna(self.params.global_mean_price)
        
        X = X.drop(columns=high_cardinality_cols, errors='ignore')
        
        # OHE
        if self.params.low_cardinality_cols and self.params.ohe_encoder is not None:
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
        
        # Remove remaining string columns
        cols_to_drop = ['transmission', 'fuelType', 'brand_model', 'model_popularity', 
                        'brand_segment', 'Brand', 'model', 'year']
        X = X.drop(columns=cols_to_drop, errors="ignore")
        
        remaining_string_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if remaining_string_cols:
            X = X.drop(columns=remaining_string_cols, errors='ignore')
        
        # Ensure same features as training
        for col in self.params.final_feature_names:
            if col not in X.columns:
                X[col] = 0
        
        X = X[self.params.final_feature_names]
        
        # Final scaling (conditional)
        if self.use_final_scaling and self.params.final_scaler is not None:
            X = pd.DataFrame(
                self.params.final_scaler.transform(X),
                columns=self.params.final_feature_names,
                index=X.index
            )
        
        return X
