from abc import ABC, abstractmethod
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
import pickle
import warnings


class BaseProcessor(ABC):
    """Abstract base class for all data processors."""
    
    @abstractmethod
    def fit(self, data: pd.Series, **kwargs) -> 'BaseProcessor':
        """Fit the processor on training data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.Series) -> pd.Series:
        """Transform the data using fitted parameters."""
        pass
    
    def fit_transform(self, data: pd.Series, **kwargs) -> pd.Series:
        """Fit and transform in one step."""
        return self.fit(data, **kwargs).transform(data)


class MissingValueProcessor(BaseProcessor):
    """Handle missing values with various strategies."""
    
    def __init__(self, strategy: str = 'mean'):
        self.strategy = strategy
        self.fill_value = None
    
    def fit(self, data: pd.Series, **kwargs) -> 'MissingValueProcessor':
        if self.strategy == 'mean':
            self.fill_value = data.mean()
        elif self.strategy == 'median':
            self.fill_value = data.median()
        elif self.strategy == 'mode':
            mode_val = data.mode()
            self.fill_value = mode_val.iloc[0] if not mode_val.empty else 0
        elif isinstance(self.strategy, (int, float)):
            self.fill_value = self.strategy
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        if pd.isna(self.fill_value):
            warnings.warn(f"Fill value is NaN for strategy '{self.strategy}'")
            self.fill_value = 0
        
        return self
    
    def transform(self, data: pd.Series) -> pd.Series:
        return data.fillna(self.fill_value)


class OutlierProcessor(BaseProcessor):
    """Handle outliers using IQR or Z-score methods."""
    
    def __init__(self, method: str = 'iqr', action: str = 'clip', threshold: float = 3.0):
        self.method = method
        self.action = action
        self.threshold = threshold
        self.lower_bound = None
        self.upper_bound = None
    
    def fit(self, data: pd.Series, **kwargs) -> 'OutlierProcessor':
        if self.method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bound = Q1 - 1.5 * IQR
            self.upper_bound = Q3 + 1.5 * IQR
        elif self.method == 'zscore':
            mean = data.mean()
            std = data.std()
            self.lower_bound = mean - self.threshold * std
            self.upper_bound = mean + self.threshold * std
        else:
            raise ValueError(f"Unknown outlier method: {self.method}")
        
        return self
    
    def transform(self, data: pd.Series) -> pd.Series:
        if self.action == 'clip':
            return data.clip(lower=self.lower_bound, upper=self.upper_bound)
        elif self.action == 'mask':
            # Return mask for filtering (caller handles removal)
            return (data >= self.lower_bound) & (data <= self.upper_bound)
        else:
            raise ValueError(f"Unknown action: {self.action}")


class TransformProcessor(BaseProcessor):
    """Apply custom transformations like log, sqrt, etc."""
    
    def __init__(self, transform_func: Callable):
        self.transform_func = transform_func
    
    def fit(self, data: pd.Series, **kwargs) -> 'TransformProcessor':
        # Transformations are typically stateless
        return self
    
    def transform(self, data: pd.Series) -> pd.Series:
        try:
            return self.transform_func(data)
        except Exception as e:
            warnings.warn(f"Transform function failed: {e}")
            return data


class ScalingProcessor(BaseProcessor):
    """Handle different scaling methods."""
    
    def __init__(self, method: str = 'standard'):
        self.method = method
        self.scaler = None
    
    def fit(self, data: pd.Series, **kwargs) -> 'ScalingProcessor':
        data_reshaped = data.values.reshape(-1, 1)
        
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        self.scaler.fit(data_reshaped)
        return self
    
    def transform(self, data: pd.Series) -> pd.Series:
        data_reshaped = data.values.reshape(-1, 1)
        return pd.Series(
            self.scaler.transform(data_reshaped).flatten(),
            index=data.index,
            name=data.name
        )


class EncodingProcessor(BaseProcessor):
    """Handle categorical encoding."""
    
    def __init__(self, method: str = 'label'):
        self.method = method
        self.encoder = None
        self.encoded_columns = None
    
    def fit(self, data: pd.Series, target: Optional[pd.Series] = None, **kwargs) -> 'EncodingProcessor':
        if self.method == 'label':
            self.encoder = LabelEncoder()
            self.encoder.fit(data.dropna())
        
        elif self.method == 'onehot':
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            data_reshaped = data.fillna('Unknown').astype(str).values.reshape(-1, 1)
            self.encoder.fit(data_reshaped)
            self.encoded_columns = self.encoder.get_feature_names_out([data.name]).tolist()
        
        elif self.method == 'mean':
            if target is None:
                raise ValueError("Target series required for mean encoding")
            
            aligned_data, aligned_target = data.align(target, join='inner')
            mean_target = aligned_target.mean()
            
            mapping = {}
            for cat in aligned_data.unique():
                if pd.isnull(cat):
                    continue
                mask = aligned_data == cat
                if mask.any():
                    mapping[cat] = aligned_target[mask].mean()
            
            # Use defaultdict for unseen categories
            self.encoder = defaultdict(lambda: mean_target, mapping)
        
        else:
            raise ValueError(f"Unknown encoding method: {self.method}")
        
        return self
    
    def transform(self, data: pd.Series) -> Union[pd.Series, pd.DataFrame]:
        if self.method == 'label':
            # Handle unseen categories
            data_clean = data.copy()
            if hasattr(self.encoder, 'classes_'):
                unseen_mask = ~data_clean.isin(self.encoder.classes_)
                if unseen_mask.any():
                    most_frequent = self.encoder.classes_[0]
                    data_clean.loc[unseen_mask] = most_frequent
            
            return pd.Series(
                self.encoder.transform(data_clean),
                index=data.index,
                name=data.name
            )
        
        elif self.method == 'onehot':
            data_reshaped = data.fillna('Unknown').astype(str).values.reshape(-1, 1)
            encoded_array = self.encoder.transform(data_reshaped)
            
            # Return DataFrame with proper column names
            return pd.DataFrame(
                encoded_array,
                index=data.index,
                columns=self.encoded_columns
            )
        
        elif self.method == 'mean':
            return pd.Series(
                data.map(self.encoder),
                index=data.index,
                name=f"{data.name}_mean"
            )


class FeaturePipeline:
    """Pipeline for processing a single feature with multiple processors."""
    
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.processors = []
        self.is_fitted = False
    
    def add_processor(self, processor: BaseProcessor) -> 'FeaturePipeline':
        """Add a processor to the pipeline."""
        self.processors.append(processor)
        return self
    
    def fit(self, data: pd.Series, target: Optional[pd.Series] = None) -> 'FeaturePipeline':
        """Fit all processors in sequence."""
        current_data = data.copy()
        
        for processor in self.processors:
            if isinstance(processor, EncodingProcessor):
                processor.fit(current_data, target=target)
            else:
                processor.fit(current_data)
            
            # Transform for next processor (except for outlier masking)
            if not (isinstance(processor, OutlierProcessor) and processor.action == 'mask'):
                current_data = processor.transform(current_data)
                if isinstance(current_data, pd.DataFrame):
                    # For one-hot encoding, use first column for next processors
                    current_data = current_data.iloc[:, 0]
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.Series) -> Union[pd.Series, pd.DataFrame]:
        """Transform data through all processors."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        current_data = data.copy()
        
        for processor in self.processors:
            result = processor.transform(current_data)
            
            if isinstance(processor, OutlierProcessor) and processor.action == 'mask':
                # Handle outlier masking differently
                continue
            
            current_data = result
            if isinstance(current_data, pd.DataFrame):
                # For one-hot encoding, keep DataFrame structure
                break
        
        return current_data


class DataPreprocessor:
    """Main class orchestrating all preprocessing operations."""
    
    def __init__(self, target_column: str = 'price', verbose: bool = False):
        self.target_column = target_column
        self.feature_pipelines: Dict[str, FeaturePipeline] = {}
        self.is_fitted = False
        self.verbose = verbose
    
    def add_feature_pipeline(self, feature_name: str, 
                           missing_strategy: Optional[str] = None,
                           outlier_method: Optional[str] = None,
                           outlier_action: str = 'clip',
                           transform_func: Optional[Callable] = None,
                           scaling_method: Optional[str] = None,
                           encoding_method: Optional[str] = None) -> 'DataPreprocessor':
        """
        Add a complete preprocessing pipeline for a feature.
        
        Args:
            feature_name: Name of the feature
            missing_strategy: 'mean', 'median', 'mode', or numeric value
            outlier_method: 'iqr' or 'zscore'
            outlier_action: 'clip' or 'mask'
            transform_func: Custom transformation function
            scaling_method: 'standard', 'minmax', or 'robust'
            encoding_method: 'label', 'onehot', or 'mean'
        """
        pipeline = FeaturePipeline(feature_name)
        
        # Add processors in logical order
        if missing_strategy:
            pipeline.add_processor(MissingValueProcessor(missing_strategy))
        
        if outlier_method:
            pipeline.add_processor(OutlierProcessor(outlier_method, outlier_action))
        
        if transform_func:
            pipeline.add_processor(TransformProcessor(transform_func))
        
        if scaling_method:
            pipeline.add_processor(ScalingProcessor(scaling_method))
        
        if encoding_method:
            pipeline.add_processor(EncodingProcessor(encoding_method))
        
        self.feature_pipelines[feature_name] = pipeline
        return self
    
    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """Fit all feature pipelines."""
        target = data[self.target_column] if self.target_column in data.columns else None
        
        for feature_name, pipeline in self.feature_pipelines.items():
            if feature_name in data.columns:
                print(f"✓ Fitting pipeline for '{feature_name}'") if self.verbose else None
                pipeline.fit(data[feature_name], target=target)
            else:
                warnings.warn(f"Feature '{feature_name}' not found in data")
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform data using fitted pipelines."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        features_to_process = features or list(self.feature_pipelines.keys())
        result_data = data.copy()
        
        for feature_name in features_to_process:
            if feature_name in self.feature_pipelines and feature_name in data.columns:
                print(f"✓ Transforming '{feature_name}'") if self.verbose else None
                
                pipeline = self.feature_pipelines[feature_name]
                transformed = pipeline.transform(data[feature_name])
                
                if isinstance(transformed, pd.DataFrame):
                    # Handle one-hot encoded features
                    result_data = result_data.drop(columns=[feature_name])
                    result_data = pd.concat([result_data, transformed], axis=1)
                else:
                    result_data[feature_name] = transformed
        
        # Split features and target
        y = result_data[self.target_column] if self.target_column in result_data.columns else None
        X = result_data.drop(columns=[self.target_column]) if self.target_column in result_data.columns else result_data
        
        return X, y
    
    def fit_transform(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data, features)
    
    def get_feature_info(self, feature_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about configured features."""
        if feature_name:
            if feature_name in self.feature_pipelines:
                pipeline = self.feature_pipelines[feature_name]
                return {
                    'feature_name': feature_name,
                    'processors': [type(p).__name__ for p in pipeline.processors],
                    'is_fitted': pipeline.is_fitted
                }
            else:
                return {'error': f"No pipeline found for feature '{feature_name}'"}
        else:
            return {
                'configured_features': list(self.feature_pipelines.keys()),
                'fitted_features': [name for name, pipeline in self.feature_pipelines.items() if pipeline.is_fitted]
            }
    
    def save_pipeline(self, filepath: str):
        """Save the preprocessing pipeline."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'target_column': self.target_column,
                'feature_pipelines': self.feature_pipelines,
                'is_fitted': self.is_fitted
            }, f)
        print(f"✓ Pipeline saved to '{filepath}'") if self.verbose else None
    
    def load_pipeline(self, filepath: str):
        """Load a preprocessing pipeline."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.target_column = data['target_column']
        self.feature_pipelines = data['feature_pipelines']
        self.is_fitted = data['is_fitted']
        print(f"✓ Pipeline loaded from '{filepath}'") if self.verbose else None
