
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def analyze_missing_data(df, dataset_name="Dataset"):
    """Comprehensive missing data analysis"""
    print(f"=== {dataset_name.upper()} MISSING DATA ANALYSIS ===")
    
    # Overall missing data stats
    total_rows = len(df)
    rows_with_nan = df.isnull().any(axis=1).sum()
    print(f"Total rows: {total_rows:,}")
    print(f"Rows with at least one NaN: {rows_with_nan:,} ({rows_with_nan/total_rows*100:.2f}%)")
    
    # Feature-wise missing data
    missing_stats = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    }).sort_values('Missing_Percentage', ascending=False)
    
    missing_stats = missing_stats[missing_stats['Missing_Count'] > 0]
    
    if len(missing_stats) > 0:
        print(f"\n=== FEATURES WITH MISSING VALUES ===")
        print(missing_stats)
        
        # Visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        missing_stats['Missing_Percentage'].plot(kind='barh')
        plt.title(f'{dataset_name} - Missing Data Percentage')
        plt.xlabel('Percentage Missing')
        
        plt.subplot(1, 2, 2)
        missing_stats['Missing_Count'].plot(kind='barh')
        plt.title(f'{dataset_name} - Missing Data Count')
        plt.xlabel('Count Missing')
        plt.tight_layout()
        plt.show()
    else:
        print("No missing values found!")
    
    return missing_stats


def analyze_numerical_feature(df:pd.DataFrame, feature_name:str,target_col:str='price', correlate:bool = True)->dict:
    """Analyze a numerical feature in the dataframe.
    Provides statistics, data quality assessment, visualizations, and transformation suggestions.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        feature_name (str): The name of the numerical feature to analyze.
        target_col (str): The target column name for correlation analysis.
        correlate (bool): Whether to compute correlation with the target column.
        
    Returns:
        dict: A dictionary containing basic statistics of the feature.
    
    Plots:
        - Distribution plot
        - Box plot
        - Q-Q plot
        - Scatter plot with target (if correlate=True)
    """
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {feature_name.upper()}")
    print(f"{'='*60}")
    
    feature_data = df[feature_name].dropna()
    
    # Basic Statistics
    stats_dict = {
        'Count': len(feature_data),
        'Missing': df[feature_name].isnull().sum(),
        'Missing %': (df[feature_name].isnull().sum() / len(df)) * 100,
        'Min': feature_data.min(),
        'Max': feature_data.max(),
        'Mean': feature_data.mean(),
        'Median': feature_data.median(),
        'Std': feature_data.std(),
        'Skewness': feature_data.skew(),
        'Kurtosis': feature_data.kurtosis()
    }
    
    print("=== BASIC STATISTICS ===")
    for key, value in stats_dict.items():
        if key in ['Missing %', 'Mean', 'Median', 'Std', 'Skewness', 'Kurtosis']:
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Data Quality Issues
    print("\n=== DATA QUALITY ASSESSMENT ===")
    negative_count = (feature_data < 0).sum()
    zero_count = (feature_data == 0).sum()
    
    print(f"Negative values: {negative_count} ({negative_count/len(feature_data)*100:.2f}%)")
    print(f"Zero values: {zero_count} ({zero_count/len(feature_data)*100:.2f}%)")
    
    # Outlier Analysis
    Q1 = feature_data.quantile(0.25)
    Q3 = feature_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
    print(f"Outliers (IQR method): {len(outliers)} ({len(outliers)/len(feature_data)*100:.2f}%)")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribution plot
    axes[0,0].hist(feature_data, bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(feature_data.mean(), color='red', linestyle='--', label='Mean')
    axes[0,0].axvline(feature_data.median(), color='green', linestyle='--', label='Median')
    axes[0,0].set_title(f'{feature_name} Distribution')
    axes[0,0].legend()
    
    # Box plot
    axes[0,1].boxplot(feature_data)
    axes[0,1].set_title(f'{feature_name} Box Plot')
    axes[0,1].set_ylabel(feature_name)
    
    # QQ plot for normality
    stats.probplot(feature_data, dist="norm", plot=axes[1,0])
    axes[1,0].set_title(f'{feature_name} Q-Q Plot')
    
    # Correlation with target (if available)
    if target_col in df.columns and correlate:
        target_data = df[target_col].dropna()
        feature_target = df[[feature_name, target_col]].dropna()
        
        correlation = feature_target[feature_name].corr(feature_target[target_col])
        axes[1,1].scatter(feature_target[feature_name], feature_target[target_col], alpha=0.5)
        axes[1,1].set_xlabel(feature_name)
        axes[1,1].set_ylabel(target_col)
        axes[1,1].set_title(f'{feature_name} vs {target_col}\nCorrelation: {correlation:.3f}')
    
    plt.tight_layout()
    plt.show()
    
    # Transformation Suggestions
    print("\n=== TRANSFORMATION SUGGESTIONS ===")
    
    if negative_count > 0:
        print("⚠️  Contains negative values - consider absolute transformation or offset")
    
    if stats_dict['Skewness'] > 1:
        print("⚠️  Highly right-skewed - consider log transformation")
    elif stats_dict['Skewness'] < -1:
        print("⚠️  Highly left-skewed - consider power transformation")
    
    if len(outliers) > len(feature_data) * 0.05:  # More than 5% outliers
        print("⚠️  High percentage of outliers - consider outlier treatment")
    
    if zero_count > len(feature_data) * 0.1:  # More than 10% zeros
        print("⚠️  Many zero values - consider log1p transformation")
    
    return stats_dict

def suggest_transformations(df:pd.DataFrame, feature_name:str)->dict:
    """
    Suggest and visualize different transformations for a numerical feature.
    Args:
        df (pd.DataFrame): The input dataframe.
        feature_name (str): The name of the numerical feature to transform.
    Returns:
        dict: A dictionary with transformed data series.
    
    Plots:
        - Histograms and Q-Q plots for each transformation.    
    """
    
    print(f"\n=== TRANSFORMATION EXPERIMENTS FOR {feature_name.upper()} ===")
    
    original_data = df[feature_name].dropna()
    
    # Try different transformations
    transformations = {}
    
    # Log transformation (handling negative/zero values)
    if (original_data <= 0).any():
        transformations['log1p_abs'] = np.log1p(np.abs(original_data))
        transformations['log1p_shifted'] = np.log1p(original_data + abs(original_data.min()) + 1)
    else:
        transformations['log'] = np.log(original_data)
        transformations['log1p'] = np.log1p(original_data)
    
    # Square root (for positive values)
    if (original_data >= 0).all():
        transformations['sqrt'] = np.sqrt(original_data)
    
    # Box-Cox (for positive values)
    if (original_data > 0).all():
        transformed_data, lambda_param = stats.boxcox(original_data)
        transformations[f'boxcox_λ={lambda_param:.3f}'] = transformed_data
    
    # Compare transformations
    fig, axes = plt.subplots(2, len(transformations) + 1, figsize=(5 * (len(transformations) + 1), 8))
    
    # Original data
    axes[0,0].hist(original_data, bins=30, alpha=0.7)
    axes[0,0].set_title('Original')
    stats.probplot(original_data, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Original Q-Q')
    
    # Transformed data
    for i, (name, transformed) in enumerate(transformations.items(), 1):
        axes[0,i].hist(transformed, bins=30, alpha=0.7)
        axes[0,i].set_title(name)
        stats.probplot(transformed, dist="norm", plot=axes[1,i])
        axes[1,i].set_title(f'{name} Q-Q')
        
        # Print skewness comparison
        original_skew = original_data.skew()
        transformed_skew = pd.Series(transformed).skew()
        print(f"{name}: Skewness {original_skew:.3f} → {transformed_skew:.3f}")
    
    plt.tight_layout()
    plt.show()
    
    return transformations




def analyze_categorical_feature(df:pd.DataFrame,feature_name:str, top_n:int=20):
    """
    Anallyzes categorical features to find 

    Args:
        df()
        feature_name (_type_): _description_
        top_n (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """
    
    
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {feature_name.upper()}")
    print(f"{'='*60}")
    
    # Basic statistics
    total_count = len(df)
    missing_count = df[feature_name].isnull().sum()
    unique_count = df[feature_name].nunique()
    
    print("=== BASIC STATISTICS ===")
    print(f"Total records: {total_count:,}")
    print(f"Missing values: {missing_count:,} ({missing_count/total_count*100:.2f}%)")
    print(f"Unique values: {unique_count:,}")
    print(f"Cardinality ratio: {unique_count/total_count:.3f}")
    
    # Value counts
    value_counts = df[feature_name].value_counts(dropna=False)
    
    print(f"\n=== TOP {min(top_n, len(value_counts))} VALUES ===")
    print(value_counts.head(top_n))
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of top categories
    top_categories = value_counts.head(top_n)
    axes[0].bar(range(len(top_categories)), top_categories.values)
    axes[0].set_xticks(range(len(top_categories)))
    axes[0].set_xticklabels(top_categories.index, rotation=45, ha='right')
    axes[0].set_title(f'Top {min(top_n, len(top_categories))} {feature_name} Categories')
    axes[0].set_ylabel('Count')
    
    # Pie chart for top categories
    if len(top_categories) <= 10:
        axes[1].pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%', startangle=90)
        axes[1].set_title(f'{feature_name} Distribution')
    else:
        # If too many categories, show top 10 + others
        top_10 = top_categories.head(10)
        others_sum = top_categories.iloc[10:].sum()
        
        pie_data = list(top_10.values) + [others_sum]
        pie_labels = list(top_10.index) + ['Others']
        
        axes[1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        axes[1].set_title(f'{feature_name} Distribution (Top 10 + Others)')
    
    plt.tight_layout()
    plt.show()
    
    # Data quality assessment
    print("\n=== DATA QUALITY ASSESSMENT ===")
    
    # Check for potential spelling variations
    if unique_count > 10:  # Only for high cardinality features
        print("⚠️  High cardinality feature - check for spelling variations")
        
        # Look for similar strings (basic check)
        values_lower = df[feature_name].dropna().str.lower().value_counts()
        original_unique = df[feature_name].dropna().nunique()
        lowercase_unique = len(values_lower)
        
        if lowercase_unique < original_unique:
            print(f"⚠️  Case sensitivity issues: {original_unique} → {lowercase_unique} unique values when lowercased")
    
    # Check for very rare categories
    rare_threshold = 0.01  # Less than 1%
    rare_categories = value_counts[value_counts < total_count * rare_threshold]
    
    if len(rare_categories) > 0:
        print(f"⚠️  {len(rare_categories)} rare categories (< {rare_threshold*100}% of data)")
        print(f"Consider grouping into 'Other' category")
    
    # Encoding suggestions
    print("\n=== ENCODING SUGGESTIONS ===")
    
    if unique_count == 2:
        print("✓ Binary feature - use Label Encoding (0/1)")
    elif unique_count <= 10:
        print("✓ Low cardinality - use One-Hot Encoding")
    elif unique_count <= 50:
        print("⚠️  Medium cardinality - consider Target Encoding or Ordinal Encoding")
    else:
        print("⚠️  High cardinality - consider:")
        print("  - Target/Mean Encoding")
        print("  - Feature hashing")
        print("  - Dimensionality reduction")
        print("  - Grouping rare categories")
    
    return value_counts


def analyze_boolean_feature(df:pd.DataFrame, feature_name:str):
    """
    Analyzes a boolean feature in the dataframe.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        feature_name (str): The name of the boolean feature to analyze.
        
    Returns:
        dict: A dictionary containing basic statistics of the feature.
    """
    
    print(f"\n{'='*60}")
    print(f"ANALYZING BOOLEAN FEATURE: {feature_name.upper()}")
    print(f"{'='*60}")
    
    feature_data = df[feature_name].dropna()
    
    # Basic Statistics
    true_count = feature_data.sum()
    false_count = len(feature_data) - true_count
    missing_count = df[feature_name].isnull().sum()
    
    stats_dict = {
        'Total Count': len(feature_data),
        'Missing': missing_count,
        'Missing %': (missing_count / len(df)) * 100,
        'True Count': true_count,
        'True %': (true_count / len(feature_data)) * 100,
        'False Count': false_count,
        'False %': (false_count / len(feature_data)) * 100
    }
    
    print("=== BASIC STATISTICS ===")
    for key, value in stats_dict.items():
        if ' %' in key:
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value}")
    
    # Visualization
    plt.figure(figsize=(6, 6))
    plt.bar(['True', 'False'], [true_count, false_count], color=['green', 'red'])
    plt.title(f'{feature_name} Distribution')
    plt.ylabel('Count')
    plt.show()
    
    return stats_dict