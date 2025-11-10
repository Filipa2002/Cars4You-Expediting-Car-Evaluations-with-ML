from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.metrics import ( # type: ignore
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, 
    mean_squared_log_error, explained_variance_score
)

def get_regression_metrics(y_true:np.ndarray, y_pred:np.ndarray) -> pd.DataFrame:
    """Calculates and formats key regression metrics into a Pandas DataFrame."""
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    
    # Structure the report
    report_data = {
        'Metric': [
            'R-squared ($R^2$)', 
            'Explained Variance Score',
            'Root Mean Squared Error (RMSE)', 
            'Mean Absolute Error (MAE)', 
            'Median Absolute Error (MedAE)'
        ],
        'Score': [r2, evs, rmse, mae, medae],
        'Interpretation': [
            'Proportion of variance explained by the model (closer to 1 is better).',
            'The variance in the error, a lower value is better.',
            'Average magnitude of errors (same units as target).',
            'Average absolute difference between true and predicted values.',
            'The median of all absolute errors (less sensitive to outliers).'
        ]
    }
    
    report_df = pd.DataFrame(report_data).set_index('Metric')
    report_df['Score'] = report_df['Score'].map('{:,.4f}'.format)
    
    return report_df

def plot_regression_diagnostics(y_true:np.ndarray, y_pred:np.ndarray, title:str = "Regression Diagnostics", save_img:bool = False) -> None:
    """Generates diagnostic plots for regression analysis.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        title (str): Title for the plots.
    
    Returns:
        None:
    
    Plots:
        1. Predicted vs. True Values Scatter Plot
        2. Distribution of Residuals (Errors) Histogram/KDE Plot
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16)
    # 1. True vs. Predicted Plot
    sns.scatterplot(x=y_true, y=y_pred, ax=axes[0])
    # Add a perfect prediction line (y=x)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
    
    axes[0].set_title('Predicted vs. True Values', fontsize=14)
    axes[0].set_xlabel('True Values ($Y_{true}$)')
    axes[0].set_ylabel('Predicted Values ($\hat{Y}$)')
    axes[0].legend()
    
    # 2. Error Distribution (Residuals Histogram/KDE)
    sns.histplot(residuals, kde=True, ax=axes[1], bins=30, color='skyblue')
    axes[1].axvline(0, color='red', linestyle='--', label='Zero Error')
    
    axes[1].set_title('Distribution of Residuals (Errors)', fontsize=14)
    axes[1].set_xlabel('Residual ($Y_{true} - \hat{Y}$)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    
    if save_img:
        fig.savefig(f"{title.replace(' ', '_').lower()}_diagnostics.png", dpi=300)