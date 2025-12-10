# Cars 4 You: Expediting Car Evaluations with ML

## Team

*   Filipa Pereira, 20240509
*   Gonçalo Silva, 20250354
*   Marta La Feria, 20211051
*   Tomás Coroa, 20250394

## 1. Project Overview

**Cars 4 You** is an online car resale company facing operational delays due to a high volume of manual mechanic inspections. This bottleneck slows down the car evaluation process, causing potential sellers to turn to competitors.

To address this, our project aimed to develop a robust machine learning model to accurately predict a car's resale price based on a set of features provided by the seller. The primary objective was to create a reliable predictive tool that streamlines the evaluation process, reduces the dependency on prior inspections, and provides fast, data-driven price estimates. Due to the presence of outliers in the dataset, the modeling focus was on **predicting the median price**, using the **Mean Absolute Error (MAE)** as the primary evaluation metric, which also aligned with the Kaggle competition's requirements.

## 2. Dataset

The project utilizes a dataset containing information about used cars. The data includes the following attributes:

| Attribute          | Description                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| `carID`            | A unique identifier for each car.                                                                       |
| `Brand`            | The car's main brand (e.g., Ford, Toyota).                                                              |
| `model`            | The car model.                                                                                          |
| `year`             | The year of registration of the car.                                                                    |
| `transmission`     | The type of transmission (e.g., Manual, Automatic).                                                     |
| `mileage`          | The total reported distance traveled by the car (in miles).                                             |
| `fuelType`         | The type of fuel used by the car (e.g., Diesel, Petrol).                                                |
| `tax`              | The amount of road tax (in £) applicable in 2020.                                                       |
| `mpg`              | Average miles per gallon.                                                                               |
| `engineSize`       | The size of the engine in liters.                                                                       |
| `paintQuality%`    | A mechanic's assessment of the car's paint quality.                                                     |
| `previousOwners`   | The number of previous registered owners.                                                               |
| `hasDamage`        | A boolean marker indicating if the car has damage.                                                      |
| **`price`**        | **(Target)** The car's price when purchased by Cars 4 You (in £).                                       |

## 3. Methodology

Our approach was structured into several key phases, from data cleaning and preparation to model training and evaluation.

### 3.1. Data Cleaning & Preprocessing

The initial dataset required significant cleaning to ensure data quality and consistency:

*   **Duplicate Removal**: Rows with duplicate feature values (excluding `carID` and `price`) were removed to prevent data leakage and model confusion.
*   **Inconsistency Correction**:
    *   Categorical features like `Brand`, `model`, `transmission`, and `fuelType` contained spelling errors and formatting inconsistencies. These were standardized using **fuzzy string matching** and validated against an external car database via the **wheel-size.com API**.
    *   Logical errors, such as negative values for `mileage`, `tax`, and `engineSize`, were corrected.
*   **Missing Data Imputation**: Missing values, which were determined to be Missing at Random (MAR), were imputed using **MICE (Multiple Imputation by Chained Equations)** through scikit-learn's `IterativeImputer` with 10 iterations and a BayesianRidge estimator.

### 3.2. Feature Engineering & Transformation

To enhance the predictive power of our models, we engineered new features and transformed existing ones:

*   **New Features**:
    *   `age`: Calculated from the `year` of registration to represent the car's age.
    *   `miles_per_year`: Derived from `mileage` and `age` to provide a normalized measure of usage.
    *   `brand_segment`: Categorical classification of brands into luxury, mid-range, and budget segments.
    *   Interaction terms between vehicle characteristics.
*   **Categorical Encoding**:
    *   **Target Encoding**: Applied to high-cardinality features (`Brand`, `model`, `brand_model`) using K-fold cross-validation to prevent data leakage. This approach captures the actual price relationships rather than just occurrence patterns.
    *   **One-Hot Encoding**: Used for low-cardinality categoricals (`transmission`, `fuelType`).
*   **Data Scaling**: The **RobustScaler** was used to scale numerical features. This scaler is less sensitive to outliers, which were prevalent in our dataset.
*   **Power Transformation**: Applied to normalize skewed distributions.

### 3.3. Feature Selection

To reduce model complexity and improve generalization, we employed a hybrid feature selection strategy using an aggregate consensus from multiple methods:

1.  **Filter Methods**:
    *   **Variance Threshold**: Removed features with near-zero variance.
    *   **Spearman Correlation**: Identified highly correlated features and low correlation with target.
2.  **Wrapper Methods**:
    *   **Recursive Feature Elimination (RFE)**: Used with both Linear Regression and Random Forest to identify the most impactful features.
3.  **Embedded Methods**:
    *   **Lasso (L1) and Ridge (L2) Regularization**: Used to assess feature importance based on coefficient magnitudes.

Based on a majority vote (features selected by multiple methods), the final feature set was chosen for modeling.

### 3.4. Model Training & Evaluation

We benchmarked a variety of regression models to identify the best performers.

*   **Evaluation Metrics**: The primary metric was **Mean Absolute Error (MAE)**. We also monitored **Root Mean Squared Error (RMSE)** and **R² Score** for a more comprehensive assessment.
*   **Log Transformation**: The target variable was log-transformed (`np.log1p`) during training to reduce the impact of high-price outliers and stabilize variance. Predictions were inverse-transformed (`np.expm1`) before evaluation.
*   **Benchmarked Models** (minimum of 5 as required):
    *   Gradient Boosting Regressor
    *   Random Forest Regressor
    *   Extra Trees Regressor
    *   K-Neighbors Regressor
    *   Elastic Net

### 3.5. Ensemble Methods

To maximize predictive performance, we implemented two ensemble strategies:

1.  **Weighted Ensemble**: Combined predictions from individual models using optimized weights determined through grid search on validation data.
2.  **Stacking Ensemble**: Used a `StackingRegressor` with all five base models and a Ridge meta-learner with 5-fold cross-validation.

## 4. Results

The models were trained and their performance was compared on the validation set.

| Model               | Validation MAE |
| ------------------- | -------------- |
| Gradient Boosting   | Best performer |
| Random Forest       | Strong         |
| Extra Trees         | Good           |
| K-Neighbors         | Moderate       |
| Elastic Net         | Baseline       |

**Ensemble Results:**
| Ensemble Method     | Validation MAE |
| ------------------- | -------------- |
| Weighted Ensemble   | ~£1,200        |
| **Stacking Ensemble** | **~£1,150**  |

The **Stacking Ensemble** achieved the best validation MAE and was selected as the final model for Kaggle submission. The stacking approach benefits from the meta-learner's ability to learn optimal combinations of base model predictions.

The final model was retrained on the combined training and validation datasets before generating predictions for the test set.

## 5. How to Run the Project

The project is organized into two main Jupyter notebooks:

1.  `car_evaluations_group37 - part1.ipynb`: Contains the code for Exploratory Data Analysis (EDA), data cleaning, preprocessing, and feature engineering.
2.  `car_evaluations_group37 - part2.ipynb`: Contains the code for feature selection, model benchmarking, ensemble methods, and final prediction generation.

### Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
wordcloud>=1.9.0
pyarrow>=12.0.0
```

### Execution Order

1. Run Part 1 notebook to preprocess data (exports parquet files)
2. Run Part 2 notebook to train models and generate Kaggle submissions