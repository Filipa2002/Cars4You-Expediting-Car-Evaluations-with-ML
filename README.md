# Cars 4 You: Expediting Car Evaluations with ML

## Team

*   Filipa Pereira, 20240509
*   Gonçalo Silva, 20250354
*   Marta La Feria, 20211051
*   Tomás Coroa, 20250394

## 1. Project Overview

**Cars 4 You** is an online car resale company facing operational delays due to a high volume of manual mechanic inspections. This bottleneck slows down the car evaluation process, causing potential sellers to turn to competitors.

To address this, our project aimed to develop a robust machine learning model to accurately predict a car's resale price based on a set of features provided by the seller. The primary objective was to create a reliable predictive tool that streamlines the evaluation process, reduces the dependency on prior inspections, and provides fast, data-driven price estimates. Due to the presence of outliers in the dataset, the modeling focus was on **predicting the median price**, using the **Mean AbsoluteError (MAE)** as the primary evaluation metric, which also aligned with the Kaggle competition's requirements.

## 2. Dataset

The project utilizes a dataset containing information about used cars. The data includes the following attributes:

| Attribute          | Description                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| `carID`            | A unique identifier for each car.                                                                       |
| `Brand`            | The car’s main brand (e.g., Ford, Toyota).                                                              |
| `model`            | The car model.                                                                                          |
| `year`             | The year of registration of the car.                                                                    |
| `transmission`     | The type of transmission (e.g., Manual, Automatic).                                                     |
| `mileage`          | The total reported distance traveled by the car (in miles).                                             |
| `fuelType`         | The type of fuel used by the car (e.g., Diesel, Petrol).                                                |
| `tax`              | The amount of road tax (in £) applicable in 2020.                                                       |
| `mpg`              | Average miles per gallon.                                                                               |
| `engineSize`       | The size of the engine in liters.                                                                       |
| `paintQuality%`    | A mechanic’s assessment of the car's paint quality.                                                     |
| `previousOwners`   | The number of previous registered owners.                                                               |
| `hasDamage`        | A boolean marker indicating if the car has damage.                                                      |
| **`price`**        | **(Target)** The car’s price when purchased by Cars 4 You (in £).                                       |

## 3. Methodology

Our approach was structured into several key phases, from data cleaning and preparation to model training and evaluation.

### 3.1. Data Cleaning & Preprocessing

The initial dataset required significant cleaning to ensure data quality and consistency:

*   **Duplicate Removal**: Rows with duplicate feature values (excluding `carID` and `price`) were removed to prevent data leakage and model confusion.
*   **Inconsistency Correction**:
    *   Categorical features like `Brand`, `model`, `transmission`, and `fuelType` contained spelling errors and formatting inconsistencies. These were standardized using **fuzzy string matching** and validated against an external car database via the **wheel-size.com API**.
    *   Logical errors, such as negative values for `mileage`, `tax`, and `engineSize`, were corrected.
*   **Missing Data Imputation**: Missing values, which were determined to be Missing at Random (MAR), were imputed using **K-Nearest Neighbors (KNNImputer)** with k=5.

### 3.2. Feature Engineering & Transformation

To enhance the predictive power of our models, we engineered new features and transformed existing ones:

*   **New Features**:
    *   `age`: Calculated from the `year` of registration to represent the car's age.
    *   `miles_per_year`: Derived from `mileage` and `age` to provide a normalized measure of usage.
*   **Categorical Encoding**:
    *   **Frequency Encoding**: Applied to `Brand` and a combined `brand_model` feature.
    *   **One-Hot Encoding**: Used for `transmission` and `fuelType` to convert them into a numerical format.
*   **Data Scaling**: The **RobustScaler** was used to scale numerical features. This scaler is less sensitive to outliers, which were prevalent in our dataset.

### 3.3. Feature Selection

To reduce model complexity and improve generalization, we employed a hybrid feature selection strategy using an aggregate consensus from multiple methods:

1.  **Filter Methods**:
    *   **Variance Threshold**: Removed features with near-zero variance (`hasDamage`, `transmission_other`, `fuelType_electric`, `fuelType_other`).
    *   **Spearman Correlation**: Identified high correlation between `mileage` and `miles_per_year`, and low correlation with the target for `paintQuality%`, `previousOwners`, and `Brand_freq_enc`.
2.  **Wrapper Methods**:
    *   **Recursive Feature Elimination (RFE)**: Used with both Linear Regression and Random Forest to identify the most impactful features.
3.  **Embedded Methods**:
    *   **Lasso (L1) and Ridge (L2) Regularization**: Used to assess feature importance based on coefficient magnitudes.

Based on a majority vote (features selected by at least 4 out of 5 methods), the final set of **8 features** was chosen for modeling:
`mileage`, `tax`, `mpg`, `engineSize`, `age`, `Brand_freq_enc`, `brand_model_freq_enc`, `transmission_manual`.

### 3.4. Model Training & Evaluation

We benchmarked a variety of regression models to identify the best performer.

*   **Evaluation Metrics**: The primary metric was **Mean Absolute Error (MAE)**. We also monitored **Root Mean Squared Error (RMSE)** and **Pinball Loss (α=0.5)** for a more comprehensive assessment.
*   **Benchmarked Models**:
    *   Linear Regression
    *   ElasticNet
    *   Support Vector Regressor (SVR)
    *   Decision Tree Regressor
    *   Random Forest Regressor
    *   K-Neighbors Regressor
    *   **Gradient Boosting Regressor**

## 4. Results

The models were trained and their performance was compared on the validation set.

| Model               | Validation MAE |
| ------------------- | -------------- |
| **Gradient Boosting** | **1414.71**    |
| Random Forest       | 1425.65        |
| K-Neighbors         | 1632.20        |
| Decision Tree       | 1690.03        |
| SVR                 | 2798.12        |
| ElasticNet          | 3560.88        |
| Linear Regression   | 3584.37        |

The **Gradient Boosting Regressor** achieved the lowest validation MAE and was therefore selected as the final model. While the SVR model showed less overfitting (a smaller gap between training and validation scores), the Gradient Boosting model's superior predictive accuracy made it the strategic choice for the competition.

The final model was retrained on the combined training and validation datasets before generating predictions for the test set.

## 5. How to Run the Project

The project is organized into two main Jupyter notebooks:

1.  `car_evaluations_group37 - part1.ipynb`: Contains the code for Exploratory Data Analysis (EDA), data cleaning, preprocessing, and feature engineering.
2.  `car_evaluations_group37 - part2.ipynb`: Contains the code for feature selection, model benchmarking, training, and final prediction generation.

### Required Libraries

To run the notebooks, you will need the following Python libraries:

```
pandas
numpy
seaborn
matplotlib
scikit-learn
pyarrow
```

You can install them using pip:
`pip install pandas numpy seaborn matplotlib scikit-learn pyarrow`

## 6. Future Work

*   **Advanced Hyperparameter Tuning**: Further fine-tune the Gradient Boosting model using more extensive search techniques like Bayesian Optimization.
*   **Explore Pinball Loss**: Investigate the use of Pinball Loss with different quantiles (α ≠ 0.5) to generate conservative price estimates, which could help minimize financial risk for Cars 4 You.
*   **Alternative Encoders**: Experiment with other categorical encoding techniques like Target Encoding.
