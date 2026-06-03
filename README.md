# Traffic Demand Prediction - ML Solution

## Project Overview

This project implements a machine learning solution to predict traffic demand at various locations and timestamps using the provided dataset. The system analyzes transportation patterns and geographic factors to forecast demand, which can help cities optimize traffic management and reduce congestion.

## Dataset Description

### Files
- **train.csv**: 77,299 records with 11 columns (includes target variable `demand`)
- **test.csv**: 41,778 records with 10 columns (for prediction)
- **sample_submission.csv**: Sample format for submission (5 x 2)

### Features
- **Index**: Unique identification of datapoint
- **geohash**: Geographic information about the location
- **day**: Day when the information is recorded
- **timestamp**: Timestamp of the record (HH:MM format)
- **RoadType**: Type of road (Residential, Street, Highway)
- **NumberofLanes**: Number of roads/lanes at the location
- **LargeVehicles**: Whether large vehicles are permitted (Allowed/Not Allowed)
- **Landmarks**: Whether landmarks are near the location (Yes/No)
- **Temperature**: Temperature at the location (°C)
- **Weather**: Weather condition (Sunny, Rainy, Foggy, Snowy)
- **demand**: Traffic demand (Target Variable)

## Evaluation Metric

**Score = max(0, 100 × R² Score)**

Where R² measures how well the predictions explain the variance in actual demand.

## Solution Architecture

### 1. **Data Preprocessing & Cleaning**
- Handled missing values:
  - Temperature: 2,495 missing in train, 1,349 in test
  - RoadType: 600 missing in train, 324 in test
  - Weather: 797 missing in train, 431 in test
- Filled missing values with median (for numerical) or "Unknown" (for categorical)
- Verified no duplicate records

### 2. **Feature Engineering**
Created 13 features for modeling:
- **Temporal Features**: 
  - `hour`: Hour extracted from timestamp
  - `minute`: Minute extracted from timestamp
  - `hour_sin`, `hour_cos`: Cyclical encoding of hour (0-23 as circular)

- **Encoded Categorical Features**:
  - `RoadType_encoded`: LabelEncoded road type
  - `Weather_encoded`: LabelEncoded weather condition
  - `LargeVehicles_encoded`: Binary (1 if Allowed, 0 otherwise)
  - `Landmarks_encoded`: Binary (1 if Yes, 0 otherwise)

- **Original Numerical Features**:
  - `day`: Day of measurement
  - `NumberofLanes`: Count of lanes
  - `Temperature`: Temperature value

- **Interaction Features**:
  - `temp_hour_interaction`: Temperature × sin(hour)
  - `lanes_vehicles_interaction`: NumberofLanes × LargeVehicles

### 3. **Feature Normalization**
- Applied StandardScaler to normalize all features
- Ensures equal contribution from features with different scales

### 4. **Model Selection & Comparison**

Trained and evaluated 6 regression models using 5-fold cross-validation:

| Model | Mean CV R² | Std Dev |
|-------|-----------|---------|
| Linear Regression | 0.1992 | 0.0410 |
| Ridge Regression | 0.1992 | 0.0410 |
| Lasso Regression | -0.0135 | 0.0081 |
| **Random Forest** | **0.7092** | **0.0295** |
| **Gradient Boosting** | **0.7385** ✓ | **0.0269** |
| XGBoost | 0.7141 | 0.0219 |

**Winner**: Gradient Boosting with mean CV R² = **0.7385 (73.85/100)**

### 5. **Hyperparameter Optimization**

Performed GridSearchCV on Gradient Boosting with parameters:
- `n_estimators`: [100, 200]
- `learning_rate`: [0.01, 0.05, 0.1]
- `max_depth`: [3, 5, 7]

**Optimal Parameters**:
- n_estimators: 100
- learning_rate: 0.05
- max_depth: 5
- **Best CV R² Score**: 0.7436 (74.36/100)

### 6. **Predictions & Submission**

Generated predictions on 41,778 test records with the following statistics:
- Min Demand: 0.0556
- Max Demand: 0.8510
- Mean Demand: 0.1391
- Median Demand: 0.0632
- Std Dev: 0.1720

## Files Generated

- **traffic_demand_prediction.ipynb**: Complete Jupyter notebook with:
  - Data exploration and visualization
  - Feature engineering pipeline
  - Model training and evaluation
  - Hyperparameter tuning
  - Prediction generation

- **dataset/submission.csv**: Final predictions in required format
  - Dimensions: 41,778 × 2 (Index, demand)
  - Format matches sample_submission.csv
  - Ready for submission

## Key Insights

1. **Best Performing Features**: NumberofLanes showed strong correlation (0.212) with demand
2. **Model Performance**: Ensemble methods (GB, RF, XGBoost) significantly outperformed linear models
3. **Gradient Boosting Advantage**: Better generalization with regularization through learning rate and depth tuning
4. **Feature Interactions**: Temporal and environmental interactions improve predictions

## How to Run

1. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost
   ```

2. **Run the notebook**:
   ```bash
   jupyter notebook traffic_demand_prediction.ipynb
   ```

3. **Output**: Submission file will be saved to `dataset/submission.csv`

## Results Summary

- **Training R² Score**: 0.7617 (76.17/100)
- **Cross-Validation R² Score**: 0.7436 (74.36/100)
- **Submission File**: Ready with 41,778 demand predictions

The model explains approximately 74% of the variance in traffic demand, providing valuable insights for urban traffic management and congestion prediction.
# Traffic-Demand-Prediction
