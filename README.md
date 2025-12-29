# Appliance Energy Consumption Prediction

## Overview
This project analyzes and models household appliance energy consumption using environmental sensor data from a smart home.  
The objective is to identify key drivers of energy usage and establish a strong, interpretable regression baseline suitable for extension to more advanced machine learning models.

The project emphasizes:
- Clean project structure and reproducible workflows
- Exploratory data analysis (EDA)
- Baseline and regularized regression modeling
- Model evaluation and interpretability

---

## Dataset
The dataset contains time-series sensor readings collected from a residential environment, including:

- Indoor temperature sensors (T1–T6)
- Indoor and outdoor humidity
- Outdoor weather conditions
- Lighting usage
- Appliance energy consumption (target variable)

**Target variable:**
`Appliances` — log-transformed using `log1p` to reduce skew and stabilize variance.


---

## Project Structure
project_root/
|-- data/
|   `-- appliances_energy.csv
|-- src/
|   |-- explore.py        # EDA and correlation analysis
|   `-- regression.py    # Feature engineering and modeling
|-- main.py               # Entry point
`-- README.md

---


## Feature Selection
Features were selected based on exploratory analysis and domain intuition:

- `lights` (proxy for occupancy)
- Indoor temperatures: `T1`, `T2`, `T3`, `T4`, `T6`
- Indoor humidity: `RH_1`
- Outdoor temperature: `T_out`
- Wind speed: `Windspeed`

All features are standardized prior to model training.

---

## Models
### Baseline Model
A naive baseline that predicts the mean of the training target.

### Linear Regression
A standard linear regression model trained on standardized features.

### Ridge Regression
A regularized linear model used to assess coefficient stability and overfitting.

---

## Results
| Model              | RMSE (log scale) |
|-------------------|-----------------|
| Baseline          | 0.6467          |
| Linear Regression | 0.5971          |
| Ridge Regression  | 0.5971          |

**Key observations:**
- Linear regression meaningfully outperforms the baseline.
- Ridge regularization does not improve RMSE, suggesting limited overfitting.
- Remaining error is likely driven by noise or non-linear relationships in the data.

---

## Model Interpretation
Because all features are standardized, model coefficients are directly comparable.

Key insights:
- Indoor temperature sensors (particularly `T6` and `T3`) are the strongest predictors of appliance energy usage.
- Lighting usage shows a strong positive relationship, likely acting as a proxy for occupancy.
- Outdoor temperature exhibits a negative relationship with energy consumption, consistent with HVAC regulation behavior.
- Humidity and wind-related variables contribute relatively limited predictive power.

Residual analysis shows no strong systematic patterns, supporting the suitability of a linear baseline.

---

## Takeaways
- Simple, interpretable models can significantly outperform naive baselines.
- Thoughtful feature selection and domain knowledge are critical in applied data science.
- This project provides a solid foundation for extending into non-linear and time-series models.

---

## Future Work
Potential extensions include:
- Tree-based models (Random Forest, Gradient Boosting)
- Time-series feature engineering (lags, rolling statistics)
- Cross-validation and hyperparameter tuning
- Model deployment as a lightweight prediction service

---

## Technologies Used
- Python
- pandas, NumPy
- scikit-learn
- matplotlib

---

## How to Run
```bash
python main.py