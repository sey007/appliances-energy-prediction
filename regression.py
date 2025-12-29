from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt




def load_data():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    csv_path = data_dir / "appliances_energy.csv"
    return pd.read_csv(csv_path)


def get_features_and_target(df):
    target = "Appliances"

    features = [
        "lights",
        "T1", "T2", "T3", "T4", "T6",
        "RH_1",
        "T_out",
        "Windspeed"
    ]

    X = df[features]
    y = np.log1p(df[target])  # log-transform target

    return X, y

def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    return rmse

# Linear regression achieved an RMSE of ~0.60 on the log-transformed target.
# Ridge regularization did not materially improve performance, suggesting limited overfitting
# and that model error is likely driven by noise or non-linear effects in the data.

def train_ridge_model(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)  # Alpha is the regularization strength
    model.fit(X_train, y_train)
    return model

def plot_residuals(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    # Compute residuals
    residuals = y_test - predictions

    # Plot residuals
    plt.scatter(predictions, residuals)
    plt.axhline(0, color='r', linestyle='--')  # Zero error line
    plt.xlabel("Predicted log(Appliances + 1)")
    plt.ylabel("Residuals (log scale)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig("residuals.png")
    plt.close()


def print_coefficients(model, feature_names):
    for name, coef in zip(feature_names, model.coef_):
        print(f"{name}: {coef:.4f}")

def run_regression():
    df = load_data()
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    model_linear = train_linear_model(X_train_scaled, y_train)
    rmse_linear = evaluate_model(model_linear, X_test_scaled, y_test)

    model_ridge = train_ridge_model(X_train_scaled, y_train)
    rmse_ridge = evaluate_model(model_ridge, X_test_scaled, y_test)

    baseline_prediction = np.full_like(y_test, y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_prediction))
    print(f"Baseline RMSE:          {baseline_rmse:.4f}")
    print(f"Linear Regression RMSE: {rmse_linear:.4f}")
    print(f"Ridge Regression RMSE:  {rmse_ridge:.4f}")
    print("\nLinear model coefficients:")
    print_coefficients(model_linear, X.columns)

    print("\nInterpretation:")
    print("Indoor temperature sensors and lighting usage are the strongest predictors of appliance energy consumption.")
    print("Outdoor temperature has a negative relationship, likely reflecting HVAC regulation effects.")
    
    plot_residuals(model_ridge, X_test_scaled, y_test)






if __name__ == "__main__":
    run_regression()
