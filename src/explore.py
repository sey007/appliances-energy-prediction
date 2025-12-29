# The goal of this file is to explore the dataset to narrow the data down.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_exploration():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    csv_path = data_dir / "appliances_energy.csv"

    df = pd.read_csv(csv_path)

    target = "Appliances"

    print(df.head())
    df.info()

    print("Rows with missing values:", df.isnull().any(axis=1).sum())
    print("Target dtype:", df[target].dtype)

    # Log-transform for visualization
    log_target = np.log1p(df[target])

    log_target.hist(bins=50)
    plt.title("Log-transformed Appliances Energy Usage")
    plt.show()

    # Correlations
    corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    print("\nTop correlations with target:")
    print(corr.head(10))
    print("\nBottom correlations with target:")
    print(corr.tail(10))


if __name__ == "__main__":
    run_exploration()

